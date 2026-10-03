import json
import operator
import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from rag_service import LM_STUDIO_BASE_URL, RAGService

AGENT_MODEL = os.getenv(
    "AGENT_MODEL",
    "lmstudio-community/Qwen3-4B-Instruct-2507-GGUF",
)
MAX_AGENT_TURNS = 6
MAX_CHAT_HISTORY = 12

AGENT_SYSTEM_PROMPT = """You are a helpful assistant for searching a user's local book collection.
The application has already searched the collection before this turn. Use the retrieved context
below as evidence for questions about books. You may call the retrieval tools for more or more
specific passages, or use the document outline tool to discover available books and sections.
Never claim that you cannot access a named file if retrieved passages from it are provided. Base
factual answers on retrieved passages, preserve uncertainty, and do not invent book contents.
Cite supporting sources by filename and section. If retrieval returns no useful passages, say so
clearly and do not imply you read the file. If the user names a book without asking a specific
question, give a brief overview based on the retrieved passages and offer to answer a follow-up."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    sources: Annotated[list[dict[str, Any]], operator.add]
    retrieved_context: str
    steps: int


class RetrievalAgent:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.tools = self._create_tools()
        self.tools_by_name = {retrieval_tool.name: retrieval_tool for retrieval_tool in self.tools}
        self.chat_model = ChatOpenAI(
            model=AGENT_MODEL,
            base_url=LM_STUDIO_BASE_URL,
            api_key="lm-studio",
            temperature=0,
            timeout=300,
        )
        self.tool_calling_model = self.chat_model.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _create_tools(self) -> list[BaseTool]:
        @tool
        def search_book_passages(
            query: str,
            limit: int = 5,
            filename: str = "",
        ) -> str:
            """Search relevant passages, optionally restricted to an exact indexed filename."""
            results = self.rag_service.search(
                query,
                max(1, min(limit, 8)),
                filename.strip() or None,
            )
            return json.dumps(results, ensure_ascii=False)

        @tool
        def get_document_outline(filename: str = "") -> str:
            """List indexed books and their section hierarchy; optionally filter by exact filename."""
            outline = self.rag_service.document_outline(filename.strip() or None)
            return json.dumps(outline[:200], ensure_ascii=False)

        return [search_book_passages, get_document_outline]

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._call_agent)
        graph.add_node("retrieve", self._retrieve_before_agent)
        graph.add_node("tools", self._run_tools)
        graph.add_node("finalize", self._finalize)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "tools": "tools",
                "finalize": "finalize",
                "end": END,
            },
        )
        graph.add_edge("tools", "agent")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _retrieve_before_agent(self, state: AgentState) -> dict[str, Any]:
        user_message = next(
            (
                item.content
                for item in reversed(state["messages"])
                if isinstance(item, HumanMessage) and isinstance(item.content, str)
            ),
            "",
        )
        if not user_message.strip():
            return {"retrieved_context": "", "sources": []}

        available_files = {
            item["filename"]
            for item in self.rag_service.document_outline()
            if item.get("filename")
        }
        lowered_query = user_message.casefold()
        mentioned_files = sorted(
            (
                filename
                for filename in available_files
                if filename.casefold() in lowered_query
            ),
            key=len,
            reverse=True,
        )
        filename_filter = mentioned_files[0] if mentioned_files else None
        results = self.rag_service.search(
            user_message,
            limit=5,
            filename=filename_filter,
        )

        if not results:
            return {
                "retrieved_context": (
                    "No matching passages were found in the indexed collection. "
                    "Do not claim that a named document has been read."
                ),
                "sources": [],
            }

        context = json.dumps(
            [
                {
                    "filename": result["filename"],
                    "title": result["title"],
                    "section": result["section"],
                    "header_path": result["header_path"],
                    "summary": result["summary"],
                    "content": result["content"],
                }
                for result in results
            ],
            ensure_ascii=False,
        )
        return {"retrieved_context": context, "sources": results}

    def _call_agent(self, state: AgentState) -> dict[str, Any]:
        messages = list(state["messages"])
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(
                content=(
                    f"{messages[0].content}\n\nRetrieved context for this turn:\n"
                    f"{state['retrieved_context']}"
                )
            )
        with self.rag_service.loaded_chat_model(AGENT_MODEL):
            response = self.tool_calling_model.invoke(messages)
        return {"messages": [response], "steps": state["steps"] + 1}

    @staticmethod
    def _route_after_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools" if state["steps"] < MAX_AGENT_TURNS else "finalize"
        return "end"

    def _run_tools(self, state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("The agent expected an assistant message with tool calls.")

        tool_messages = []
        collected_sources = []
        for tool_call in last_message.tool_calls:
            retrieval_tool = self.tools_by_name.get(tool_call["name"])
            if retrieval_tool is None:
                raise ValueError(f"The agent requested an unavailable tool: {tool_call['name']}")

            tool_output = retrieval_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                )
            )
            if tool_call["name"] == "search_book_passages":
                search_results = json.loads(tool_output)
                collected_sources.extend(
                    result
                    for result in search_results
                    if isinstance(result, dict) and result.get("filename")
                )

        return {"messages": tool_messages, "sources": collected_sources}

    def _finalize(self, state: AgentState) -> dict[str, Any]:
        messages = list(state["messages"])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            messages.pop()
        messages.insert(
            1,
            SystemMessage(
                content=(
                    "Do not call more tools. Answer now using the retrieved passages and "
                    "conversation context already available. Cite any source filenames and "
                    "sections you used, and state if the evidence is incomplete."
                )
            ),
        )
        with self.rag_service.loaded_chat_model(AGENT_MODEL):
            response = self.chat_model.invoke(messages)
        return {"messages": [response]}

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        previous_messages: list[AnyMessage] = []
        for turn in (history or [])[-MAX_CHAT_HISTORY:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                previous_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                previous_messages.append(AIMessage(content=content))

        result = self.graph.invoke(
            {
                "messages": [
                    SystemMessage(content=AGENT_SYSTEM_PROMPT),
                    *previous_messages,
                    HumanMessage(content=message),
                ],
                "sources": [],
                "retrieved_context": "",
                "steps": 0,
            }
        )
        answer_messages = [
            item
            for item in result["messages"]
            if isinstance(item, AIMessage) and not item.tool_calls and item.content
        ]
        if not answer_messages:
            raise RuntimeError("The local agent finished without producing an answer.")

        unique_sources = {}
        for source in result["sources"]:
            key = (source.get("filename"), source.get("chunk_index"))
            unique_sources[key] = source

        return {
            "answer": str(answer_messages[-1].content),
            "sources": list(unique_sources.values()),
        }
