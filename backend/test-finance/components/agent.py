import os
import re
import uuid
import json
from datetime import datetime, UTC
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from components.mcp_manager import MCPToolManager
from components.schemas import NewsSummaryResponse
from components.utils import SYSTEM_PROMPT

_NEWS_TOOLS = {"get_market_news", "get_company_news", "scrape_article"}

class FinancialAgent:
    """
    LangGraph ReAct agent with Multi-Provider Failover.

    Graph topology
    ──────────────
    START ──► agent ──(tool_calls?)──► tools ──► agent ──► ...
                     └──(done)──────► END
    """

    def __init__(self, mcp_managers: list[MCPToolManager], session_id: str = "__default__"):
        self.tools = []
        self.session_id = session_id
        for mgr in mcp_managers:
            self.tools.extend(mgr.langchain_tools)

        # Primary engine: Groq Llama-3.3-70B
        self.llm_groq = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY")
        ).bind_tools(self.tools)

        # Fallback engine: Gemini 2.5 Flash
        self.llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0, 
            google_api_key=os.getenv("GEMINI_API_KEY")
        ).bind_tools(self.tools)

        # Base models for structured extraction
        self.base_llm_groq = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.base_llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        self._graph = self._compile_graph()
        self._history: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        
    async def load_history(self) -> None:
        """Fetch the last 10 messages from Supabase to provide persistent memory."""
        if self.session_id == "__default__":
            return

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            print("[agent] Supabase credentials missing — persistence disabled.")
            return

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                query_url = f"{supabase_url}/rest/v1/chat_messages?session_id=eq.{self.session_id}&order=created_at.asc&limit=10"
                headers = {
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}"
                }
                resp = await client.get(query_url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    for msg in data:
                        role = msg.get("role")
                        text = msg.get("text", "")
                        if role == "user":
                            self._history.append(HumanMessage(content=text))
                        elif role == "bot":
                            self._history.append(AIMessage(content=text))
                    if data:
                        print(f"[agent] Re-loaded {len(data)} messages from Supabase for {self.session_id}")
        except Exception as e:
            print(f"[agent] Failed to load history from Supabase: {e}")

    def _compile_graph(self):
        tool_node = ToolNode(self.tools)

        def agent_node(state: MessagesState) -> dict:
            messages = state["messages"]

            # Determine current date for temporal context
            current_date_str = datetime.now(UTC).strftime("%Y-%m-%d")
            prompt_with_date = f"{SYSTEM_PROMPT}\n\nCURRENT_DATE: {current_date_str}"

            # Inject system prompt at the front if it's not already there
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=prompt_with_date)] + list(messages)
            else:
                # If it already has a system prompt, update it to include the current date
                msg_list = list(messages)
                for i, m in enumerate(msg_list):
                    if isinstance(m, SystemMessage):
                        msg_list[i] = SystemMessage(content=prompt_with_date)
                        break
                messages = msg_list

            response = None
            error_reason = ""
            use_fallback = False
            
            # --- PHASE 1: TRY GROQ ---
            use_fallback = True # FORCED FOR TESTING
            try:
                print(f"[agent] Invoking Groq (Primary: llama-3.3-70b-versatile)...")
                response = self.llm_groq.invoke(messages)
                
                # Check for hallucinations in Groq response
                if response.content and not response.tool_calls:
                    if re.search(r"(?:<function[=>]|<\|python_tag\|>|[\[\(]call:)", response.content):
                        print(f"[agent] ⚠️ Groq hallucinated a tool-call tag. Triggering fallback...")
                        use_fallback = True
                
                if not use_fallback:
                    print(f"[agent] SUCCESS: Response received from Groq.")
            except Exception as e:
                print(f"[agent] ⚠️ Groq Primary Failed: {e}")
                error_reason = str(e)
                use_fallback = True

            # --- PHASE 2: FALLBACK TO GEMINI ---
            if use_fallback or not response:
                try:
                    print(f"[agent] 🔄 FALLBACK: Switching to Gemini engine...")
                    response = self.llm_gemini.invoke(messages)
                    print(f"[agent] SUCCESS: Response received from Gemini Fallback.")
                except Exception as e:
                    print(f"[agent] ❌ CRITICAL: Gemini fallback also failed: {e}")
                    raise e # Both failed

            # DEBUG LOGGING
            if response.tool_calls:
                print(f"[agent] Generated {len(response.tool_calls)} tool call(s):")
                for tc in response.tool_calls:
                    print(f"   • {tc['name']}({json.dumps(tc['args'])})")
            else:
                print(f"[agent] Engine response received. No tool calls.")

            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")

        return graph.compile()

    async def chat(self, user_input: str) -> dict:
        """Process one user turn with history persistence and structured output."""
        if len(self._history) <= 1:
            await self.load_history()

        initial_history = list(self._history)
        result = await self._graph.ainvoke({"messages": [HumanMessage(content=user_input)]})
        self._history = list(result["messages"])
        new_messages = self._history[len(initial_history)+1:]

        tool_names_used = {m.name for m in new_messages if isinstance(m, ToolMessage)}
        if not tool_names_used:
            return {"type": "general", "query": user_input, "answer": self._extract_text(), "top_articles": []}

        news_tools_used = any(name in _NEWS_TOOLS for name in tool_names_used)
        if not news_tools_used:
            return {"type": "financial", "query": user_input, "answer": self._extract_text(), "top_articles": []}

        try:
            structured: NewsSummaryResponse = await self._extract_news_structured(user_input)
            return {
                "type": "news",
                "query": structured.query,
                "answer": structured.answer,
                "top_articles": [a.model_dump() for a in structured.top_articles],
            }
        except Exception as exc:
            print(f"[extraction failed: {exc}] — falling back")
            answer_text = self._extract_text()
            articles = self._extract_top_articles_from_text(answer_text)
            return {"type": "news", "query": user_input, "answer": answer_text, "top_articles": articles}

    async def _extract_news_structured(self, user_input: str) -> NewsSummaryResponse:
        schema_json = NewsSummaryResponse.model_json_schema()
        extraction_prompt = (
            f"The user asked: \"{user_input}\"\n\n"
            "Using ONLY the article data returned by the tools, provide a JSON object matching this schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n\n"
            "Return ONLY valid JSON."
        )
        messages_for_extraction = [
            SystemMessage(content=SYSTEM_PROMPT),
            *self._history,
            HumanMessage(content=extraction_prompt),
        ]
        
        # Try Groq first for extraction
        try:
            response = await self.base_llm_groq.ainvoke(messages_for_extraction)
        except:
            response = await self.base_llm_gemini.ainvoke(messages_for_extraction)

        content = response.content.strip()
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
             data = json.loads(match.group(1))
             return NewsSummaryResponse(**data)
        raise ValueError("JSON matching failed")

    def _extract_text(self) -> str:
        for msg in reversed(self._history):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str): return msg.content
                if isinstance(msg.content, list):
                    return "\n".join(b["text"] for b in msg.content if b.get("type") == "text")
        return "(no response)"

    def _extract_top_articles_from_text(self, text: str) -> list[dict]:
        top_articles = []
        if "Sources:" in text:
            import re
            parts = re.split(r'Sources?:?', text)
            if len(parts) > 1:
                for line in parts[-1].strip().split('\n')[:5]:
                    if line.strip().startswith('-'):
                        top_articles.append({"headline": line.lstrip('- ').strip(), "source": "N/A"})
        return top_articles

    def reset(self) -> None:
        self._history = []
        print("🔄 Conversation history cleared.")