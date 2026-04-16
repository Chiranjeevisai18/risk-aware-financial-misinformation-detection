import asyncio
import os
import json
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from dotenv import load_dotenv

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

load_dotenv()

class StockMarketClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Initialize Groq via LangChain
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    def _convert_mcp_to_langchain_tool(self, mcp_tool) -> Dict[str, Any]:
        """Convert MCP tool schema to LangChain/OpenAI format"""
        return {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema
        }

    def web_search(self, query: str, search_depth: str = "advanced") -> str:
        """Search the web using Tavily — optimized for AI agent workflows."""
        print(f"Executing web search for: {query}")
        try:
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            response = tavily.search(
                query=query,
                search_depth=search_depth,
                max_results=5,
                include_answer=True,       # Get a pre-summarized answer
                include_raw_content=False  # Keep token usage low
            )

            parts = []

            # Include the pre-synthesized Tavily answer if available
            if response.get("answer"):
                parts.append(f"Summary: {response['answer']}")

            # Include top source snippets for reference
            results = response.get("results", [])
            if results:
                parts.append("\nSources:")
                for r in results:
                    title = r.get("title", "")
                    snippet = r.get("content", "")[:500]  # Limit snippet length
                    url = r.get("url", "")
                    parts.append(f"- {title}\n  {snippet}\n  Source: {url}")

            return "\n".join(parts) if parts else f"No results found for '{query}'"
        except Exception as e:
            return f"Error performing web search: {str(e)}"

    async def _scrape_news_articles(self, news_json_str: str, max_articles: int = 6) -> str:
        """
        Given the raw JSON string from get_market_news/get_company_news,
        extract up to max_articles URLs and scrape each one.
        Returns a formatted string with all article content appended.
        """
        try:
            data = json.loads(news_json_str)
        except (json.JSONDecodeError, TypeError):
            return ""

        # Handle both list responses and dict responses with an "articles" key
        articles = []
        if isinstance(data, list):
            articles = data
        elif isinstance(data, dict):
            articles = data.get("articles", [])

        if not articles:
            return ""

        scraped_parts = ["\n\n---\n📰 Article Content (scraped):\n"]
        count = 0

        for article in articles[:max_articles]:
            url = article.get("url", "")
            headline = article.get("headline") or article.get("title", "No title")
            source = article.get("source", "")

            if not url:
                continue

            print(f"  Scraping article: {headline[:60]}...")
            try:
                result = await self.session.call_tool("scrape_article", {"url": url})
                raw = ""
                for c in result.content:
                    if getattr(c, "type", "") == "text":
                        raw += c.text

                # Parse scrape_article JSON response
                try:
                    scraped = json.loads(raw)
                    title = scraped.get("title", headline)
                    text = scraped.get("text", "")[:2000]  # cap per article
                except (json.JSONDecodeError, TypeError):
                    title = headline
                    text = raw[:2000]

                if text:
                    scraped_parts.append(
                        f"\n### {title}\n"
                        f"Source: {source} | URL: {url}\n"
                        f"{text}\n"
                    )
                    count += 1

            except Exception as e:
                # Skip articles that fail to scrape — don't break the flow
                print(f"  Skipped (scrape failed): {e}")
                continue

        if count == 0:
            return ""

        return "\n".join(scraped_parts)

    async def process_query(self, query: str) -> str:
        """Process a query using Groq, MCP tools, and Web Search"""
        system_prompt = (
            "You are a premium financial intelligence agent. Your goal is to provide deep, accurate, and synthesized "
            "financial analysis using real-time market data and scraped news content.\n\n"
            "CORE MISSION:\n"
            "- Use all available tools to gather a complete picture (price, financials, news, and scraped article text).\n"
            "- Provide a COMPREHENSIVE synthesis of market news, price movements, and company financials.\n\n"
            "RESPONSE STRUCTURE (MANDATORY):\n"
            "1. **Executive Summary**: A high-level 2-3 sentence overview.\n"
            "2. **Detailed Analysis**: Synthesis of Price, Financials, and News context.\n"
            "3. **Article Summaries**: Concise summary of EACH scraped article provided.\n"
            "4. 📰 **Sources**: Mandatory list of citations.\n\n"
            "MANDATORY REQUIREMENT: EVERY citation in the Sources section MUST include the full clickable Link (URL). "
            "Omitting links or providing headlines without URLs is a violation of your protocol. "
            "If a URL was provided in the tool output, it MUST appear in the final answer."
        )
        messages: List[BaseMessage] = [
            AIMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        # Get MCP tools
        response = await self.session.list_tools()
        tools = [self._convert_mcp_to_langchain_tool(t) for t in response.tools]
        
        # Add local web_search tool
        tools.append({
            "name": "web_search",
            "description": (
                "Search the web for real-time information, breaking news, general knowledge, or any topic "
                "the specialized financial tools cannot answer. Craft a specific, concise search query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific, targeted web search query."
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Use 'advanced' for financial or complex queries; 'basic' for simple facts."
                    }
                },
                "required": ["query"]
            }
        })
        
        # Bind all tools to the LLM
        llm_with_tools = self.llm.bind_tools(tools)

        while True:
            # Call Groq — catch malformed tool-call errors (Groq 400 tool_use_failed)
            try:
                response_message = await llm_with_tools.ainvoke(messages)
            except Exception as invoke_err:
                err_str = str(invoke_err)
                if "tool_use_failed" in err_str or "400" in err_str:
                    # Groq generated a malformed tool call — fall back to web_search
                    print(f"  Tool-call format error from model, falling back to web_search...")
                    fallback_result = self.web_search(query, "advanced")
                    messages.append(HumanMessage(content=(
                        f"Web search results for '{query}':\n{fallback_result}\n\n"
                        "Please synthesize the above results into a comprehensive answer following the mandatory response structure."
                    )))
                    response_message = await self.llm.ainvoke(messages)
                    messages.append(response_message)
                    break
                raise  # Re-raise non-tool-call errors

            messages.append(response_message)

            # Check if tools need to be called
            if not response_message.tool_calls:
                break

            # Execute tool calls
            for tool_call in response_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                try:
                    if tool_name == "web_search":
                        # Handle local web_search tool
                        result_content = self.web_search(
                            tool_args["query"],
                            tool_args.get("search_depth", "advanced")
                        )
                    else:
                        # Handle MCP tool
                        print(f"Calling MCP tool: {tool_name}")
                        result = await self.session.call_tool(tool_name, tool_args)
                        raw_content = ""
                        for c in result.content:
                            if getattr(c, "type", "") == "text":
                                raw_content += c.text

                        # If it's a news tool, auto-scrape top articles for full content
                        if tool_name in ("get_market_news", "get_company_news"):
                            result_content = raw_content + await self._scrape_news_articles(raw_content)
                        else:
                            result_content = raw_content or str(result.content)
                    
                    # Append ToolMessage
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=result_content
                    ))
                except Exception as e:
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"Error: {str(e)}"
                    ))

        return messages[-1].content

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nUnified MCP & Search Client Started!")
        print("Type your queries, 'test' to run all tool tests, or 'quit' to exit.")

        test_queries = [
            "What is the current stock price of NVDA?",
            "What is the latest world news about space exploration?",
            "Show me D resolution stock candles for AAPL for the last 3 days",
            "Who won the last major tennis tournament?",
        ]

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                
                if query.lower() == 'test':
                    print("\n--- Running Unified Tests ---")
                    for q in test_queries:
                        print(f"\nQuery: {q}")
                        response = await self.process_query(q)
                        print(f"Assistant: {response}")
                        print("-" * 30)
                    continue

                response = await self.process_query(query)
                print("\nAssistant: " + str(response))

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = StockMarketClient()
    try:
        await client.connect_to_server("./stock_market_server.py")
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
