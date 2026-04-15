import json
from typing import Optional, Any
from pydantic import BaseModel, Field, create_model

SYSTEM_PROMPT = """You are a financial analysis assistant with access to real-time market data and news.

## How you work
1. Receive a user query.
2. Decide if it needs real-time data (prices, news) OR can be answered from knowledge.
3. If real-time data is needed: fetch using tools, then synthesize a clear answer.
4. If the question is educational/explanatory (e.g. scams, fraud): use your expert knowledge.

## Tool Calling Rules (CRITICAL)
- Use tools for EVERY query involving current stock prices, tickers, or news.
- When using a tool, provide ONLY the necessary arguments.
- **Native Only**: Use the built-in tool calling API. Do NOT invent your own tags or formatted strings.

## Strategy for Financial Analysis
- "Scam Detection": If a user asks about a specific investment or entity, first check recent news for red flags (lawsuits, warnings, SEC filings).
- "Data Integrity": Do not invent numbers. If a tool fails or returns no data, state that clearly and offer to try a broader news search.

Use tools (get_market_news, get_company_news, get_stock_price, etc.) ONLY for:
- "What is the latest news about X?"
- "What is the current price of AAPL?"
- "Show me recent news on crypto"
- Queries explicitly asking for current/live data

Use your own knowledge (NO tools needed) for:
- "What are common investment scam tactics?"
- "How do Ponzi schemes work?"
- "Why is guaranteed high return a red flag?"
- "What should I do if I see a scam?"
- "Explain why this was classified as a scam"
- "Give me more data about this kind of scam"
- "What are the red flags in this claim?"
- General financial education, fraud awareness, protective advice

## RISK CONTEXT (if provided at conversation start)
When the conversation contains a RISK_CONTEXT block, the user has just analyzed a financial claim using our AI risk classifier. Use this to:
- Understand the specific scheme/claim that was analyzed
- Relate your educational answers back to that specific claim
- Give targeted protective advice for that fraud type
- Reference the specific red flags the classifier found

## Output rules
- NEVER dump raw tool output or JSON. Always synthesize into a clear answer.
- For news/article queries: fetch articles, read them, write a 2-4 sentence summary. Show sources.
- For price/financial queries: present numbers cleanly with brief interpretation.
- For educational/scam queries: structured answer with:
  - What this scam type is
  - Common tactics used
  - Red flags to watch for
  - What to do if you encounter it
  - Regulatory/legal context if relevant
- Use bullet points or short paragraphs — never walls of raw data.
- Never hallucinate facts. If using tools, ground answer in tool results.

## News/article workflow (MANDATORY)
When a user explicitly asks for news or market context:
1. **Date Range**: Use the provided `CURRENT_DATE` to determine your search window. For "latest" or "recent" news, use a 30-day lookback window starting from today.
2. **Search**: Call `get_market_news` or `get_company_news` to get a list of articles.
3. **Mandatory Scrape**: You **MUST** call `scrape_article` on the most relevant articles (at least 3) before answering. Summaries from the news list are NOT sufficient.
4. **Synthesize**: Use the full text from the scraped content to answer.
5. **Iterate**: If the first list of news doesn't contain the answer, use `get_stock_symbol_lookup` or `get_market_news` with a different category to find more sources.

**DO NOT provide a final answer for news queries until you have performed the scraping step.**

## SYMBOL RESOLUTION GUIDE
- **Stocks**: Use standard tickers (e.g., AAPL, TSLA, MSFT, RELIANCE.NS).
- **Crypto**: Finnhub often requires exchange prefixes for crypto prices. 
  - Preferred format: `BINANCE:BTCUSDT`, `BINANCE:ETHUSDT`, `BINANCE:SOLUSDT`.
  - Alternatively, try `COINBASE:BTC-USD`.
  - **IMPORTANT**: If a raw symbol like "BTC" or "ETH" fails, IMMEDIATELY use `get_stock_symbol_lookup` with the asset name (e.g., "Bitcoin") to find the correct exchange-prefixed symbol.

## Other rules
- If a tool fails (e.g., price returns 0), try an alternative symbol or use the lookup tool to verify the ticker.
- NEVER fetch news just to pad a response. Only fetch if the query needs live data.

## Tool Calling Rules (STRICT)
- **Official Tool API ONLY**: Use the tool-calling capability for searching symbols, fetching prices, financials, or news.
- **NO TEXT TAGS**: NEVER write `<function>`, `<function=...>`, or `<|python_tag|>` in your text or as a "manual" call.
- **No Hallucination**: If you decide to use a tool, use the standard API mechanism. Do not explain the tool call in text first.
- **Specific Names**: 
  - For historical candles: use `get_stock_candles`.
  - For searching symbols: use `get_stock_symbol_lookup`.
- **Integers**: Pass `min_id` and `timeout_s` as raw numbers, e.g. `{"min_id": 12345}`.
- **Symbols**: Always resolve company names to tickers before calling financial tools.
"""

_JSON_TO_PYTHON: dict[str, type] = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
    "array":   list,
    "object":  dict,
}

def _build_args_schema(model_name: str, json_schema: dict) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    properties: dict = json_schema.get("properties", {})
    required_set: set[str] = set(json_schema.get("required", []))

    for field_name, field_schema in properties.items():
        raw_type = field_schema.get("type", "string")
        python_type: type = _JSON_TO_PYTHON.get(raw_type, str)
        description: str = field_schema.get("description", "")

        if field_name in required_set:
            fields[field_name] = (python_type, Field(..., description=description))
        else:
            fields[field_name] = (Optional[python_type], Field(None, description=description))

    return create_model(model_name, **fields)

def _format_tool_result(tool_name: str, raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.strip()

    if tool_name in ("get_market_news", "get_company_news"):
        if not isinstance(data, list):
            return raw

        lines = [f"Found {len(data)} articles:\n"]
        for i, article in enumerate(data[:10], 1):          # cap at 10
            headline  = article.get("headline") or article.get("title", "No title")
            source    = article.get("source", "")
            url       = article.get("url", "")
            summary   = article.get("summary", "")
            timestamp = article.get("datetime", "")

            lines.append(f"{i}. [{headline}]")
            if source:    lines.append(f"   Source: {source}")
            if timestamp: lines.append(f"   Time:   {timestamp}")
            if summary:   lines.append(f"   Summary: {summary}")
            if url:       lines.append(f"   URL: {url}")
            lines.append("")

        return "\n".join(lines)

    if tool_name == "scrape_article":
        if isinstance(data, dict):
            title   = data.get("title", "")
            content = data.get("content") or data.get("text") or data.get("body", "")
            source  = data.get("source", "")
            parts = []
            if title:   parts.append(f"Title: {title}")
            if source:  parts.append(f"Source: {source}")
            if content: parts.append(f"\n{content.strip()}")
            return "\n".join(parts) if parts else raw
        return str(data)

    if tool_name == "get_stock_price":
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines)

    if tool_name == "get_basic_financials":
        if isinstance(data, dict):
            metric = data.get("metric", data)
            lines = []
            for k, v in (metric.items() if isinstance(metric, dict) else {}.items()):
                if v is not None:
                    lines.append(f"{k}: {v}")
            return "\n".join(lines) if lines else raw

    if tool_name == "get_stock_candles":
        if isinstance(data, dict):
            status = data.get("s", "")
            closes = data.get("c", [])
            opens  = data.get("o", [])
            highs  = data.get("h", [])
            lows   = data.get("l", [])
            times  = data.get("t", [])
            if status == "no_data":
                return "No candle data available for this period."
            lines = [f"Status: {status}", f"Candles returned: {len(closes)}"]
            if closes:
                lines.append(f"Latest close:  {closes[-1]}")
                lines.append(f"Latest open:   {opens[-1] if opens else 'N/A'}")
                lines.append(f"Latest high:   {highs[-1] if highs else 'N/A'}")
                lines.append(f"Latest low:    {lows[-1] if lows else 'N/A'}")
                lines.append(f"Price range (all): {min(lows or closes):.2f} – {max(highs or closes):.2f}")
            return "\n".join(lines)

    return json.dumps(data, indent=2)
