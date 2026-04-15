import os
import finnhub
import json
import time
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from readability import Document
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("stock-market")

# Get API key from environment
api_key = os.getenv('FINNHUB_API_KEY')
if not api_key:
    raise ValueError("FINNHUB_API_KEY environment variable not set")

finnhub_client = finnhub.Client(api_key=api_key)

MAX_ARTICLE_CHARS = 12000

def _extract_readable_text(html: str) -> tuple[str, str]:
    """Extract the main text and title from an HTML document."""
    title = ""
    text = ""
    try:
        doc = Document(html)
        title = doc.short_title() or ""
        readable_html = doc.summary()
        soup = BeautifulSoup(readable_html, "lxml")
        text = soup.get_text(separator="\n")
    except Exception:
        # Fallback to raw HTML if readability fails
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        text = soup.get_text(separator="\n")

    # Normalize whitespace and trim
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS].rstrip() + "..."

    return title, text

@mcp.tool()
def get_stock_symbol_lookup(query: str) -> str:
    """
    Stock Symbol Lookup - searches for best-matching symbols based on your query.

    Args:
        query: You can input anything from symbol, security's name to ISIN and CUSIP e.g. apple

    Returns:
        str: A list of matching symbols with formatted information
    """
    try:
        data = finnhub_client.symbol_lookup(query)

        if not data or "result" not in data:
            return "Unable to look up symbol."

        if len(data["result"]) == 0:
            return f"No symbols found matching '{query}'."

        formatted_data = {
            "search_query": query,
            "total_matches": data["count"],
            "matches": []
        }

        for item in data["result"]:
            formatted_data["matches"].append({
                "symbol": item["symbol"],
                "name": item["description"],
                "display_symbol": item["displaySymbol"],
                "type": item["type"]
            })

        return json.dumps(formatted_data, indent=2)

    except Exception as e:
        return f"Error looking up symbol: {str(e)}"

@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """
    Get the latest stock price and related information for a given symbol.

    Args:
        symbol: The stock symbol to look up (e.g., AAPL for Apple Inc.)

    Returns:
        str: Current price information in JSON format
    """
    try:
        # 1. Try raw symbol
        data = finnhub_client.quote(symbol.upper())
        
        # 2. Crypto Fallback: if c is 0, try Binance format (e.g. BINANCE:BTCUSDT)
        if (not data or data.get("c") == 0) and ":" not in symbol:
            crypto_symbol = f"BINANCE:{symbol.upper()}USDT"
            crypto_data = finnhub_client.quote(crypto_symbol)
            if crypto_data and crypto_data.get("c", 0) > 0:
                data = crypto_data
                symbol = crypto_symbol

        if not data or "c" not in data:
            return f"Unable to get price information for symbol '{symbol}'."
            
        if data["c"] == 0 and data["h"] == 0 and data["l"] == 0:
            return f"No price data available for '{symbol}'. Please verify the symbol is correct (For crypto, try 'BINANCE:BTCUSDT' format)."
        
        # Format the response with readable labels
        formatted_data = {
            "symbol": symbol.upper(),
            "current_price": data["c"],
            "day_high": data["h"],
            "day_low": data["l"],
            "day_open": data["o"],
            "previous_close": data["pc"],
            "timestamp": data["t"]
        }
        
        return json.dumps(formatted_data, indent=2)
        
    except Exception as e:
        return f"Error getting stock price: {str(e)}"

@mcp.tool()
def get_basic_financials(symbol: str) -> str:
    """
    Get basic financial information for a company.

    Args:
        symbol: The stock symbol to look up (e.g., AAPL for Apple Inc.)

    Returns:
        str: Basic financial metrics in JSON format including P/E ratio, market cap, etc.
    """
    try:
        # Get basic financials from Finnhub
        data = finnhub_client.company_basic_financials(symbol.upper(), 'all')
        
        if not data or "metric" not in data:
            return f"Unable to get financial information for symbol '{symbol}'."
        
        metrics = data["metric"]
        
        # Select the most important metrics
        important_metrics = {
            "symbol": symbol.upper(),
            "company_name": data.get("series", {}).get("name", "Unknown"),
            "market_capitalization": metrics.get("marketCapitalization", None),
            "pe_ratio": metrics.get("peBasicExclExtraTTM", None),
            "pb_ratio": metrics.get("pbQuarterlyTTM", None),
            "dividend_yield": metrics.get("dividendYieldIndicatedAnnual", None),
            "52_week_high": metrics.get("52WeekHigh", None),
            "52_week_low": metrics.get("52WeekLow", None),
            "52_week_change": metrics.get("52WeekPriceReturnDaily", None),
            "beta": metrics.get("beta", None),
            "eps_ttm": metrics.get("epsBasicExclExtraItemsTTM", None),
            "revenue_per_share_ttm": metrics.get("revenuePerShareTTM", None),
            "revenue_growth_ttm": metrics.get("revenueGrowthTTM3Y", None),
            "debt_to_equity": metrics.get("totalDebtEquityQuarterly", None),
            "roa": metrics.get("roaTTM", None),
            "roe": metrics.get("roeTTM", None)
        }
        
        return json.dumps(important_metrics, indent=2)
        
    except Exception as e:
        return f"Error getting financial information: {str(e)}"

@mcp.tool()
def get_market_news(category: str = "general", min_id: int = 0) -> str:
    """
    Get the latest market news.

    Args:
        category: News category. Available values: general, forex, crypto, merger.
        min_id: Use this to get only news after this ID. This MUST be an integer, not a string.

    Returns:
        str: Latest market news in JSON format with headlines, summaries, and URLs.
    """
    try:
        valid_categories = ["general", "forex", "crypto", "merger"]
        if category.lower() not in valid_categories:
            return f"Invalid category. Please use one of: {', '.join(valid_categories)}"
        
        data = finnhub_client.general_news(category.lower(), min_id=min_id)
        
        if not data or len(data) == 0:
            return f"No news available for category '{category}'."
        
        formatted_data = {
            "category": category.lower(),
            "news_count": len(data),
            "articles": []
        }
        
        for item in data[:10]:
            formatted_data["articles"].append({
                "id": item.get("id", 0),
                "headline": item.get("headline", "No headline"),
                "summary": item.get("summary", "No summary available"),
                "source": item.get("source", "Unknown source"),
                "datetime": item.get("datetime", 0),
                "url": item.get("url", ""),
                "related_symbols": item.get("related", [])
            })
        
        return json.dumps(formatted_data, indent=2)
        
    except Exception as e:
        return f"Error getting market news: {str(e)}"

@mcp.tool()
def get_company_news(symbol: str, from_date: str, to_date: str) -> str:
    """
    Get news for a specific company over a date range.

    Args:
        symbol: The stock symbol (e.g., AAPL for Apple Inc.)
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format

    Returns:
        str: Company-specific news in JSON format with headlines, summaries, and URLs.
    """
    try:
        if not (len(from_date) == 10 and len(to_date) == 10):
            return "Date format must be YYYY-MM-DD"
        if from_date[4] != '-' or from_date[7] != '-' or to_date[4] != '-' or to_date[7] != '-':
            return "Date format must be YYYY-MM-DD (with dashes)"

        data = finnhub_client.company_news(symbol.upper(), _from=from_date, to=to_date)

        if not data or len(data) == 0:
            return f"No news available for {symbol} between {from_date} and {to_date}."

        formatted_data = {
            "symbol": symbol.upper(),
            "from_date": from_date,
            "to_date": to_date,
            "news_count": len(data),
            "articles": []
        }

        for item in data[:10]:
            formatted_data["articles"].append({
                "headline": item.get("headline", "No headline"),
                "summary": item.get("summary", "No summary available"),
                "source": item.get("source", "Unknown source"),
                "datetime": item.get("datetime", 0),
                "url": item.get("url", ""),
                "related_symbols": item.get("related", [])
            })

        return json.dumps(formatted_data, indent=2)

    except Exception as e:
        return f"Error getting company news: {str(e)}"

@mcp.tool()
def get_stock_candles(symbol: str, resolution: str = "D", from_time: str = None, to_time: str = None) -> str:
    """
    Get historical price data (candles) for a stock.
    Note: This is a premium feature, your API key may not have access to this feature.

    Args:
        symbol: The stock symbol (e.g., AAPL for Apple Inc.)
        resolution: Time interval between data points. Supported values: 1, 5, 15, 30, 60, D, W, M
        from_time: Start time in YYYY-MM-DD format or Unix timestamp
        to_time: End time in YYYY-MM-DD format or Unix timestamp

    Returns:
        str: Historical price data in JSON format with open, high, low, close values.
    """
    try:
        if from_time is None:
            from_timestamp = int(time.time()) - (30 * 24 * 60 * 60)
        else:
            try:
                if '-' in from_time:
                    dt = datetime.strptime(from_time, '%Y-%m-%d')
                    from_timestamp = int(dt.timestamp())
                else:
                    from_timestamp = int(from_time)
            except ValueError:
                return "Invalid from_time format. Use YYYY-MM-DD or Unix timestamp."

        if to_time is None:
            to_timestamp = int(time.time())
        else:
            try:
                if '-' in to_time:
                    dt = datetime.strptime(to_time, '%Y-%m-%d')
                    to_timestamp = int(dt.timestamp())
                else:
                    to_timestamp = int(to_time)
            except ValueError:
                return "Invalid to_time format. Use YYYY-MM-DD or Unix timestamp."

        valid_resolutions = ["1", "5", "15", "30", "60", "D", "W", "M"]
        if resolution not in valid_resolutions:
            return f"Invalid resolution. Please use one of: {', '.join(valid_resolutions)}"

        data = finnhub_client.stock_candles(symbol.upper(), resolution, from_timestamp, to_timestamp)

        if not data or data.get("s") != "ok":
            return f"Unable to get candle data for {symbol}. Status: {data.get('s', 'unknown')}"

        formatted_data = {
            "symbol": symbol.upper(),
            "resolution": resolution,
            "from_time": from_timestamp,
            "to_time": to_timestamp,
            "status": data.get("s"),
            "candle_count": len(data.get("t", [])),
            "candles": []
        }

        timestamps = data.get("t", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        volumes = data.get("v", [])

        for i in range(min(len(timestamps), 100)):
            formatted_data["candles"].append({
                "timestamp": timestamps[i],
                "datetime": datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M:%S'),
                "open": opens[i] if i < len(opens) else None,
                "high": highs[i] if i < len(highs) else None,
                "low": lows[i] if i < len(lows) else None,
                "close": closes[i] if i < len(closes) else None,
                "volume": volumes[i] if i < len(volumes) else None
            })

        return json.dumps(formatted_data, indent=2)

    except Exception as e:
        return f"Error getting stock candles: {str(e)}"

@mcp.tool()
def scrape_article(url: str, timeout_s: int = 15) -> str:
    """
    Scrape a news article URL and return extracted text content.

    Args:
        url: The article URL to scrape (http/https).
        timeout_s: Request timeout in seconds. This MUST be an integer, not a string.

    Returns:
        str: JSON payload with title and extracted text.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Invalid URL scheme. Only http/https are supported."

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        }

        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return "URL did not return HTML content."

        title, text = _extract_readable_text(response.text)

        payload = {
            "url": url,
            "title": title,
            "text": text,
            "content_length": len(text),
            "truncated": len(text) >= MAX_ARTICLE_CHARS,
        }

        return json.dumps(payload, indent=2)

    except httpx.HTTPError as e:
        return f"HTTP error scraping article: {str(e)}"
    except Exception as e:
        return f"Error scraping article: {str(e)}"

@mcp.prompt("stock_analysis")
def stock_analysis_prompt():
    return """
    I need to analyze a stock for potential investment. Please help me with the following:
    
    1. Look up the symbol for {company_name}
    2. Get the current price for the best matching symbol
    3. Retrieve the basic financial information
    4. Based on the P/E ratio, dividend yield, and recent price movements, provide a brief assessment of whether this might be a good investment opportunity
    
    Please format your analysis in a clear, structured way with sections for each piece of information.
    """

if __name__ == "__main__":
    mcp.run(transport='stdio')
