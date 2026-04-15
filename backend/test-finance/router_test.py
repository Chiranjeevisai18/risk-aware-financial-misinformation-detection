import asyncio
import os
import sys
from dotenv import load_dotenv
import certifi

# --- SSL CERTIFICATE FIX ---
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.agent import FinancialAgent
from components.mcp_manager import MCPToolManager

async def run_router_tests():
    load_dotenv()
    
    print("STARTING Router Based Tests for Financial Agent...")
    
    # 1. Initialization
    print("\n--- Phase 1: Initialization ---")
    mcp_managers = []
    try:
        # Assuming the server script is in the same directory for this test
        manager = MCPToolManager(server_script="./stock_market_server.py")
        mcp_managers.append(manager)
        print("[OK] MCP Manager initialized.")
    except Exception as e:
        print(f"[FAIL] Failed to initialize MCP Manager: {e}")
        return

    agent = FinancialAgent(mcp_managers)
    print(f"[OK] Agent initialized with model: {agent.model_name}")

    # 2. Define test scenarios
    tests = [
        {"name": "Symbol Lookup", "query": "Lookup the ticker for 'Nvidia Corporation'"},
        {"name": "Stock Price", "query": "What is the current price of AAPL?"},
        {"name": "Basic Financials", "query": "Show me the PE ratio and market cap for Microsoft (MSFT)"},
        {"name": "Market News", "query": "Give me the top 5 general market news headlines"},
        {"name": "Company News", "query": "What are the latest news articles for Tesla (TSLA)?"},
        {"name": "Stock Candles", "query": "Show me the historical price movements (candles) for Google (GOOGL) over the last week"},
    ]

    # 3. Execute tests
    print("\n--- Phase 2: Tool Execution ---")
    for test in tests:
        print(f"\n[Testing {test['name']}] Query: '{test['query']}'")
        try:
            # We use a 60s timeout per test
            result = await asyncio.wait_for(agent.chat(test["query"]), timeout=60.0)
            
            print(f"  Response Type: {result.get('type')}")
            answer = result.get('answer', '')
            print(f"  Answer Snippet: {answer[:200]}...")
            
            if "no response" in answer.lower() or "error" in answer.lower():
                print(f"  [WARN] Potential Failure detected in answer.")
            else:
                print(f"  [OK] Tool seems to have triggered and returned data.")
                
            if result.get('top_articles'):
                 print(f"  [OK] Articles found: {len(result['top_articles'])}")
                 
        except asyncio.TimeoutError:
            print(f"  [FAIL] Timeout: Agent took too long to respond.")
        except Exception as e:
            print(f"  [FAIL] Error during test: {e}")

    # 4. Cleanup
    print("\n--- Phase 3: Cleanup ---")
    print("[OK] Tests completed.")

if __name__ == "__main__":
    asyncio.run(run_router_tests())
