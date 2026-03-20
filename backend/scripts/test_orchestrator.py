import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from backend.rag.chat_orchestrator import ChatOrchestrator, ChatRequest

def test_chat():
    print("Testing ChatOrchestrator...")
    orchestrator = ChatOrchestrator()
    
    request = ChatRequest(
        query="Can Chelsea win against Aston Villa? And is over 2.5 a good bet?",
        history=[],
        home_team_id=1, # Fake IDs, but the names will be used for anchor resolution
        away_team_id=2,
        home_team_name="Chelsea",
        away_team_name="Aston Villa",
        league_id="17", # Premier League
    )
    
    print("\n--- Sending multi-intent query ---")
    print(f"Query: {request.query}")
    print(f"Fixture: {request.home_team_name} vs {request.away_team_name}")
    print("Streaming response:")
    
    for chunk in orchestrator.stream(request):
        print(chunk, end="", flush=True)
    
    print("\n\n--- Test Complete ---")

if __name__ == "__main__":
    test_chat()
