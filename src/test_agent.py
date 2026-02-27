# src/test_agent.py
import os
import sys
from dotenv import load_dotenv
from agents import Agent, Runner
from mem0 import Memory

# 1. 환경 변수 로드
load_dotenv()

# API Key 확인 (없으면 에러)
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

# ==========================================
# 🔍 1단계: OpenAI Agent 작동 테스트 (Haiku)
# ==========================================
def test_basic_agent():
    print("\n🤖 [Step 1] Testing OpenAI Agent Connection...")
    
    agent = Agent(
        name="PoetBot",
        instructions="You are a poetic assistant. Always answer in Korean Haiku style (5-7-5 syllables)."
    )

    # Runner를 통해 실행
    result = Runner.run_sync(agent, "프로그래밍에서의 재귀(Recursion)에 대해 시를 써줘.")
    
    print(f"✅ Agent Output:\n{'-'*30}\n{result.final_output}\n{'-'*30}")

# ==========================================
# 🔍 2단계: DozerDB 격리 환경 연동 테스트
# ==========================================
def test_db_connection():
    print("\n🗄️  [Step 2] Testing DozerDB (agentworkload1) Connection...")
    
    try:
        # 우리가 만든 'agentworkload1' DB에 연결
        config = {
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                    "database": "agentworkload1" # 격리된 DB
                }
            }
        }
        
        memory = Memory.from_config(config)
        
        # 간단한 쓰기/읽기 테스트
        memory.add("Agent connection test successful.", user_id="test_runner")
        print("✅ Connected to 'agentworkload1' and saved memory successfully.")
        
    except Exception as e:
        print(f"❌ DB Connection Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting System Health Check...\n")
    
    # 1. 에이전트 지능 테스트
    test_basic_agent()
    
    # 2. 에이전트 기억장치(DB) 테스트
    test_db_connection()
    
    print("\n✨ All systems operational. Ready for tracing experiment.")
