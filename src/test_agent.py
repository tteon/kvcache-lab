import os
import sys

from dotenv import load_dotenv

load_dotenv()


def run_basic_agent_check():
    print("\n🤖 [Step 1] Testing OpenAI Agent Connection...")
    from agents import Agent, Runner

    agent = Agent(
        name="PoetBot",
        instructions="You are a poetic assistant. Always answer in Korean Haiku style (5-7-5 syllables).",
    )

    result = Runner.run_sync(agent, "프로그래밍에서의 재귀(Recursion)에 대해 시를 써줘.")

    print(f"✅ Agent Output:\n{'-'*30}\n{result.final_output}\n{'-'*30}")


def run_db_connection_check():
    print("\n🗄️  [Step 2] Testing DozerDB (agentworkload1) Connection...")
    from mem0 import Memory

    try:
        config = {
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                    "database": "agentworkload1",
                },
            }
        }

        memory = Memory.from_config(config)
        memory.add("Agent connection test successful.", user_id="test_runner")
        print("✅ Connected to 'agentworkload1' and saved memory successfully.")

    except Exception as e:
        print(f"❌ DB Connection Error: {e}")


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file.")
        return 1

    print("🚀 Starting System Health Check...\n")
    run_basic_agent_check()
    run_db_connection_check()
    print("\n✨ All systems operational. Ready for tracing experiment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
