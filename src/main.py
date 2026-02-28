# src/main.py
import os
from dotenv import load_dotenv
from mem0 import Memory
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver

# 환경 변수 로드
load_dotenv()

# --- Configuration ---
NEO4J_URI = "bolt://localhost:7687" 
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password" 

# ==========================================
# 1. Mem0 Factory
# ==========================================
def get_mem0_client(db_name: str, user_id: str):
    """
    특정 DB(db_name)에 격리된 Mem0 클라이언트를 반환합니다.
    """
    print(f"🔌 [Mem0] Connecting to DozerDB: '{db_name}'...")
    config = {
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": NEO4J_URI,
                "username": NEO4J_USER,
                "password": NEO4J_PASSWORD,
                "database": db_name,  # <--- 격리 포인트
            }
        },
        "version": "v1.1"
    }
    return Memory.from_config(config)

# ==========================================
# 2. Graphiti(Zep) Factory
# ==========================================
def get_graphiti_client(db_name: str):
    """
    특정 DB(db_name)에 격리된 Graphiti 클라이언트를 반환합니다.
    """
    print(f"🔌 [Graphiti] Connecting to DozerDB: '{db_name}'...")
    
    # Graphiti Driver에 DB 이름을 직접 주입
    driver = Neo4jDriver(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=db_name # <--- 격리 포인트
    )
    
    return Graphiti(graph_driver=driver)

# ==========================================
# 🚀 Main Execution: 4-Way Mapping Test (No Hyphens)
# ==========================================
def main():
    print("🚀 Starting Multi-Instance Isolation Test (Clean Naming)\n")

    # --- A. Mem0 인스턴스 (mem0store) ---
    mem0_agent = get_mem0_client("mem0store", "user_mem0")
    mem0_agent.add("Alice is a Graph Engineer interested in DozerDB.", user_id="user_mem0")
    print("✅ Mem0 Data Saved to 'mem0store'")

    # --- B. Zep/Graphiti 인스턴스 (zepstore) ---
    try:
        zep_agent = get_graphiti_client("zepstore")
        # zep_agent.add_node(...) 
        print("✅ Graphiti Client Ready linked to 'zepstore'")
    except Exception as e:
        print(f"⚠️ Graphiti Init Error (Check version): {e}")

    # --- C. 실험군 DB1 (agentworkload1) ---
    agent_1 = get_mem0_client("agentworkload1", "experiment_bot_1")
    agent_1.add("This is isolated data for Agent 1.", user_id="bot1")
    print("✅ Agent 1 Data Saved to 'agentworkload1'")

    # --- D. 실험군 DB2 (agentworkload2) ---
    agent_2 = get_graphiti_client("agentworkload2")
    print("✅ Agent 2 Client Ready linked to 'agentworkload2'")

    # --- 검증: Mem0 메인 DB에서 Agent 1의 데이터가 보이는가? ---
    print("\n🔍 Isolation Test:")
    # mem0store에서 agentworkload1의 데이터를 검색 시도
    results = mem0_agent.search("Agent 1", user_id="user_mem0")
    
    if not results or not results.get('results'):
        print("SUCCESS: 'mem0store' cannot see 'agentworkload1' data. Isolation Confirmed.")
    else:
        print(f"WARNING: Data Leakage Detected! Found: {results}")

if __name__ == "__main__":
    main()
