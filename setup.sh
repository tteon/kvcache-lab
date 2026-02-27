#!/bin/bash

echo "🚀 Setting up directories for Neo4j with DozerDB, OpenGDS, and n10s..."

# 1. 디렉토리 구조 생성 (Data, Logs, Plugins 등)
mkdir -p data/neo4j/data
mkdir -p data/neo4j/logs
mkdir -p data/neo4j/import
mkdir -p data/neo4j/plugins

# 2. OpenGDS 플러그인 다운로드 (DozerDB 호환 버전)
# n10s와 APOC는 Docker ENV 설정을 통해 자동 설치되므로 여기서는 OpenGDS만 받습니다.
echo "⬇️  Downloading OpenGDS 2.12.0..."
wget -O data/neo4j/plugins/open-gds-2.12.0.jar https://dist.dozerdb.org/plugins/open-gds/open-gds-2.12.0.jar

echo "✅ Setup Complete. Ready to run docker-compose up."
