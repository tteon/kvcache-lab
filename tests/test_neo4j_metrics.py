from trace_collector.neo4j_metrics import classify_cypher_query, cypher_hash


def test_cypher_hash_is_stable_for_whitespace():
    q1 = "MATCH (n) RETURN n"
    q2 = "  MATCH   (n)\nRETURN n  "
    assert cypher_hash(q1) == cypher_hash(q2)


def test_classify_cypher_query():
    assert classify_cypher_query("SHOW INDEXES") == "indexing"
    assert classify_cypher_query("MATCH (n) RETURN n LIMIT 10") == "read"
    assert classify_cypher_query("MERGE (n:Node {id: 1}) RETURN n") == "write"
    assert (
        classify_cypher_query(
            "MATCH (n) WITH n, vector.similarity.cosine(n.embedding, $e) AS s RETURN s"
        )
        == "search"
    )
