CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node, score
RETURN id(node) AS nodeId, score
ORDER BY score DESC, nodeId
