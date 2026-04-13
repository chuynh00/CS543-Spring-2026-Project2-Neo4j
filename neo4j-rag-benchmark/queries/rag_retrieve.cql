CALL rag.retrieve($index_name, $embedding, $top_k, $depth, $config)
YIELD node, hopDepth, score
RETURN id(node) AS nodeId, hopDepth, score
ORDER BY hopDepth, nodeId
