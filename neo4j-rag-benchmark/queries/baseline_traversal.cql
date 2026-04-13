MATCH p=(seed)-[*0..__DEPTH__]-(node)
WHERE id(seed) IN $seed_ids
RETURN id(node) AS nodeId, min(length(p)) AS hopDepth
ORDER BY hopDepth, nodeId
