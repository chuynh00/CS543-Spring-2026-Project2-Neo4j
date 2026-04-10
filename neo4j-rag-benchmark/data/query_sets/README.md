Place benchmark query sets here as JSONL files.

Each line must contain:
- query_id: string
- embedding: list of floats
- top_k: integer
- depth: integer

Optional:
- config: object forwarded to rag.retrieve(..., config)
