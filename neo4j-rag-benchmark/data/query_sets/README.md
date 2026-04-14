Place benchmark case manifests here as JSON files.

Each manifest must be a JSON array. Each entry must contain:
- query_id: string
- native_query: relative path under queries/
- baseline_vector_query: relative path under queries/
- baseline_traversal_query: relative path under queries/
- params: object passed to the query files

Optional:
- enabled: boolean, defaults to true
- native_config: object merged into the default rag.retrieve config for this case
