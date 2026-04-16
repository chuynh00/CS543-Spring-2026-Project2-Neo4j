/*
 * Copyright (c) "Neo4j"
 * Neo4j Sweden AB [https://neo4j.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
package org.neo4j.rag;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.neo4j.exceptions.KernelException;
import org.neo4j.graphdb.NotFoundException;
import org.neo4j.graphdb.Transaction;
import org.neo4j.internal.helpers.MathUtil;
import org.neo4j.internal.kernel.api.IndexQueryConstraints;
import org.neo4j.internal.kernel.api.IndexReadSession;
import org.neo4j.internal.kernel.api.NodeValueIndexCursor;
import org.neo4j.internal.kernel.api.PropertyIndexQuery;
import org.neo4j.internal.schema.IndexDescriptor;
import org.neo4j.kernel.api.KernelTransaction;
import org.neo4j.kernel.internal.GraphDatabaseAPI;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

/**
 * Single-call RAG retrieval procedure that combines vector similarity search
 * with k-hop graph traversal inside the Neo4j kernel, bypassing Cypher
 * parsing and planning for both operations.
 */
public class RagRetrieveProcedure {

    @Context
    public KernelTransaction ktx;

    @Context
    public Transaction tx;

    @Context
    public GraphDatabaseAPI db;

    @Description("Single-call RAG retrieval: vector search + k-hop graph traversal in one kernel execution.")
    @Procedure(name = "rag.retrieve", mode = Mode.READ)
    public Stream<SubgraphRow> retrieve(
            @Name("indexName") String indexName,
            @Name("embedding") List<Double> embedding,
            @Name("topK") long topK,
            @Name("depth") long depth,
            @Name(value = "config", defaultValue = "{}") Map<String, Object> config)
            throws KernelException {

        // 1. Resolve the vector index the caller wants to search.
        IndexDescriptor index = ktx.schemaRead().indexGetForName(indexName);

        // 2. Convert the input embedding into the Lucene/Neo4j vector format and size the candidate pool.
        float[] queryVector = toFloatArray(embedding);
        int requestedTopK = Math.toIntExact(topK);
        RerankConfig rerankConfig = RerankConfig.from(config);
        int candidateK = candidateCount(requestedTopK, rerankConfig);

        // 3. Run ANN search first; this gives us a candidate pool rather than the final seed set.
        List<CandidateSeed> candidates = vectorSearch(index, queryVector, candidateK);
        if (candidates.isEmpty()) {
            return Stream.empty();
        }

        // 4. Enrich each ANN candidate with a bounded local neighborhood for overlap scoring.
        List<CandidateSeed> enrichedCandidates = new ArrayList<>(candidates.size());
        for (CandidateSeed candidate : candidates) {
            Set<Long> neighborhood = NeighborhoodCollector.collectOneHop(
                    candidate.nodeId(),
                    rerankConfig.maxNeighborhoodSize(),
                    ktx.dataRead(),
                    ktx.cursors(),
                    ktx.cursorContext(),
                    ktx.memoryTracker());
            enrichedCandidates.add(new CandidateSeed(candidate.nodeId(), candidate.annScore(), neighborhood));
        }

        // 5. Apply the graph-aware reranker to choose the final traversal seeds.
        List<CandidateSeed> selectedSeeds = SeedReranker.selectTopK(enrichedCandidates, requestedTopK, rerankConfig);
        Set<Long> seedIds = new LinkedHashSet<>();
        Map<Long, Double> seedScores = new LinkedHashMap<>();
        for (CandidateSeed seed : selectedSeeds) {
            seedIds.add(seed.nodeId());
            seedScores.put(seed.nodeId(), seed.annScore());
        }

        // 6. Traverse from the reranked seeds (parallel frontier expansion unless parallelism == 1).
        Map<Long, Integer> visited =
                ParallelBfsTraversal.run(seedIds, Math.toIntExact(depth), ktx, rerankConfig.parallelism());

        // 7. Convert the visited node ids back into procedure rows with hop depth and seed score.
        List<SubgraphRow> results = new ArrayList<>();
        for (Map.Entry<Long, Integer> entry : visited.entrySet()) {
            long nodeId = entry.getKey();
            int hopDepth = entry.getValue();
            double score = seedScores.getOrDefault(nodeId, 0.0);
            try {
                results.add(new SubgraphRow(tx.getNodeById(nodeId), hopDepth, score));
            } catch (NotFoundException ignored) {
                // Node deleted by concurrent transaction — skip it
            }
        }
        return results.stream();
    }

    /**
     * Convert the procedure's boxed embedding input into the float array format
     * expected by Neo4j's vector index path.
     */
    private float[] toFloatArray(List<Double> embedding) {
        float[] queryVector = new float[embedding.size()];
        for (int i = 0; i < embedding.size(); i++) {
            queryVector[i] = embedding.get(i).floatValue();
        }
        return queryVector;
    }

    /**
     * Compute how many ANN candidates to request before graph-aware reranking.
     */
    private int candidateCount(int requestedTopK, RerankConfig config) {
        long oversampled = (long) requestedTopK * config.oversampleFactor();
        long requestedCandidateCount = Math.max(requestedTopK, oversampled);
        return (int) Math.min(config.maxCandidateK(), requestedCandidateCount);
    }

    /**
     * Run the kernel-level vector search and collect ANN candidates with their
     * normalized similarity scores.
     */
    private List<CandidateSeed> vectorSearch(IndexDescriptor index, float[] queryVector, int candidateK)
            throws KernelException {
        List<CandidateSeed> candidates = new ArrayList<>();
        IndexReadSession indexSession = ktx.dataRead().indexReadSession(index);
        NodeValueIndexCursor cursor =
                ktx.cursors().allocateNodeValueIndexCursor(ktx.cursorContext(), ktx.memoryTracker());

        try {
            // Ask Neo4j's vector index for an oversampled candidate pool.
            ktx.dataRead()
                    .nodeIndexSeek(
                            ktx.queryContext(),
                            indexSession,
                            cursor,
                            IndexQueryConstraints.unconstrained(),
                            PropertyIndexQuery.nearestNeighbors(candidateK, queryVector));

            // Preserve the ANN score; v1 reranking uses it as the base relevance term.
            while (cursor.next()) {
                long nodeId = cursor.nodeReference();
                double score = MathUtil.clamp(cursor.score(), 0.0, 1.0);
                candidates.add(new CandidateSeed(nodeId, score, Set.of()));
            }
        } finally {
            cursor.close();
        }

        return candidates;
    }
}
