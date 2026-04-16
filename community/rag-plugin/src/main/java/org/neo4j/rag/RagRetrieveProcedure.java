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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
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
            @Name(value = "parallelism", defaultValue = "0") long parallelism)
            throws KernelException {

        // 1. Resolve vector index by name
        IndexDescriptor index = ktx.schemaRead().indexGetForName(indexName);

        // 2. Convert List<Double> to float[]
        float[] queryVector = new float[embedding.size()];
        for (int i = 0; i < embedding.size(); i++) {
            queryVector[i] = embedding.get(i).floatValue();
        }

        // 3. Kernel-level vector search (bypasses Cypher entirely)
        IndexReadSession indexSession = ktx.dataRead().indexReadSession(index);
        NodeValueIndexCursor cursor =
                ktx.cursors().allocateNodeValueIndexCursor(ktx.cursorContext(), ktx.memoryTracker());

        ktx.dataRead()
                .nodeIndexSeek(
                        ktx.queryContext(),
                        indexSession,
                        cursor,
                        IndexQueryConstraints.unconstrained(),
                        PropertyIndexQuery.nearestNeighbors(Math.toIntExact(topK), queryVector));

        // Collect seed nodes with their similarity scores
        Set<Long> seedIds = new LinkedHashSet<>();
        Map<Long, Double> seedScores = new LinkedHashMap<>();
        while (cursor.next()) {
            long nodeId = cursor.nodeReference();
            double score = MathUtil.clamp(cursor.score(), 0.0, 1.0);
            seedIds.add(nodeId);
            seedScores.put(nodeId, score);
        }
        cursor.close();

        // 4. BFS traversal from seed nodes (bypasses Cypher entirely); parallel frontier expansion when parallelism != 1
        Map<Long, Integer> visited = ParallelBfsTraversal.run(
                seedIds, Math.toIntExact(depth), ktx, Math.toIntExact(parallelism));

        // 5. Build result stream
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
}
