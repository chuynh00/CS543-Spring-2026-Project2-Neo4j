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

import static org.neo4j.storageengine.api.RelationshipSelection.ALL_RELATIONSHIPS;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.neo4j.internal.kernel.api.CursorFactory;
import org.neo4j.internal.kernel.api.NodeCursor;
import org.neo4j.internal.kernel.api.Read;
import org.neo4j.internal.kernel.api.RelationshipTraversalCursor;
import org.neo4j.io.pagecache.context.CursorContext;
import org.neo4j.kernel.api.ExecutionContext;
import org.neo4j.kernel.api.KernelTransaction;
import org.neo4j.memory.MemoryTracker;

/**
 * Breadth-first expansion using index-free adjacency, with optional parallel
 * frontier expansion. Each worker uses its own {@link NodeCursor} and
 * {@link RelationshipTraversalCursor} via a dedicated {@link ExecutionContext};
 * discovered nodes are merged in a {@link ConcurrentHashMap}.
 */
public final class ParallelBfsTraversal {

    private static final int AWAIT_TERMINATION_MINUTES = 10;

    private ParallelBfsTraversal() {}

    /**
     * @param parallelism use {@code 1} for single-threaded traversal (same behavior as
     *     {@link BfsTraversal}); use {@code 0} or less to choose a thread count from
     *     {@link Runtime#availableProcessors()}; otherwise caps worker threads at this value.
     */
    public static Map<Long, Integer> run(
            Set<Long> seedNodeIds, int maxDepth, KernelTransaction ktx, int parallelism) {
        if (parallelism == 1) {
            return BfsTraversal.run(
                    seedNodeIds,
                    maxDepth,
                    ktx.dataRead(),
                    ktx.cursors(),
                    ktx.cursorContext(),
                    ktx.memoryTracker());
        }

        int maxWorkers =
                parallelism <= 0 ? Runtime.getRuntime().availableProcessors() : parallelism;

        ConcurrentHashMap<Long, Integer> visited = new ConcurrentHashMap<>();
        for (long seedId : seedNodeIds) {
            visited.put(seedId, 0);
        }

        List<Long> currentFrontier = new ArrayList<>(seedNodeIds);
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, maxWorkers));

        try {
            for (int hop = 1; hop <= maxDepth && !currentFrontier.isEmpty(); hop++) {
                int n = currentFrontier.size();
                int workers = Math.min(Math.max(1, maxWorkers), n);
                ConcurrentLinkedQueue<Long> nextFrontier = new ConcurrentLinkedQueue<>();

                if (workers <= 1) {
                    expandPartition(
                            ktx.dataRead(),
                            ktx.cursors(),
                            ktx.cursorContext(),
                            ktx.memoryTracker(),
                            currentFrontier,
                            0,
                            n,
                            hop,
                            visited,
                            nextFrontier);
                } else {
                    parallelExpandHop(
                            ktx,
                            currentFrontier,
                            n,
                            workers,
                            hop,
                            visited,
                            nextFrontier,
                            pool);
                }

                currentFrontier = new ArrayList<>(nextFrontier);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel BFS interrupted", e);
        } finally {
            pool.shutdown();
            try {
                if (!pool.awaitTermination(AWAIT_TERMINATION_MINUTES, TimeUnit.MINUTES)) {
                    pool.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                pool.shutdownNow();
            }
        }

        return visited;
    }

    private static void parallelExpandHop(
            KernelTransaction ktx,
            List<Long> frontier,
            int n,
            int workers,
            int hop,
            ConcurrentHashMap<Long, Integer> visited,
            ConcurrentLinkedQueue<Long> nextFrontier,
            ExecutorService pool)
            throws InterruptedException {

        int chunkSize = (n + workers - 1) / workers;
        CountDownLatch done = new CountDownLatch(workers);
        List<ExecutionContext> contexts = new ArrayList<>(workers);

        for (int w = 0; w < workers; w++) {
            int start = w * chunkSize;
            if (start >= n) {
                done.countDown();
                continue;
            }
            int end = Math.min(n, start + chunkSize);
            ExecutionContext ec = ktx.createExecutionContext();
            contexts.add(ec);
            pool.execute(() -> {
                try {
                    ec.performCheckBeforeOperation();
                    expandPartition(
                            ec.dataRead(),
                            ec.cursors(),
                            ec.cursorContext(),
                            ec.memoryTracker(),
                            frontier,
                            start,
                            end,
                            hop,
                            visited,
                            nextFrontier);
                } finally {
                    ec.complete();
                    done.countDown();
                }
            });
        }

        done.await();
        for (ExecutionContext ec : contexts) {
            ec.close();
        }
    }

    private static void expandPartition(
            Read read,
            CursorFactory cursors,
            CursorContext cursorContext,
            MemoryTracker memoryTracker,
            List<Long> frontier,
            int start,
            int end,
            int hop,
            ConcurrentHashMap<Long, Integer> visited,
            ConcurrentLinkedQueue<Long> nextFrontier) {

        try (NodeCursor nodeCursor = cursors.allocateNodeCursor(cursorContext, memoryTracker);
                RelationshipTraversalCursor relCursor =
                        cursors.allocateRelationshipTraversalCursor(cursorContext, memoryTracker)) {

            for (int i = start; i < end; i++) {
                long nodeId = frontier.get(i);
                read.singleNode(nodeId, nodeCursor);
                if (!nodeCursor.next()) {
                    continue;
                }
                nodeCursor.relationships(relCursor, ALL_RELATIONSHIPS);
                while (relCursor.next()) {
                    long neighborId = relCursor.otherNodeReference();
                    if (visited.putIfAbsent(neighborId, hop) == null) {
                        nextFrontier.offer(neighborId);
                    }
                }
            }
        }
    }
}
