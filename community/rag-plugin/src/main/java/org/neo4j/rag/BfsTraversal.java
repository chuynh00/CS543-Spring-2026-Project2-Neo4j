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

import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import org.neo4j.internal.kernel.api.CursorFactory;
import org.neo4j.internal.kernel.api.NodeCursor;
import org.neo4j.internal.kernel.api.Read;
import org.neo4j.internal.kernel.api.RelationshipTraversalCursor;
import org.neo4j.io.pagecache.context.CursorContext;
import org.neo4j.memory.MemoryTracker;

/**
 * BFS traversal using Neo4j internal cursor APIs (index-free adjacency).
 * Bypasses Cypher parsing and planning entirely.
 */
public class BfsTraversal {

    /**
     * BFS from seed nodes up to maxDepth hops.
     *
     * @return map of nodeId to hop depth (0 for seeds)
     */
    public static Map<Long, Integer> run(
            Set<Long> seedNodeIds,
            int maxDepth,
            Read read,
            CursorFactory cursors,
            CursorContext cursorContext,
            MemoryTracker memoryTracker) {

        Map<Long, Integer> visited = new LinkedHashMap<>();
        Queue<Long> currentFrontier = new ArrayDeque<>();

        for (long seedId : seedNodeIds) {
            visited.put(seedId, 0);
            currentFrontier.add(seedId);
        }

        try (NodeCursor nodeCursor = cursors.allocateNodeCursor(cursorContext, memoryTracker);
                RelationshipTraversalCursor relCursor =
                        cursors.allocateRelationshipTraversalCursor(cursorContext, memoryTracker)) {

            for (int hop = 1; hop <= maxDepth && !currentFrontier.isEmpty(); hop++) {
                Queue<Long> nextFrontier = new ArrayDeque<>();

                for (long nodeId : currentFrontier) {
                    read.singleNode(nodeId, nodeCursor);
                    if (nodeCursor.next()) {
                        nodeCursor.relationships(relCursor, ALL_RELATIONSHIPS);
                        while (relCursor.next()) {
                            long neighborId = relCursor.otherNodeReference();
                            if (!visited.containsKey(neighborId)) {
                                visited.put(neighborId, hop);
                                nextFrontier.add(neighborId);
                            }
                        }
                    }
                }

                currentFrontier = nextFrontier;
            }
        }

        return visited;
    }
}
