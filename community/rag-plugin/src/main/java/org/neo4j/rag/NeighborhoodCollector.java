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

import java.util.LinkedHashSet;
import java.util.Set;
import org.neo4j.internal.kernel.api.CursorFactory;
import org.neo4j.internal.kernel.api.NodeCursor;
import org.neo4j.internal.kernel.api.Read;
import org.neo4j.internal.kernel.api.RelationshipTraversalCursor;
import org.neo4j.io.pagecache.context.CursorContext;
import org.neo4j.memory.MemoryTracker;

/**
 * Collects a bounded local neighborhood for a node using Neo4j's internal
 * cursor API. The reranker uses this neighborhood as a cheap approximation of
 * the graph region that BFS would expand into.
 */
final class NeighborhoodCollector {
    private NeighborhoodCollector() {}

    /**
     * Collect the 1-hop neighbor ids for a candidate seed.
     *
     * @param nodeId the candidate seed whose neighborhood should be inspected
     * @param maxNeighborhoodSize hard cap to avoid spending too much time on very high-degree nodes
     * @return a bounded set of direct neighbor ids
     */
    static Set<Long> collectOneHop(
            long nodeId,
            int maxNeighborhoodSize,
            Read read,
            CursorFactory cursors,
            CursorContext cursorContext,
            MemoryTracker memoryTracker) {
        Set<Long> neighborhood = new LinkedHashSet<>();

        if (maxNeighborhoodSize <= 0) {
            return neighborhood;
        }

        try (NodeCursor nodeCursor = cursors.allocateNodeCursor(cursorContext, memoryTracker);
                RelationshipTraversalCursor relCursor =
                        cursors.allocateRelationshipTraversalCursor(cursorContext, memoryTracker)) {
            // Position the node cursor on the candidate seed we want to inspect.
            read.singleNode(nodeId, nodeCursor);
            if (!nodeCursor.next()) {
                return neighborhood;
            }

            // Walk outgoing and incoming relationships and collect the other endpoint ids.
            nodeCursor.relationships(relCursor, ALL_RELATIONSHIPS);
            while (relCursor.next() && neighborhood.size() < maxNeighborhoodSize) {
                neighborhood.add(relCursor.otherNodeReference());
            }
        }

        return neighborhood;
    }
}
