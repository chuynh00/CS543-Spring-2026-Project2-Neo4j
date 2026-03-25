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

import org.neo4j.graphdb.Node;
import org.neo4j.procedure.Description;

public class SubgraphRow {
    @Description("A node in the retrieved subgraph")
    public Node node;

    @Description("Hop distance from seed: 0 = vector search hit, 1+ = BFS neighbor")
    public long hopDepth;

    @Description("Cosine similarity score (only for seed nodes; 0.0 for traversed neighbors)")
    public double score;

    public SubgraphRow(Node node, long hopDepth, double score) {
        this.node = node;
        this.hopDepth = hopDepth;
        this.score = score;
    }
}
