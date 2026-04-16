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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.neo4j.harness.Neo4j;
import org.neo4j.harness.Neo4jBuilders;

class RagRetrieveProcedureIT {

    @Test
    void shouldReturnSeedNodesAndNeighbors() {
        try (Neo4j neo4j = Neo4jBuilders.newInProcessBuilder()
                .withProcedure(RagRetrieveProcedure.class)
                .withFixture(
                        """
                    CREATE (a:Post {content: 'hello', embedding: [1.0, 0.0, 0.0]})
                    CREATE (b:Post {content: 'world', embedding: [0.0, 1.0, 0.0]})
                    CREATE (c:Post {content: 'foo',   embedding: [0.0, 0.0, 1.0]})
                    CREATE (d:Post {content: 'bar'})
                    CREATE (a)-[:RELATED]->(b)
                    CREATE (b)-[:RELATED]->(c)
                    CREATE (c)-[:RELATED]->(d)
                    """)
                .withFixture(
                        "CREATE VECTOR INDEX postEmbeddings FOR (n:Post) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 3, `vector.similarity_function`: 'cosine'}}")
                .build()) {

            // Wait for index to come online
            try (var tx = neo4j.defaultDatabaseService().beginTx()) {
                tx.execute("CALL db.awaitIndexes(120)");
                tx.commit();
            }

            // Call our procedure: topK=1 (nearest to [1,0,0] = node a), depth=2
            try (var tx = neo4j.defaultDatabaseService().beginTx()) {
                var result = tx.execute("CALL rag.retrieve('postEmbeddings', [1.0, 0.0, 0.0], 1, 2) "
                        + "YIELD node, hopDepth, score "
                        + "RETURN node.content AS content, hopDepth, score "
                        + "ORDER BY hopDepth, content");

                List<Map<String, Object>> rows = result.stream().toList();

                // Seed: a (hop 0), Neighbors: b (hop 1), c (hop 2)
                assertThat(rows).hasSize(3);
                assertThat(rows.get(0).get("content")).isEqualTo("hello");
                assertThat(rows.get(0).get("hopDepth")).isEqualTo(0L);
                assertThat((double) rows.get(0).get("score")).isGreaterThan(0.0);
                assertThat(rows.get(1).get("content")).isEqualTo("world");
                assertThat(rows.get(1).get("hopDepth")).isEqualTo(1L);
                assertThat(rows.get(2).get("content")).isEqualTo("foo");
                assertThat(rows.get(2).get("hopDepth")).isEqualTo(2L);

                tx.commit();
            }
        }
    }

    @Test
    void shouldMatchStandardTwoTripApproach() {
        try (Neo4j neo4j = Neo4jBuilders.newInProcessBuilder()
                .withProcedure(RagRetrieveProcedure.class)
                .withFixture(
                        """
                    CREATE (a:Post {content: 'alpha', embedding: [1.0, 0.0, 0.0]})
                    CREATE (b:Post {content: 'beta',  embedding: [0.9, 0.1, 0.0]})
                    CREATE (c:Post {content: 'gamma', embedding: [0.0, 0.0, 1.0]})
                    CREATE (a)-[:LINK]->(b)
                    CREATE (b)-[:LINK]->(c)
                    """)
                .withFixture(
                        "CREATE VECTOR INDEX idx FOR (n:Post) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 3, `vector.similarity_function`: 'cosine'}}")
                .build()) {

            try (var tx = neo4j.defaultDatabaseService().beginTx()) {
                tx.execute("CALL db.awaitIndexes(120)");
                tx.commit();
            }

            // Our single-call approach: topK=2 seeds, depth=1
            List<Map<String, Object>> allRows;
            try (var tx = neo4j.defaultDatabaseService().beginTx()) {
                allRows = tx
                        .execute("CALL rag.retrieve('idx', [1.0, 0.0, 0.0], 2, 1) "
                                + "YIELD node, hopDepth, score "
                                + "RETURN node.content AS content, hopDepth, score "
                                + "ORDER BY hopDepth, content")
                        .stream()
                        .toList();
                tx.commit();
            }

            // topK=2 should find alpha (score~1.0) and beta (score~0.99) as seeds
            List<String> seeds = allRows.stream()
                    .filter(r -> (long) r.get("hopDepth") == 0L)
                    .map(r -> (String) r.get("content"))
                    .toList();
            assertThat(seeds).containsExactly("alpha", "beta");

            // 1-hop neighbors from {alpha, beta} should include gamma (via beta->gamma)
            List<String> neighbors = allRows.stream()
                    .filter(r -> (long) r.get("hopDepth") == 1L)
                    .map(r -> (String) r.get("content"))
                    .toList();
            assertThat(neighbors).containsExactly("gamma");
        }
    }

    @Test
    void parallelTraversalShouldMatchSequentialOnWideFrontier() {
        StringBuilder create = new StringBuilder();
        create.append("CREATE (hub:Post {content: 'hub', embedding: [1.0, 0.0, 0.0]})\n");
        for (int i = 0; i < 32; i++) {
            create.append(
                    "CREATE (leaf")
                            .append(i)
                            .append(":Post {content: 'leaf")
                            .append(i)
                            .append("', embedding: [0.1, 0.9, 0.0]})\n");
            create.append("CREATE (hub)-[:SPOKE]->(leaf").append(i).append(")\n");
        }

        try (Neo4j neo4j = Neo4jBuilders.newInProcessBuilder()
                .withProcedure(RagRetrieveProcedure.class)
                .withFixture(create.toString())
                .withFixture(
                        "CREATE VECTOR INDEX postEmbeddings FOR (n:Post) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 3, `vector.similarity_function`: 'cosine'}}")
                .build()) {

            try (var tx = neo4j.defaultDatabaseService().beginTx()) {
                tx.execute("CALL db.awaitIndexes(120)");
                tx.commit();
            }

            List<Map<String, Object>> sequential = runRetrieve(neo4j, 1);
            List<Map<String, Object>> parallel = runRetrieve(neo4j, 8);

            assertThat(normalizeRows(sequential)).isEqualTo(normalizeRows(parallel));
        }
    }

    private static List<Map<String, Object>> runRetrieve(Neo4j neo4j, int parallelism) {
        try (var tx = neo4j.defaultDatabaseService().beginTx()) {
            var result = tx.execute(
                    "CALL rag.retrieve('postEmbeddings', [1.0, 0.0, 0.0], 5, 1, $p) "
                            + "YIELD node, hopDepth, score "
                            + "RETURN node.content AS content, hopDepth, score",
                    Map.of("p", (long) parallelism));
            List<Map<String, Object>> rows = result.stream().toList();
            tx.commit();
            return rows;
        }
    }

    /**
     * Stable comparison: hop depth, then content. (Parallel BFS may return rows in different order.)
     */
    private static List<Map<String, Object>> normalizeRows(List<Map<String, Object>> rows) {
        return rows.stream()
                .sorted(Comparator.comparing((Map<String, Object> r) -> (Long) r.get("hopDepth"))
                        .thenComparing(r -> (String) r.get("content")))
                .map(r -> {
                    Map<String, Object> copy = new HashMap<>();
                    copy.put("content", r.get("content"));
                    copy.put("hopDepth", r.get("hopDepth"));
                    copy.put("score", r.get("score"));
                    return copy;
                })
                .collect(Collectors.toList());
    }
}
