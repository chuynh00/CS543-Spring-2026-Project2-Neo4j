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

import java.util.Map;

/**
 * Runtime knobs for graph-aware seed reranking.
 *
 * <p>The defaults keep the feature usable without any extra arguments, while
 * still allowing experiments with oversampling and overlap sensitivity.
 */
record RerankConfig(
        int oversampleFactor,
        double overlapPenaltyWeight,
        int maxCandidateK,
        int maxNeighborhoodSize,
        int overlapHop,
        int parallelism) {
    private static final int DEFAULT_OVERSAMPLE_FACTOR = 5;
    private static final double DEFAULT_OVERLAP_PENALTY_WEIGHT = 0.3;
    private static final int DEFAULT_MAX_CANDIDATE_K = 100;
    private static final int DEFAULT_MAX_NEIGHBORHOOD_SIZE = 200;
    private static final int DEFAULT_OVERLAP_HOP = 1;
    /** 0 = use {@link Runtime#availableProcessors()}; 1 = sequential BFS; &gt; 1 caps parallel workers. */
    private static final int DEFAULT_PARALLELISM = 0;

    /**
     * Build a config object from the procedure's config map, filling in any
     * omitted values with v1 defaults.
     */
    static RerankConfig from(Map<String, Object> config) {
        Map<String, Object> safeConfig = config == null ? Map.of() : config;

        // Read user-supplied overrides while falling back to the defaults above.
        int oversampleFactor = intValue(safeConfig, "oversampleFactor", DEFAULT_OVERSAMPLE_FACTOR);
        double overlapPenaltyWeight =
                doubleValue(safeConfig, "overlapPenaltyWeight", DEFAULT_OVERLAP_PENALTY_WEIGHT);
        int maxCandidateK = intValue(safeConfig, "maxCandidateK", DEFAULT_MAX_CANDIDATE_K);
        int maxNeighborhoodSize = intValue(safeConfig, "maxNeighborhoodSize", DEFAULT_MAX_NEIGHBORHOOD_SIZE);
        int overlapHop = intValue(safeConfig, "overlapHop", DEFAULT_OVERLAP_HOP);
        int parallelism = intValue(safeConfig, "parallelism", DEFAULT_PARALLELISM);

        // Fail and throw an exception on invalid inputs so the procedure does not run with an invalid configuration.
        if (oversampleFactor < 1) {
            throw new IllegalArgumentException("'oversampleFactor' must be at least 1");
        }
        if (overlapPenaltyWeight < 0.0) {
            throw new IllegalArgumentException("'overlapPenaltyWeight' must be non-negative");
        }
        if (maxCandidateK < 1) {
            throw new IllegalArgumentException("'maxCandidateK' must be at least 1");
        }
        if (maxNeighborhoodSize < 0) {
            throw new IllegalArgumentException("'maxNeighborhoodSize' must be non-negative");
        }
        if (overlapHop != 1) {
            throw new IllegalArgumentException("Only 'overlapHop' = 1 is supported in this version");
        }
        if (parallelism < 0) {
            throw new IllegalArgumentException("'parallelism' must be 0 (auto), 1 (sequential), or a positive worker cap");
        }

        return new RerankConfig(
                oversampleFactor,
                overlapPenaltyWeight,
                maxCandidateK,
                maxNeighborhoodSize,
                overlapHop,
                parallelism);
    }

    /**
     * Parse an integer-valued config entry.
     */
    private static int intValue(Map<String, Object> config, String key, int defaultValue) {
        Object rawValue = config.get(key);
        if (rawValue == null) {
            return defaultValue;
        }
        if (rawValue instanceof Number number) {
            return number.intValue();
        }
        throw new IllegalArgumentException("'" + key + "' must be numeric");
    }

    /**
     * Parse a floating-point config entry.
     */
    private static double doubleValue(Map<String, Object> config, String key, double defaultValue) {
        Object rawValue = config.get(key);
        if (rawValue == null) {
            return defaultValue;
        }
        if (rawValue instanceof Number number) {
            return number.doubleValue();
        }
        throw new IllegalArgumentException("'" + key + "' must be numeric");
    }
}
