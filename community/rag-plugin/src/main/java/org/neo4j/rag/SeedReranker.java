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
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Greedy graph-aware seed reranker.
 *
 * <p>The reranker keeps the highest ANN-scoring seed first, then fills the
 * remaining slots with candidates that stay relevant while avoiding strong
 * neighborhood overlap with already selected seeds.
 */
final class SeedReranker {
    private static final Comparator<CandidateSeed> CANDIDATE_ORDER =
            Comparator.comparingDouble(CandidateSeed::annScore).reversed().thenComparingLong(CandidateSeed::nodeId);

    private SeedReranker() {}

    /**
     * Select the final seed set from an oversampled vector-search candidate pool.
     *
     * @param candidates ANN candidates enriched with local graph neighborhoods
     * @param topK number of final seeds to keep
     * @param config reranking knobs
     * @return the final graph-aware seed set
     */
    static List<CandidateSeed> selectTopK(List<CandidateSeed> candidates, int topK, RerankConfig config) {
        if (candidates.isEmpty() || topK <= 0) {
            return List.of();
        }

        // Start from ANN order so the strongest semantic matches are considered first.
        List<CandidateSeed> orderedCandidates = new ArrayList<>(candidates);
        orderedCandidates.sort(CANDIDATE_ORDER);

        List<CandidateSeed> selected = new ArrayList<>();
        Set<Long> selectedNodeIds = new HashSet<>();

        // The first seed has no overlap penalty yet, so we keep the best ANN hit.
        CandidateSeed firstSeed = orderedCandidates.get(0);
        selected.add(firstSeed);
        selectedNodeIds.add(firstSeed.nodeId());

        while (selected.size() < topK && selected.size() < orderedCandidates.size()) {
            CandidateSeed bestCandidate = null;
            double bestScore = Double.NEGATIVE_INFINITY;

            // for each candidate, compute a rerank score that penalizes neighborhood overlap
            for (CandidateSeed candidate : orderedCandidates) {
                if (selectedNodeIds.contains(candidate.nodeId())) {
                    continue;
                }

                // Discount candidates that would lead BFS into a neighborhood already covered by a chosen seed.
                double overlap = maxOverlap(candidate, selected);
                double rerankScore = candidate.annScore() - config.overlapPenaltyWeight() * overlap;

                // Prefer the best rerank score, then break ties by ANN score and node id for deterministic output.
                if (bestCandidate == null
                        || rerankScore > bestScore
                        || (rerankScore == bestScore
                                && candidate.annScore() > bestCandidate.annScore())
                        || (rerankScore == bestScore
                                && candidate.annScore() == bestCandidate.annScore()
                                && candidate.nodeId() < bestCandidate.nodeId())) {
                    bestCandidate = candidate; // Update the best candidate found so far
                    bestScore = rerankScore; // Update the best score found so far
                }
            }

            if (bestCandidate == null) {
                break;
            }

            selected.add(bestCandidate);
            selectedNodeIds.add(bestCandidate.nodeId());
        }

        return selected;
    }

    /**
     * Compute the strongest neighborhood-overlap penalty against any selected seed.
     */
    private static double maxOverlap(CandidateSeed candidate, List<CandidateSeed> selected) {
        double maxOverlap = 0.0;
        for (CandidateSeed selectedSeed : selected) {
            maxOverlap = Math.max(maxOverlap, jaccard(candidate.neighborhood(), selectedSeed.neighborhood()));
        }
        return maxOverlap;
    }

    /**
     * Compute Jaccard overlap between two neighborhood sets.
     */
    private static double jaccard(Set<Long> left, Set<Long> right) {
        if (left.isEmpty() && right.isEmpty()) {
            return 0.0;
        }

        // Shared neighbors capture how much graph region two seeds have in common.
        Set<Long> intersection = new HashSet<>(left);
        intersection.retainAll(right);

        // The union normalizes the overlap so high-degree seeds are not automatically over-penalized.
        Set<Long> union = new HashSet<>(left);
        union.addAll(right);

        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
}
