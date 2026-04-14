from __future__ import annotations

from statistics import median



def ns_to_ms(duration_ns: int) -> float:
    """Convert a nanosecond duration into milliseconds."""

    return duration_ns / 1_000_000.0



def percentile(sorted_values: list[float], fraction: float) -> float:
    """Return a simple nearest-rank percentile from a pre-sorted list."""

    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty list")

    index = max(0, min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]



def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    """Create a compact latency summary for one benchmark method."""

    ordered = sorted(latencies_ms)
    if not ordered:
        return {
            "count": 0.0,
            "min_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": float(len(ordered)),
        "min_ms": ordered[0],
        "median_ms": median(ordered),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "max_ms": ordered[-1],
    }
