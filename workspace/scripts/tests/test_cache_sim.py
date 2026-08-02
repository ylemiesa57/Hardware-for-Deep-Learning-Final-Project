"""Tests for cache_sim.py's trace generator and three cache-replacement policies.

This module had no test coverage at all before this change. The tests below
check the invariants the docstrings claim: every access is accounted for as
exactly one hit or one miss, the no-cache baseline never hits, and -- the
main correctness property worth pinning down -- Belady's OPT algorithm is a
lower bound on misses, so its hit rate should never be worse than the
cost-aware LFU policy's, which in turn should never be worse than not
caching at all, for the same trace and capacity.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from cache_sim import (
    simulate_no_cache,
    simulate_lfu_cost_aware,
    simulate_belady_opt,
    synth_trace,
)


def _make_trace():
    # Enough queries/docs/reuse that capacity-limited caching actually evicts
    # something, so the three policies can meaningfully differ.
    return synth_trace(n_queries=200, n_docs=500, reuse_ratio=3.0, k_per_query=10, seed=1)


class TestSynthTrace:
    def test_length_matches_queries_times_k(self):
        trace = synth_trace(n_queries=50, n_docs=200, reuse_ratio=2.0, k_per_query=10, seed=0)
        assert len(trace) == 50 * 10

    def test_each_query_draws_distinct_docs(self):
        n_queries, k = 20, 8
        trace = synth_trace(n_queries=n_queries, n_docs=100, reuse_ratio=2.0, k_per_query=k, seed=0)
        for q in range(n_queries):
            chunk = trace[q * k : (q + 1) * k]
            assert len(set(chunk)) == k, "each query must draw k distinct document ids"

    def test_doc_ids_within_corpus_range(self):
        trace = synth_trace(n_queries=30, n_docs=50, reuse_ratio=2.0, k_per_query=5, seed=2)
        assert all(0 <= d < 50 for d in trace)

    def test_reproducible_with_same_seed(self):
        t1 = synth_trace(n_queries=20, n_docs=100, reuse_ratio=2.0, k_per_query=5, seed=7)
        t2 = synth_trace(n_queries=20, n_docs=100, reuse_ratio=2.0, k_per_query=5, seed=7)
        assert t1 == t2


class TestSimulateNoCache:
    def test_all_misses(self):
        trace = _make_trace()
        result = simulate_no_cache(trace)
        assert result.hits == 0
        assert result.misses == len(trace)
        assert result.hit_rate == 0.0
        assert result.miss_positions == list(range(len(trace)))


class TestHitMissAccounting:
    def test_lfu_hits_plus_misses_equals_trace_length(self):
        trace = _make_trace()
        gen_latency = {d: 1.0 for d in set(trace)}
        result = simulate_lfu_cost_aware(trace, capacity=50, gen_latency=gen_latency)
        assert result.hits + result.misses == len(trace)
        assert result.hit_rate == result.hits / len(trace)

    def test_opt_hits_plus_misses_equals_trace_length(self):
        trace = _make_trace()
        result = simulate_belady_opt(trace, capacity=50)
        assert result.hits + result.misses == len(trace)
        assert result.hit_rate == result.hits / len(trace)

    def test_zero_capacity_behaves_like_no_cache(self):
        trace = _make_trace()
        gen_latency = {d: 1.0 for d in set(trace)}
        lfu_result = simulate_lfu_cost_aware(trace, capacity=0, gen_latency=gen_latency)
        opt_result = simulate_belady_opt(trace, capacity=0)
        assert lfu_result.hits == 0
        assert opt_result.hits == 0


class TestDecayActuallyStalesIdleEntries:
    """Regression test for the eviction-priority staleness bug.

    simulate_lfu_cost_aware() is documented as evicting
    argmin(gen_latency * counter) with per-step exponential decay applied to
    the counter. Before the fix, a doc's heap priority was frozen at
    gen_latency * base_count computed at the step it was last touched (i.e.
    decay's `elapsed` term pinned to 0 at push time), and was never revisited
    while the doc just sat in cache untouched. That let a doc which built up
    a large counter long ago, then went idle, keep out-ranking (and so
    out-surviving) fresher, single-touch docs indefinitely -- exactly
    backwards from what "decay" is supposed to buy the eviction policy.

    This drives the scenario entirely through the public simulate_lfu_cost_aware()
    API (no reaching into internals): doc 0 is touched 10x up front (building
    a base_count that decays to ~6.5), then goes idle for 400 steps while a
    long stream of distinct, single-touch docs cycles through a small cache,
    then the 5 most-recently-inserted docs from that stream are re-touched.
    With decay correctly taken into account, doc 0 should have long since
    been evicted in favor of fresher docs, so those 5 re-touches should all
    be cache hits. Under the bug, doc 0 permanently occupies one of the 5
    slots and, worse, its stale priority (~6.5) outranks every fresh doc's
    priority (1.0) at push time too, so none of the re-touched docs are
    still resident either -- all 5 come back as misses instead.
    """

    def test_heavily_accessed_then_idle_doc_stops_blocking_fresh_docs(self):
        n_fresh = 400
        capacity = 5
        last_five = list(range(n_fresh - 4, n_fresh + 1))
        trace = [0] * 10 + list(range(1, n_fresh + 1)) + last_five
        gen_latency = {d: 1.0 for d in range(n_fresh + 1)}

        result = simulate_lfu_cost_aware(
            trace, capacity=capacity, gen_latency=gen_latency, decay_factor=0.9
        )

        phase3_positions = range(len(trace) - 5, len(trace))
        phase3_hits = sum(1 for p in phase3_positions if p not in result.miss_positions)
        assert phase3_hits == 5, (
            f"expected all 5 recently-inserted docs to still be cached and "
            f"hit on re-touch, got {phase3_hits}/5 -- doc 0's decayed-long-ago "
            f"priority is still blocking fresher docs from the cache"
        )


class TestBeladyIsOptimal:
    """Belady's OPT should never do worse than a cache-limited heuristic or no cache."""

    def test_opt_hit_rate_at_least_lfu_hit_rate(self):
        trace = _make_trace()
        gen_latency = {d: 1.0 for d in set(trace)}
        for capacity in (10, 50, 150):
            lfu_result = simulate_lfu_cost_aware(trace, capacity=capacity, gen_latency=gen_latency)
            opt_result = simulate_belady_opt(trace, capacity=capacity)
            assert opt_result.hit_rate >= lfu_result.hit_rate - 1e-9, (
                f"OPT ({opt_result.hit_rate}) should be >= LFU ({lfu_result.hit_rate}) "
                f"at capacity={capacity}"
            )

    def test_opt_hit_rate_at_least_no_cache(self):
        trace = _make_trace()
        no_cache_result = simulate_no_cache(trace)
        for capacity in (10, 50, 150):
            opt_result = simulate_belady_opt(trace, capacity=capacity)
            assert opt_result.hit_rate >= no_cache_result.hit_rate

    def test_larger_capacity_never_hurts_opt(self):
        """Monotonicity: more cache capacity should never reduce OPT's hit rate."""
        trace = _make_trace()
        rates = [simulate_belady_opt(trace, capacity=c).hit_rate for c in (5, 20, 50, 100, 200)]
        assert rates == sorted(rates)
