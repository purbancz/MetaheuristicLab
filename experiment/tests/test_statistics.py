"""Integration tests for the cross-problem statistics in experiment/retrieve.py.

Synthetic campaigns are written through the real HDF5 writer and read through
the real loader. Algorithm A is constructed to beat B, which beats C, on every
problem; per-problem fitness SCALES differ wildly, which the rank-based
statistics must ignore (the old min-max-normalized-means pipeline did not).
"""

import numpy as np
import pytest

import experiment.runner as runner
import experiment.retrieve as retrieve

ALGOS = {"A": 1.0, "B": 2.0, "C": 3.0}
N_PROBLEMS = 8
N_RUNS = 4


def _write_campaign(directory, scale_per_problem=False):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(N_PROBLEMS):
        problem_name = f"P{i}"
        scale = 10.0 ** i if scale_per_problem else 1.0
        for algo, base in ALGOS.items():
            finals = (base + rng.uniform(-0.1, 0.1, size=N_RUNS)) * scale
            curves = np.column_stack([finals * 3, finals * 2, finals])
            result = {
                "data": curves,
                "final_fitness": finals,
                "seeds": list(range(N_RUNS)),
                "run_times": np.ones(N_RUNS),
                "avg_fitness": float(np.mean(finals)),
                "std_dev": float(np.std(finals)),
                "avg_time": 1.0,
            }
            path = runner._h5_path(str(directory), problem_name, algo, 3, N_RUNS)
            runner._write_h5_algo_result(path, problem_name, algo, result, 3, N_RUNS)
    return retrieve.collect_h5_files_from_paths([str(directory)])


def test_all_vs_all_uses_ranks_and_reports_effect_sizes(tmp_path):
    files = _write_campaign(tmp_path)
    out = retrieve.all_vs_all_algorithm_stats(files)

    assert out["algorithms"] == ["A", "B", "C"]
    assert out["n_cases"] == N_PROBLEMS
    # Mean ranks are the effect size: A strictly best, C strictly worst.
    assert out["mean_ranks"]["A"] == pytest.approx(1.0)
    assert out["mean_ranks"]["C"] == pytest.approx(3.0)
    assert out["friedman_p_value"] < 0.05
    # Blocked (Friedman-consistent) post-hoc separates the extremes.
    assert out["nemenyi_p_values"].loc["A", "C"] < 0.05


def test_all_vs_all_is_scale_invariant_across_problems(tmp_path):
    plain = retrieve.all_vs_all_algorithm_stats(
        _write_campaign(tmp_path / "plain", scale_per_problem=False)
    )
    scaled = retrieve.all_vs_all_algorithm_stats(
        _write_campaign(tmp_path / "scaled", scale_per_problem=True)
    )

    # Multiplying each problem's fitness by a different power of 10 must not
    # move rank-based statistics at all.
    assert scaled["mean_ranks"] == plain["mean_ranks"]
    assert scaled["friedman_p_value"] == pytest.approx(plain["friedman_p_value"])


def test_all_vs_all_fails_loudly_without_common_roster(tmp_path):
    files = _write_campaign(tmp_path)
    with pytest.raises(ValueError):
        retrieve.all_vs_all_algorithm_stats(files, algos_to_compare=["A"])


def test_many_to_one_vs_baseline_counts_wins_and_rejects(tmp_path):
    files = _write_campaign(tmp_path)
    out = retrieve.many_to_one_vs_baseline(files, ["B", "C"], baseline="A")

    by_algo = {r["algo"]: r for r in out["results"]}
    for algo in ("B", "C"):
        assert by_algo[algo]["N"] == N_PROBLEMS
        assert by_algo[algo]["wins"] == 0
        assert by_algo[algo]["losses"] == N_PROBLEMS
        assert by_algo[algo]["direction"] == "WORSE"
    assert all(out["reject"])


def test_group_friedman_with_holm_ranks_groups(tmp_path):
    files = _write_campaign(tmp_path)
    out = retrieve.friedman_wilcoxon_algorithm_groups_with_holm(
        files, {"A": "GroupA", "B": "GroupB", "C": "GroupC"}
    )

    assert out["n_cases"] == N_PROBLEMS
    assert out["friedman_significant"]
    # Group scores are mean ranks: GroupA must hold rank 1 on every problem.
    assert np.allclose(out["group_scores_per_problem"]["GroupA"], 1.0)
    assert np.allclose(out["group_scores_per_problem"]["GroupC"], 3.0)
    for pair in out["pairwise_results"]:
        assert pair["significant_holm"]
        assert pair["lower_mean_group"] == min(
            (pair["group_1"], pair["group_2"]),
            key=lambda g: {"GroupA": 1, "GroupB": 2, "GroupC": 3}[g],
        )


def test_head_to_head_uses_sign_test(tmp_path, capsys):
    files = _write_campaign(tmp_path)
    retrieve.head_to_head_champions(files, algo_A="A", algo_B="C")
    printed = capsys.readouterr().out

    assert "Wins A: 8 | Wins C: 0 | Ties: 0" in printed
    assert "A wins on significantly more problems" in printed
