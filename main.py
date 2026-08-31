from experiment.retrieve import friedman_wilcoxon_algorithm_groups, head_to_head_champions, all_vs_all_algorithm_stats, \
    many_to_one_vs_baseline, \
    extract_results_to_csv, wilcoxon_rank_sum_vs_baselines, \
    friedman_wilcoxon_algorithm_groups_with_holm, collect_h5_files_from_paths, plot_all_from_h5, \
    kruskal_wallis_with_posthoc, plot_combined_data_from_h5

if __name__ == "__main__":
    paths = [
        'experiment_results/dim100_runs2',
    ]

    h5_files = collect_h5_files_from_paths(paths)
    if not h5_files:
        raise SystemExit(
            f"No .h5 result files found under {paths} - nothing to analyze."
        )


    plot_all_from_h5(h5_files)
    plot_combined_data_from_h5(h5_files)

    kruskal_wallis_with_posthoc(h5_files)
    wilcoxon_rank_sum_vs_baselines(h5_files)

    algo_groups = {
        "FRAPSO": "FRAPSO",
        "HybridFullDisjointRestarterPSO": "HybridFullDisjointRestarterPSO",
        "HybridPartialDisjointRestarterPSO": "HybridPartialDisjointRestarterPSO",
        "HybridAdditiveRestarterPSO": "HybridAdditiveRestarterPSO",
    }

    wilcoxon_rank_sum_vs_baselines(
        h5_files,
        algo_groups=algo_groups,
        lower_is_better=True,
        alpha=0.05,
        print_examples=False
    )

    friedman_wilcoxon_algorithm_groups(h5_files, algo_groups)
    friedman_wilcoxon_algorithm_groups_with_holm(h5_files, algo_groups)
    head_to_head_champions(h5_files)
    all_vs_all_algorithm_stats(h5_files)

    algos_to_compare = [
        "FRAPSO",
        "HybridFullDisjointRestarterPSO",
        "HybridPartialDisjointRestarterPSO",
        "HybridAdditiveRestarterPSO",
    ]

    many_to_one_vs_baseline(h5_files, algos_to_compare)
    many_to_one_vs_baseline(h5_files, list(algo_groups.keys()))


    extract_results_to_csv(h5_files, output_prefix="all_algorithms_all_problems")
