import numpy as np


def combine_data(data_list):
    """
    Combine experiment result data while preserving problem dimensionality.

    Results are aggregated only when both problem name and dimensionality
    match. Therefore, e.g. ("Rastrigin", 100) and ("Rastrigin", 500)
    are treated as separate experimental instances.

    The number of runs is stored separately:
    - for every problem/dimension instance,
    - for every algorithm within that instance.

    Returns
    -------
    dict
        Aggregated data keyed by:

            (problem_name, n_vars)

        Example:

            {
                ("Rastrigin", 100): {
                    "problem": "Rastrigin",
                    "n_vars": 100,
                    "runs": 50,
                    "results": {
                        "PSO": {
                            "data": ...,
                            "runs": 50,
                            "avg_fitness": ...,
                            "std_dev": ...,
                            "avg_time": ...,
                        }
                    }
                }
            }
    """

    combined_data = {}

    valid_data_list = [
        data
        for data in data_list
        if data is not None
    ]

    if not valid_data_list:
        print("Warning: No valid data loaded from pickle files.")
        return {}

    # ---------------------------------------------------------
    # Collect raw result arrays
    # ---------------------------------------------------------

    for data_source in valid_data_list:

        if isinstance(data_source, list):
            problems_in_source = data_source

        elif (
            isinstance(data_source, dict)
            and "problem" in data_source
            and "results" in data_source
        ):
            problems_in_source = [data_source]

        else:
            print(
                "Warning: Skipping unrecognized data source structure: "
                f"{type(data_source)}"
            )
            continue

        for problem_data in problems_in_source:

            if (
                not isinstance(problem_data, dict)
                or "problem" not in problem_data
                or "results" not in problem_data
            ):
                print(
                    "Warning: Skipping invalid problem data entry: "
                    f"{problem_data}"
                )
                continue

            problem_name = problem_data["problem"]
            n_vars = problem_data.get("n_vars", -1)

            # Old result files may contain a NumPy integer.
            if isinstance(n_vars, np.integer):
                n_vars = int(n_vars)

            results = problem_data["results"]

            # Problem dimensionality is part of experimental identity.
            instance_key = (problem_name, n_vars)

            if instance_key not in combined_data:
                combined_data[instance_key] = {
                    "problem": problem_name,
                    "n_vars": n_vars,
                    "results": {},
                }

            for algo, algo_data_in in results.items():

                if (
                    not isinstance(algo_data_in, dict)
                    or "data" not in algo_data_in
                ):
                    print(
                        f"Warning: Skipping invalid algorithm data for "
                        f"'{algo}' in problem '{problem_name}', "
                        f"dimension {n_vars}."
                    )
                    continue

                if algo not in combined_data[instance_key]["results"]:
                    combined_data[instance_key]["results"][algo] = {
                        "data_list": [],
                        "avg_fitness_list": [],
                        "std_dev_list": [],
                        "avg_time_list": [],
                        "final_fitness_list": [],
                        "run_times_list": [],
                        "seeds_list": [],
                    }

                collected = combined_data[instance_key]["results"][algo]

                if isinstance(algo_data_in["data"], np.ndarray):
                    collected["data_list"].append(
                        algo_data_in["data"]
                    )
                else:
                    print(
                        f"Warning: Result data for '{algo}' in "
                        f"problem '{problem_name}', dimension {n_vars} "
                        f"is not a NumPy array."
                    )

                if "avg_fitness" in algo_data_in:
                    collected["avg_fitness_list"].append(
                        algo_data_in["avg_fitness"]
                    )

                if "std_dev" in algo_data_in:
                    collected["std_dev_list"].append(
                        algo_data_in["std_dev"]
                    )

                if "avg_time" in algo_data_in:
                    collected["avg_time_list"].append(
                        algo_data_in["avg_time"]
                    )

                if "final_fitness" in algo_data_in:
                    arr = np.asarray(algo_data_in["final_fitness"])
                    if arr.ndim == 1 and arr.size > 0:
                        collected["final_fitness_list"].append(arr)

                if "run_times" in algo_data_in:
                    arr = np.asarray(algo_data_in["run_times"])
                    if arr.ndim == 1 and arr.size > 0:
                        collected["run_times_list"].append(arr)

                if "seeds" in algo_data_in:
                    seeds = algo_data_in["seeds"]
                    if isinstance(seeds, (list, np.ndarray)):
                        collected["seeds_list"].extend(list(seeds))

    # ---------------------------------------------------------
    # Aggregate collected arrays
    # ---------------------------------------------------------

    final_aggregated_data = {}

    for instance_key, problem_data in combined_data.items():

        problem_name = problem_data["problem"]
        n_vars = problem_data["n_vars"]

        final_aggregated_data[instance_key] = {
            "problem": problem_name,
            "n_vars": n_vars,
            "runs": 0,
            "results": {},
        }

        for algo, collected_data in problem_data["results"].items():

            if not collected_data["data_list"]:
                print(
                    f"Warning: No data found to aggregate for '{algo}' "
                    f"in problem '{problem_name}', dimension {n_vars}. "
                    "Skipping."
                )
                continue

            valid_data_arrays = [
                arr
                for arr in collected_data["data_list"]
                if isinstance(arr, np.ndarray)
            ]

            if not valid_data_arrays:
                print(
                    f"Warning: No valid NumPy arrays found for "
                    f"'{algo}' in problem '{problem_name}', "
                    f"dimension {n_vars}. Skipping."
                )
                continue

            try:
                concatenated_data = np.concatenate(
                    valid_data_arrays,
                    axis=0,
                )

            except ValueError as exc:
                print(
                    f"Error concatenating data for '{algo}' in "
                    f"problem '{problem_name}', dimension {n_vars}. "
                    f"Skipping. Error: {exc}"
                )

                for index, arr in enumerate(
                    collected_data["data_list"]
                ):
                    print(
                        f"  Array {index}: "
                        f"type={type(arr)}, "
                        f"shape={getattr(arr, 'shape', 'N/A')}"
                    )

                continue

            # Exact number of runs for this algorithm and instance.
            runs_count = concatenated_data.shape[0]

            # Store instance-level run count when algorithms agree.
            if final_aggregated_data[instance_key]["runs"] == 0:
                final_aggregated_data[instance_key]["runs"] = runs_count

            elif (
                final_aggregated_data[instance_key]["runs"]
                != runs_count
            ):
                print(
                    f"Warning: Different run counts for algorithms in "
                    f"problem '{problem_name}', dimension {n_vars}: "
                    f"{final_aggregated_data[instance_key]['runs']} "
                    f"vs {runs_count} for '{algo}'."
                )

            # -------------------------------------------------
            # Calculate final-run fitness statistics
            # -------------------------------------------------

            if concatenated_data.ndim == 2:
                final_run_fitness = concatenated_data[:, -1]
            else:
                final_run_fitness = concatenated_data

            valid_final_fitness = final_run_fitness[
                np.isfinite(final_run_fitness)
            ]

            if valid_final_fitness.size > 0:
                final_avg_fitness = np.mean(
                    valid_final_fitness
                )

                final_std_dev = np.std(
                    valid_final_fitness
                )

            else:
                print(
                    f"Warning: No valid fitness values found for "
                    f"'{algo}' in problem '{problem_name}', "
                    f"dimension {n_vars}."
                )

                final_avg_fitness = float("inf")
                final_std_dev = float("nan")

            # -------------------------------------------------
            # Calculate average runtime
            # -------------------------------------------------

            valid_avg_times = [
                value
                for value in collected_data["avg_time_list"]
                if isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
                and np.isfinite(value)
            ]

            final_avg_time = (
                float(np.mean(valid_avg_times))
                if valid_avg_times
                else 0.0
            )

            # -------------------------------------------------
            # Store aggregated algorithm result
            # -------------------------------------------------

            final_aggregated_data[
                instance_key
            ]["results"][algo] = {
                "data": concatenated_data,
                "runs": runs_count,
                "avg_fitness": final_avg_fitness,
                "std_dev": final_std_dev,
                "avg_time": final_avg_time,
                "final_fitness": (
                    np.concatenate(collected_data["final_fitness_list"])
                    if collected_data["final_fitness_list"]
                    else (concatenated_data[:, -1] if concatenated_data.ndim == 2 else concatenated_data)
                ),
                "run_times": (
                    np.concatenate(collected_data["run_times_list"])
                    if collected_data["run_times_list"]
                    else np.zeros(runs_count)
                ),
                "seeds": collected_data["seeds_list"],
            }

    print(
        "Data combined. "
        f"Aggregated benchmark instances: "
        f"{len(final_aggregated_data)}"
    )

    return final_aggregated_data