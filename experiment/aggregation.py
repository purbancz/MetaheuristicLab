import numpy as np


def combine_data(data_list):
    """
    Combine experiment result data while preserving problem dimensionality.

    Results are aggregated only when both problem name and dimensionality
    match. Therefore, e.g. ("Rastrigin", 100) and ("Rastrigin", 500)
    are treated as separate experimental instances.

    Returns
    -------
    tuple
        (aggregated_data, detected_number_of_runs)

        aggregated_data is keyed by:
            (problem_name, n_vars)
    """
    combined_data = {}
    total_runs_calculated = 0
    runs_set = False

    valid_data_list = [data for data in data_list if data is not None]

    if not valid_data_list:
        print("Warning: No valid data loaded from pickle files.")
        return {}, 0

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

        temp_runs_set_for_source = False

        if not runs_set:
            for p_data_check in problems_in_source:

                if (
                    p_data_check
                    and "results" in p_data_check
                    and p_data_check["results"]
                ):
                    first_algo_name = next(iter(p_data_check["results"]))
                    first_algo_data = p_data_check["results"][first_algo_name]

                    if (
                        "data" in first_algo_data
                        and isinstance(first_algo_data["data"], np.ndarray)
                    ):
                        current_source_runs = first_algo_data["data"].shape[0]
                        total_runs_calculated += current_source_runs
                        temp_runs_set_for_source = True
                        runs_set = True
                        break

            if not temp_runs_set_for_source:
                print(
                    "Warning: Could not determine number of runs "
                    "from data source."
                )

        for problem_data in problems_in_source:

            if (
                not isinstance(problem_data, dict)
                or "problem" not in problem_data
                or "results" not in problem_data
            ):
                print(
                    f"Warning: Skipping invalid problem data entry: "
                    f"{problem_data}"
                )
                continue

            problem_name = problem_data["problem"]
            n_vars = problem_data.get("n_vars", -1)

            # Normalize numpy integer types if they occur in old result files.
            if isinstance(n_vars, np.integer):
                n_vars = int(n_vars)

            results = problem_data["results"]

            # Dimension is part of the experimental-instance identity.
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
                    }

                collected = combined_data[instance_key]["results"][algo]

                if isinstance(algo_data_in["data"], np.ndarray):
                    collected["data_list"].append(algo_data_in["data"])

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

    # ---------------------------------------------------------
    # Aggregate collected arrays
    # ---------------------------------------------------------

    final_aggregated_data = {}
    actual_total_runs = 0

    for instance_key, problem_data in combined_data.items():

        problem_name = problem_data["problem"]
        n_vars = problem_data["n_vars"]

        final_aggregated_data[instance_key] = {
            "problem": problem_name,
            "n_vars": n_vars,
            "results": {},
        }

        first_algo_runs_set = False

        for algo, collected_data in problem_data["results"].items():

            if not collected_data["data_list"]:
                print(
                    f"Warning: No data found to aggregate for '{algo}' "
                    f"in problem '{problem_name}', dimension {n_vars}. "
                    "Skipping."
                )
                continue

            try:
                valid_data_arrays = [
                    arr
                    for arr in collected_data["data_list"]
                    if isinstance(arr, np.ndarray)
                ]

                if not valid_data_arrays:
                    print(
                        f"Warning: No valid numpy arrays found for "
                        f"'{algo}' in problem '{problem_name}', "
                        f"dimension {n_vars}. Skipping."
                    )
                    continue

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

            if concatenated_data.ndim == 2:
                final_run_fitness = concatenated_data[:, -1]
            else:
                final_run_fitness = concatenated_data

            valid_final_fitness = final_run_fitness[
                np.isfinite(final_run_fitness)
            ]

            if valid_final_fitness.size > 0:
                final_avg_fitness = np.mean(valid_final_fitness)
                final_std_dev = np.std(valid_final_fitness)

            else:
                print(
                    f"Warning: No valid fitness values found for "
                    f"'{algo}' in problem '{problem_name}', "
                    f"dimension {n_vars}."
                )

                final_avg_fitness = float("inf")
                final_std_dev = float("nan")

            valid_avg_times = [
                value
                for value in collected_data["avg_time_list"]
                if isinstance(value, (int, float))
                and np.isfinite(value)
            ]

            final_avg_time = (
                np.mean(valid_avg_times)
                if valid_avg_times
                else 0.0
            )

            final_aggregated_data[instance_key]["results"][algo] = {
                "data": concatenated_data,
                "avg_fitness": final_avg_fitness,
                "std_dev": final_std_dev,
                "avg_time": final_avg_time,
            }

            if not first_algo_runs_set:
                actual_total_runs = concatenated_data.shape[0]
                first_algo_runs_set = True

    if actual_total_runs == 0 and total_runs_calculated != 0:
        print(
            f"Warning: Calculated total runs "
            f"({total_runs_calculated}) but aggregated data has "
            f"0 runs. Using calculated value."
        )
        actual_total_runs = total_runs_calculated

    elif actual_total_runs == 0:
        print(
            "Warning: Could not determine total runs "
            "from aggregated data."
        )

    print(
        f"Data combined. Runs per aggregated instance detected: "
        f"{actual_total_runs}"
    )

    return final_aggregated_data, actual_total_runs