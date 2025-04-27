import re
from pathlib import Path
import pandas as pd
import ast
import os

# Helper method for normalizing parameters
def normalize_fractions(config, alg_name):
    normalized_config = config.copy()

    if alg_name == 'HybridPartialDisjointPSO':
        cognitive_group = ["rejector_fraction", "defeatist_fraction", "escapist_fraction"]
        social_group = ["rebel_fraction", "contrarian_fraction", "eschewer_fraction"]

        cog_sum = sum(config.get(param, 0) for param in cognitive_group)
        if cog_sum > 1.0:
            for param in cognitive_group:
                normalized_config[param] = config.get(param, 0) / cog_sum

        soc_sum = sum(config.get(param, 0) for param in social_group)
        if soc_sum > 1.0:
            for param in social_group:
                normalized_config[param] = config.get(param, 0) / soc_sum

    elif alg_name == 'HybridFullDisjointPSO':
        all_group = [
            "rejector_fraction", "defeatist_fraction", "escapist_fraction",
            "rebel_fraction", "contrarian_fraction", "eschewer_fraction"
        ]
        total_sum = sum(config.get(param, 0) for param in all_group)
        if total_sum > 1.0:
            for param in all_group:
                normalized_config[param] = config.get(param, 0) / total_sum

    return normalized_config

# Helper method for repairing constraints
def repair_max_param_constraints(config: dict, verbose: bool = True) -> dict:
    repaired_config = config.copy()
    epsilon = 1e-9

    constraints_to_check = [
        ("c1", "max_c1"),
        ("c2", "max_c2"),
        ("min_inertia", "base_inertia"),
        ("base_inertia", "max_inertia"),
        ("min_inertia", "max_inertia"),
        ("eschewer_fraction", "max_eschewer_fraction"),
        ("escapist_fraction", "max_escapist_fraction"),
        ("contrarian_fraction", "max_contrarian_fraction"),
        ("defeatist_fraction", "max_defeatist_fraction"),
        ("rebel_fraction", "max_rebel_fraction"),
        ("rejector_fraction", "max_rejector_fraction"),
    ]

    for param, max_param in constraints_to_check:
        param_val = repaired_config.get(param, None)
        max_param_val = repaired_config.get(max_param, None)

        if param_val is not None and max_param_val is not None:
            if max_param_val + epsilon < param_val:
                repaired_config[max_param] = param_val
                if verbose:
                    print(f"Repairing constraint: {max_param} ({max_param_val:.4f}) < {param} ({param_val:.4f}). Setting {max_param} = {param_val:.4f}")

    return repaired_config


def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = re.split(r'Optimizing parameters for\s+(.+?)\s*\n', content)

    dataframes = {}
    for i in range(1, len(segments), 2):
        alg_name = segments[i].strip()
        if alg_name.endswith("..."):
            alg_name = alg_name[:-3].strip()
        segment_text = segments[i + 1]

        pattern_old = r"Evaluated config: OrderedDict\((\{.*?\})\) with average objective: ([0-9.]+)"
        pattern_new = r"Evaluated config: OrderedDict\((\{.*?\})\).*?-> Final Avg Cost: ([0-9.]+)"

        matches_old = re.findall(pattern_old, segment_text)
        if matches_old:
            matches = matches_old
        else:
            matches_new = re.findall(pattern_new, segment_text, re.DOTALL)
            matches = matches_new

        records = []
        for config_str, avg_obj in matches:
            try:
                config_dict = ast.literal_eval(config_str)
                repaired_config_dict = repair_max_param_constraints(config_dict, verbose=False)
                normalized_config_dict = normalize_fractions(repaired_config_dict, alg_name)
            except Exception as e:
                print(f"Error processing configuration: {config_str}. Error: {e}")
                continue
            normalized_config_dict["average_objective"] = float(avg_obj)
            records.append(normalized_config_dict)

        if records:
            df = pd.DataFrame(records)
            df.drop_duplicates(inplace=True)
            df.sort_values(by="average_objective", ascending=True, inplace=True)
            dataframes[alg_name] = df
            output_csv = f"{alg_name}_results.csv"
            df.to_csv(output_csv, index=False)
            print(f"Results for '{alg_name}' saved in {output_csv}")

            df_top10 = df.head(10)
            param_columns = [col for col in df_top10.columns if col != "average_objective"]
            txt_lines = []
            txt_lines.append(f"'{alg_name}': [")

            for param in param_columns:
                if df_top10[param].dtype == bool:
                    txt_lines.append(f'    Bool("{param}"),')
                else:
                    values = df_top10[param]
                    p_min, p_max = values.min(), values.max()
                    p_range = p_max - p_min
                    new_lower = p_min - 0.1 * p_range
                    new_upper = p_max + 0.1 * p_range

                    if pd.api.types.is_integer_dtype(values):
                        new_lower = int(max(round(new_lower), values.min()))
                        new_upper = int(min(round(new_upper), values.max()))
                        txt_lines.append(f'    Integer("{param}", {new_lower}, {new_upper}),')
                    else:
                        txt_lines.append(f'    Real("{param}", {new_lower:.5f}, {new_upper:.5f}),')

            txt_lines.append("],")
            result_txt = "\n".join(txt_lines)

            print("\nParameter ranges for 10 best results:")
            print(result_txt)

            output_txt = f"{alg_name}_results.txt"
            with open(output_txt, 'w', encoding='utf-8') as txt_file:
                txt_file.write(result_txt)
            print(f"Ranges of best results saved in {output_txt}")
        else:
            print(f"No records for '{alg_name}'")

    return dataframes


if __name__ == "__main__":
    base_dir = Path(".")
    out_files = list(base_dir.glob("*.out"))

    for out_file in out_files:
        process_file(out_file)
