import re
import traceback
from pathlib import Path
import pandas as pd
import ast
import os

from optimization.irace_tune_universal import normalize_fraction_sum, repair_max_param_constraints_random


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
    print(f"\n>>> Processing file: {filepath} <<<")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}"); return {}

    # Split by algorithm start marker
    segments = re.split(r'Optimizing parameters for\s+([A-Za-z0-9_]+(?:_WithRandom|_DefaultStd|RestarterPSO)?)\s*(?:\.\.\.)?\s*\n', content)

    dataframes = {}
    total_records_processed_file = 0
    total_records_saved_file = 0

    for i in range(1, len(segments), 2):
        alg_name = segments[i].strip()
        print(f"\n--- Processing Segment for Algorithm: '{alg_name}' ---")
        if i + 1 >= len(segments): continue
        segment_text = segments[i + 1]

        # --- More Robust Regex ---
        # 1. Find "Evaluated config...OrderedDict("
        # 2. Capture the dictionary content "{...}" (non-greedy)
        # 3. Find "-> Final Avg Cost:" potentially much later
        # 4. Capture the cost value (allowing inf/nan)
        # Use MULTILINE flag as well as DOTALL
        pattern_flexible = r"Evaluated config.*?OrderedDict\((.*?})\).*?-> Final Avg Cost:\s*([0-9.infNaNINF]+)"
        # Explanation:
        # Evaluated config.*?OrderedDict\(   : Find the start
        # (\{.*?\})                         : Capture Group 1: The dict content {...} (non-greedy)
        # \).*?                            : Match the closing paren and anything non-greedily until cost
        # -> Final Avg Cost:\s*            : Match the cost marker
        # ([0-9.infNaNINF]+)               : Capture Group 2: Digits, dot, inf, nan (case insensitive via flag)

        try:
            matches = re.findall(pattern_flexible, segment_text, re.DOTALL | re.IGNORECASE)
        except Exception as regex_err:
             print(f"Regex error for algorithm '{alg_name}': {regex_err}")
             matches = []

        print(f"Found {len(matches)} potential records for '{alg_name}'.")

        records = []
        processed_count_alg = 0
        error_count_alg = 0

        for match_tuple in matches:
            processed_count_alg += 1
            if len(match_tuple) != 2: # Ensure regex captured two groups
                print(f"Warning: Regex match returned unexpected number of groups ({len(match_tuple)}) for {alg_name}. Skipping.")
                error_count_alg += 1
                continue

            config_str, avg_obj_str_match = match_tuple

            try:
                # --- Config Dictionary Parsing ---
                # Add extra cleaning: ensure matching outer braces {}
                config_str_cleaned = config_str.strip()
                if not (config_str_cleaned.startswith('{') and config_str_cleaned.endswith('}')):
                     print(f"Warning: Captured config string doesn't look like dict literal for {alg_name}. Skipping. String: '{config_str_cleaned[:50]}...'")
                     error_count_alg += 1
                     continue
                try:
                    config_dict_raw = ast.literal_eval(config_str_cleaned)
                except (SyntaxError, ValueError) as eval_err:
                     print(f"!!! Error (ast.literal_eval) processing config for {alg_name}: {eval_err}")
                     print(f"    Problematic Config String: {config_str_cleaned}")
                     error_count_alg += 1
                     continue # Skip this record

                if not isinstance(config_dict_raw, dict):
                     print(f"Warning: ast.literal_eval did not return a dict for {alg_name}. Got type {type(config_dict_raw)}. Skipping record.")
                     error_count_alg += 1
                     continue
                config_dict = dict(config_dict_raw) # Ensure regular dict

                # --- Cost Conversion ---
                cost_str_cleaned = avg_obj_str_match.strip().lower()
                try:
                    if cost_str_cleaned == 'inf': avg_obj = float('inf')
                    elif cost_str_cleaned == 'nan': avg_obj = float('nan')
                    else:
                        # remove all ASCII letters
                        numeric_part = re.sub(r'[A-Za-z]', '', cost_str_cleaned)
                        # now match only digits, decimal point, sign and exponent
                        if re.fullmatch(r'[+-]?\d+(\.\d*)?([eE][+-]?\d+)?', numeric_part):
                            avg_obj = float(numeric_part)
                        else:
                         print(f"Warning: Could not convert cleaned cost string '{cost_str_cleaned}' to float for {alg_name}. Skipping record.")
                         error_count_alg += 1
                         continue
                except ValueError:
                     print(f"Warning: ValueError converting cost string '{cost_str_cleaned}' to float for {alg_name}. Skipping record.")
                     error_count_alg += 1
                     continue

                # --- Repair and Normalize ---
                repaired_config_dict = repair_max_param_constraints_random(config_dict) # Use the desired repair func
                # Conditional normalization...
                normalized_config_dict = repaired_config_dict # Default
                # ... (Add back your specific normalization logic based on alg_name if needed) ...
                # Example (needs full logic):
                # if 'PartialDisjoint' in alg_name: ...
                # elif 'FullDisjoint' in alg_name: ...

            except Exception as e:
                print(f"!!! Unexpected Error processing record #{processed_count_alg} for {alg_name}: {e}")
                print(f"    Config String: {config_str}")
                print(f"    Avg Obj String: {avg_obj_str_match}")
                traceback.print_exc()
                error_count_alg += 1
                continue

            # Append successfully processed record
            normalized_config_dict["average_objective"] = avg_obj
            records.append(normalized_config_dict)
            # --- DEBUG ---
            # if processed_count_alg <= 5: # Print first few successful records
            #     print(f"  Successfully processed record #{processed_count_alg} for {alg_name}. Cost: {avg_obj}")
            # --- END DEBUG ---


        print(f"Processed {processed_count_alg} potential records for '{alg_name}'. {len(records)} added successfully ({error_count_alg} errors).")

        if records:
             # --- Create DataFrame and Save ---
            try:
                df = pd.DataFrame(records)
                # ... (rest of saving logic - check for empty after drop_duplicates etc.) ...
                if df.empty: print(f"DataFrame empty after creation for '{alg_name}'"); continue
                df.drop_duplicates(inplace=True);
                if df.empty: print(f"DataFrame empty after drop_duplicates for '{alg_name}'"); continue
                df.sort_values(by="average_objective", ascending=True, inplace=True)
                # output_dir = Path(filepath).parent / "parsed_results"; output_dir.mkdir(parents=True, exist_ok=True)
                output_csv = f"{alg_name}_parsed_results.csv"; df.to_csv(output_csv, index=False)
                print(f"Results for '{alg_name}' saved in {output_csv}")
                total_records_saved_file += len(df)
                # ... (TXT generation) ...
            except Exception as save_err: print(f"Error saving files for '{alg_name}': {save_err}")
        else:
            print(f"No records successfully processed/saved for '{alg_name}'")

    print(f"\nFinished processing file {filepath}. Total records saved: {total_records_saved_file}")
    return dataframes # Might be empty if errors occurred


if __name__ == "__main__":
    base_dir = Path(".")
    out_files = list(base_dir.glob("*.out"))

    for out_file in out_files:
        process_file(out_file)
