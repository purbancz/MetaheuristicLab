#!/usr/bin/env python3
import re
from pathlib import Path

import pandas as pd
import ast
import os


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

        pattern = r"Evaluated config: OrderedDict\((\{.*?\})\) with average objective: ([0-9.]+)"
        matches = re.findall(pattern, segment_text)

        records = []
        for config_str, avg_obj in matches:
            try:
                config_dict = ast.literal_eval(config_str)
            except Exception as e:
                print(f"Error during processing the configuration: {config_str}. Error: {e}")
                continue
            config_dict["average_objective"] = float(avg_obj)
            records.append(config_dict)

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
            txt_lines = [f"'{alg_name}': ["]

            for param in param_columns:
                if df_top10[param].dtype == bool:
                    # For boolean parameters, just declare as Bool without range
                    txt_lines.append(f'    Bool("{param}"),')
                else:
                    values = df_top10[param]
                    p_min = values.min()
                    p_max = values.max()
                    p_range = p_max - p_min
                    new_lower = p_min - 0.1 * p_range
                    new_upper = p_max + 0.1 * p_range

                    # Handle integer values separately
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
