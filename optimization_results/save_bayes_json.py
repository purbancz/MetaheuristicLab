#!/usr/bin/env python3
import json
import pandas as pd
import os


def json_to_csv_and_params(json_filepath):
    records = []
    # Wczytanie danych z pliku JSON (jeden rekord na linię)
    with open(json_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Błąd dekodowania JSON: {e}")
                continue
            # Pobranie słownika parametrów
            params = record.get('params', {})
            # Dołączenie wartości average_result (run_number pomijamy)
            params['average_result'] = record.get('average_result')
            records.append(params)

    # Utworzenie DataFrame i sortowanie wg average_result rosnąco
    df = pd.DataFrame(records)
    df.sort_values(by='average_result', inplace=True)

    # Zapis do CSV o tej samej nazwie (zmiana rozszerzenia na .csv)
    base, _ = os.path.splitext(json_filepath)
    output_csv = base + '.csv'
    df.to_csv(output_csv, index=False)
    print(f"CSV zapisano do pliku: {output_csv}")

    # Dla 10 pierwszych rekordów obliczamy przedziały dla poszczególnych parametrów
    df_top10 = df.head(10)
    # Przyjmujemy, że kolumny parametrów to wszystkie poza "average_result"
    param_columns = [col for col in df_top10.columns if col != 'average_result']

    result_lines = []
    # Algorytm pobieramy z nazwy pliku: pierwsze słowo przed znakiem podkreślenia
    filename = os.path.basename(json_filepath)
    algo_name = filename.split('_')[0]

    result_lines.append(f"'{algo_name}': [")
    for param in param_columns:
        values = df_top10[param]
        p_min = values.min()
        p_max = values.max()
        p_range = p_max - p_min
        # Rozszerzamy przedział o 10% z każdej strony
        new_lower = p_min - 0.1 * p_range
        new_upper = p_max + 0.1 * p_range
        # Formatowanie wartości z 5 miejscami po przecinku
        result_lines.append(f'    Real("{param}", {new_lower:.5f}, {new_upper:.5f}),')
    result_lines.append("],")

    result_text = "\n".join(result_lines)

    # Wypisanie wyniku
    print("\nZakresy parametrów dla 10 najlepszych wyników:")
    print(result_text)

    # Zapis do pliku TXT o tej samej nazwie (zmiana rozszerzenia na .txt)
    output_txt = base + '.txt'
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(result_text)
    print(f"Wyniki zakresów zapisano do pliku: {output_txt}")


if __name__ == '__main__':
    # Podaj tutaj nazwę Twojego pliku JSON
    json_file = "SPPPSO_results.json"  # przykładowo: "PSO_podloga.json"
    json_to_csv_and_params(json_file)
