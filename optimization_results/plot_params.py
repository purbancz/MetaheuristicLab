import os
import shutil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import matplotlib.colors as mcolors
from pandas.plotting import parallel_coordinates
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import numpy as np


def save_fig(given_title, plot_dir="plots"):
    """Saves the plot in a specified directory, creating the directory if needed."""
    os.makedirs(plot_dir, exist_ok=True)
    filename = "".join(c if c.isalnum() or c in "_-" else "_" for c in given_title.replace(" ", "_"))  # sanitize
    filepath = os.path.join(plot_dir, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {filepath}")


def auto_normalize(df, exclude_columns):
    """Automatically determines min-max ranges by rounding down min and rounding up max."""
    min_max_values = {}
    for col in df.columns:
        if col not in exclude_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            rounded_min = np.floor(min_val * 100) / 100  # Floor to 2 decimal places
            rounded_max = np.ceil(max_val * 100) / 100  # Ceil to 2 decimal places
            min_max_values[col] = (rounded_min, rounded_max)
    return min_max_values


def generate_parallel_coordinates_plot(df, target_col, algo_name, plots_folder):
    """Generates a Parallel Coordinates Plot with automatic min-max normalization."""

    # Extract algorithm name from filename
    # algo_name = Path(csv_file).stem.split("_")[0]

    # Exclude constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1 and col != target_col]
    df = df.drop(columns=constant_columns)

    # Automatically determine min-max values for all plotted columns
    custom_min_max = auto_normalize(df, exclude_columns=[target_col])

    # Normalize dataset using automatically determined min-max values
    df_custom_normalized = df.copy()
    for col, (custom_min, custom_max) in custom_min_max.items():
        df_custom_normalized[col] = (df[col] - custom_min) / (custom_max - custom_min)

    # Prepare dataframe for plotting
    df_custom_normalized["color"] = (df[target_col] - df[target_col].min()) / (
            df[target_col].max() - df[target_col].min()
    )
    df_sorted = df_custom_normalized.sort_values(by="color", ascending=False)  # Reverse sorting

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    parallel_coordinates(df_sorted.drop(columns=[target_col]), class_column="color", colormap=cm.viridis, alpha=0.7,
                         ax=ax)
    ax.get_legend().remove()

    # Set title
    title = f"Parallel coordinates plot for {algo_name} parameters"
    plt.title(title)
    plt.xlabel("Parameters")
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    # Remove the frame around the columns (keep only vertical lines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set dynamically determined min-max values as tick labels
    ax.set_yticklabels([])  # Remove default 0-1 scale
    for i, col in enumerate(custom_min_max.keys()):
        min_val, max_val = custom_min_max[col]
        ax.text(i, -0.01, f"{min_val:.2f}", ha='center', va='top', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))  # Adjust spacing
        ax.text(i, 1.01, f"{max_val:.2f}", ha='center', va='bottom', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))  # Adjust spacing

    # Add colorbar for objective function
    sm = plt.cm.ScalarMappable(cmap=cm.viridis_r,
                               norm=mcolors.Normalize(vmin=df[target_col].min(), vmax=df[target_col].max()))
    cbar = fig.colorbar(sm, ax=ax, aspect=20)
    cbar.set_label("Average objective")

    # Save & Show the plot
    save_fig(title, plots_folder)
    plt.show()


# ---------------------------
# 1. Correlation Heatmap
# ---------------------------

def generate_correlation_matrix_plot(df, algo_name, plots_folder):
    # plt.figure(figsize=(8, 6))

    constant_columns = [col for col in df.columns if df[col].nunique() <= 1 and col != target_col]
    df = df.drop(columns=constant_columns)

    n_cols = len(df.columns)
    fig_width = max(8, int(n_cols * 0.7))
    fig_height = max(6, int(n_cols * 0.5))

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    title = f"Correlation matrix for {algo_name} parameters"
    plt.title(title)
    save_fig(title, plots_folder)
    plt.show()


# ---------------------------
# 2. Scatter Plots (Each Parameter vs. Average Objective)
# ---------------------------

def generate_scatter_plots(df, target_col, algo_name, plots_folder):
    for col in df.columns:
        if col != target_col:  # Exclude target itself
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col], y=df[target_col])
            plt.xlabel(col)
            plt.ylabel("Average Objective")
            title = f"{col} vs. average objective in {algo_name}"
            plt.title(title)
            save_fig(title, plots_folder)
            plt.show()


def generate_feature_importance_plot(X, importances, algo_name, plots_folder):
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=importance_df, hue="Feature", palette="viridis", legend=False)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    title = f"Importance of {algo_name} parameters"
    plt.title(title)
    save_fig(title, plots_folder)
    plt.show()


def generate_partial_dependence_plots(rf, X, algo_name, plots_folder):
    for col in X.columns:
        x_values = np.linspace(X[col].min(), X[col].max(), 50)
        y_values = []

        for val in x_values:
            X_temp = X.copy()
            X_temp[col] = val
            y_pred = rf.predict(X_temp)
            y_values.append(np.mean(y_pred))

        plt.figure(figsize=(6, 4))
        plt.plot(x_values, y_values, marker="o")
        plt.xlabel(col)
        plt.ylabel("Predicted Average Objective")
        title = f"Partial dependence plot for {col} in {algo_name}"
        plt.title(title)
        plt.grid(True)
        save_fig(title, plots_folder)
        plt.show()


def generate_residual_plots(rf, X_test, y_test, algo_name, plots_folder):
    y_pred = rf.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    title = f"Residual plot (checking model fit) for {algo_name} parameters"
    plt.title(title)
    save_fig(title, plots_folder)
    plt.show()


def generate_rf_dependent_plots(df, target_col, algo_name, plots_folder):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    generate_feature_importance_plot(X, importances, algo_name, plots_folder)
    generate_partial_dependence_plots(rf, X, algo_name, plots_folder)
    generate_residual_plots(rf, X_test, y_test, algo_name, plots_folder)


if __name__ == "__main__":
    base_dir = Path(".")
    folder_suffix = "_irace_rough"
    csv_files = list(base_dir.glob("*.csv"))

    for csv_file in csv_files:
        algo_name = csv_file.stem.split("_")[0]

        target_folder = base_dir / f"{algo_name}{folder_suffix}"
        plots_folder = target_folder / "plots"

        os.makedirs(plots_folder, exist_ok=True)

        new_csv_path = target_folder / csv_file.name
        shutil.move(str(csv_file), str(new_csv_path))

        txt_file = base_dir / f"{csv_file.stem}.txt"
        if txt_file.exists():
            shutil.move(str(txt_file), str(target_folder / txt_file.name))

        print(f"Moved {csv_file.name} to {target_folder}/")
        if txt_file.exists():
            print(f"Moved {txt_file.name} to {target_folder}/")

        df = pd.read_csv(new_csv_path)
        target_col = "average_objective"

        generate_correlation_matrix_plot(df, algo_name, plots_folder)
        generate_scatter_plots(df, target_col, algo_name, plots_folder)
        generate_rf_dependent_plots(df, target_col, algo_name, plots_folder)
        generate_parallel_coordinates_plot(df, target_col, algo_name, plots_folder)
