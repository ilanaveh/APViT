"""
This code is for visualizing the original apvit models, using masked test-images.
Not to be confused with the apvit-training with masks I did later on.
"""


from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_mask_results_to_df(json_path, n_class, include_total_acc=False):
    """
    Convert nested results JSON into a tidy DataFrame and compute binomial SE.

    Expected JSON structure:
        model -> mask -> test_condition -> emotion -> accuracy

    Assumptions:
    - emotion accuracies are stored as proportions in [0, 1]
    - total_acc is stored as percent in [0, 100]
    """

    json_path = Path(json_path)

    with open(json_path, "r") as f:
        results = json.load(f)

    rows = []
    total_n = sum(n_class.values())

    for model_name, model_data in results.items():
        for mask_name, mask_data in model_data.items():
            for test_condition, acc_data in mask_data.items():
                for emotion, acc in acc_data.items():

                    if emotion == "total_acc" and not include_total_acc:
                        continue

                    if emotion == "total_acc":
                        p = acc / 100.0
                        n = total_n
                    else:
                        p = acc
                        n = n_class[emotion]

                    se = np.sqrt((p * (1 - p)) / n)

                    rows.append({
                        "model": model_name,
                        "mask": mask_name,
                        "test_condition": test_condition,
                        "emotion": emotion,
                        "accuracy_percent": p * 100,
                        "se_percent": se * 100,
                        "n": n,
                    })

    return pd.DataFrame(rows)


def add_mean_across_emotions(
    df,
    n_class,
    mean_mode="balanced",
    source_emotions=None,
):
    """
    Add mean-across-emotions rows.

    mean_mode:
    - "balanced": simple average of class accuracies
    - "native": weighted average of class accuracies using n_class

    Error bars:
    - balanced: propagated SE of the mean of independent class proportions
    - native: binomial SE of the pooled weighted accuracy
    """

    if mean_mode not in {"balanced", "native"}:
        raise ValueError("mean_mode must be either 'balanced' or 'native'")

    if source_emotions is None:
        source_emotions = list(n_class.keys())

    class_df = df[df["emotion"].isin(source_emotions)].copy()

    rows = []

    group_cols = ["model", "mask", "test_condition"]

    for group_values, group_df in class_df.groupby(group_cols):
        model_name, mask_name, test_condition = group_values

        group_df = group_df.set_index("emotion").loc[source_emotions].reset_index()

        p = group_df["accuracy_percent"].to_numpy() / 100.0
        n = np.array([n_class[e] for e in group_df["emotion"]], dtype=float)

        if mean_mode == "balanced":
            mean_p = np.mean(p)

            # Propagated SE of mean:
            # SE(mean) = sqrt(sum(SE_i^2)) / K
            class_se = np.sqrt((p * (1 - p)) / n)
            mean_se = np.sqrt(np.sum(class_se ** 2)) / len(p)

            n_used = np.sum(n)

        elif mean_mode == "native":
            mean_p = np.sum(p * n) / np.sum(n)

            # Pooled binomial SE
            mean_se = np.sqrt((mean_p * (1 - mean_p)) / np.sum(n))

            n_used = np.sum(n)

        rows.append({
            "model": model_name,
            "mask": mask_name,
            "test_condition": test_condition,
            "emotion": f"mean_{mean_mode}",
            "accuracy_percent": mean_p * 100,
            "se_percent": mean_se * 100,
            "n": n_used,
        })

    return pd.DataFrame(rows)


def _plot_grouped_bars(
    df,
    fixed_col,
    fixed_value,
    x_col,
    group_col,
    x_order,
    group_order,
    title,
    output_path=None,
    figsize=(12, 5),
    ylim=(0, 100),
):
    """
    Internal helper for grouped bar plots with error bars.
    """

    plot_df = df[df[fixed_col] == fixed_value].copy()

    x = np.arange(len(x_order))
    n_groups = len(group_order)
    bar_width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=figsize)

    for i, group_value in enumerate(group_order):
        values = []
        errors = []

        for x_value in x_order:
            match = plot_df[
                (plot_df[x_col] == x_value) &
                (plot_df[group_col] == group_value)
            ]

            if len(match) == 0:
                values.append(np.nan)
                errors.append(np.nan)
            else:
                values.append(match["accuracy_percent"].iloc[0])
                errors.append(match["se_percent"].iloc[0])

        offset = (i - (n_groups - 1) / 2) * bar_width

        ax.bar(
            x + offset,
            values,
            width=bar_width,
            yerr=errors,
            capsize=4,
            label=str(group_value),
        )

    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel(x_col.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(x_order, rotation=45, ha="right")
    ax.set_ylim(*ylim)
    ax.legend(title=group_col.capitalize())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
def _safe_name(x):
    return str(x).replace("/", "_").replace(" ", "_")


def _get_order_for_col(col, df, model_order, mask_order, emotion_order):
    if col == "model":
        return model_order
    elif col == "mask":
        return mask_order
    elif col == "emotion":
        return emotion_order
    else:
        raise ValueError("Column must be one of: 'model', 'mask', 'emotion'")


def plot_mask_breakdown_bars(
    json_path,
    n_class,
    output_dir=None,
    include_total_acc=False,
    split_by="model",
    mean_mode=None,
    x_axis=None,
    bars=None,
    model_order=None,
    mask_order=("eyes", "mouth", "none"),
    emotion_order=None,
    figsize=(12, 5),
):
    """
    Plot grouped bar plots with optional mean across emotions and flexible x-axis/bar assignment.

    Parameters
    ----------
    split_by : {"model", "mask", "emotion"}
        Which variable gets one separate figure per value.

    x_axis : {"model", "mask", "emotion"} or None
        Which variable appears on the x-axis.
        Must be different from split_by and bars.

    bars : {"model", "mask", "emotion"} or None
        Which variable defines the grouped bars.
        Must be different from split_by and x_axis.

    mean_mode : {None, "balanced", "native", "both"}
        - None: plot each emotion separately.
        - "balanced": unweighted average across emotion accuracies.
        - "native": weighted average across emotion accuracies using n_class.
        - "both": plot both balanced and native means.

    Examples
    --------
    Old behavior for split_by="model":
        split_by="model", x_axis="emotion", bars="mask"

    Old behavior for split_by="emotion":
        split_by="emotion", x_axis="mask", bars="model"

    New requested behavior:
        split_by="emotion", x_axis="model", bars="mask"
    """

    df = load_mask_results_to_df(
        json_path=json_path,
        n_class=n_class,
        include_total_acc=include_total_acc,
    )

    if emotion_order is None:
        emotion_order = [
            "Anger",
            "Disgust",
            "Fear",
            "Happiness",
            "Neutral",
            "Sadness",
            "Surprise",
        ]

    if mean_mode is not None:
        if mean_mode == "both":
            mean_modes = ["balanced", "native"]
        elif mean_mode in {"balanced", "native"}:
            mean_modes = [mean_mode]
        else:
            raise ValueError("mean_mode must be one of: None, 'balanced', 'native', 'both'")

        mean_dfs = []

        for mode in mean_modes:
            mean_dfs.append(
                add_mean_across_emotions(
                    df=df,
                    n_class=n_class,
                    mean_mode=mode,
                    source_emotions=emotion_order,
                )
            )

        df = pd.concat(mean_dfs, ignore_index=True)
        emotion_order = [f"mean_{mode}" for mode in mean_modes]

    else:
        if include_total_acc:
            emotion_order = emotion_order + ["total_acc"]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if model_order is None:
        model_order = list(df["model"].unique())

    split_by = split_by.lower()

    if split_by not in {"model", "mask", "emotion"}:
        raise ValueError("split_by must be one of: 'model', 'mask', 'emotion'")

    # Default behavior, matching the previous function
    if x_axis is None or bars is None:
        default_layouts = {
            "model":   {"x_axis": "emotion", "bars": "mask"},
            "mask":    {"x_axis": "emotion", "bars": "model"},
            "emotion": {"x_axis": "mask",    "bars": "model"},
        }

        if x_axis is None:
            x_axis = default_layouts[split_by]["x_axis"]

        if bars is None:
            bars = default_layouts[split_by]["bars"]

    x_axis = x_axis.lower()
    bars = bars.lower()

    valid_cols = {"model", "mask", "emotion"}

    if x_axis not in valid_cols:
        raise ValueError("x_axis must be one of: 'model', 'mask', 'emotion'")

    if bars not in valid_cols:
        raise ValueError("bars must be one of: 'model', 'mask', 'emotion'")

    if len({split_by, x_axis, bars}) != 3:
        raise ValueError(
            "split_by, x_axis, and bars must be three different variables. "
            "For example: split_by='emotion', x_axis='model', bars='mask'."
        )

    split_order = _get_order_for_col(
        split_by,
        df=df,
        model_order=model_order,
        mask_order=mask_order,
        emotion_order=emotion_order,
    )

    x_order = _get_order_for_col(
        x_axis,
        df=df,
        model_order=model_order,
        mask_order=mask_order,
        emotion_order=emotion_order,
    )

    bar_order = _get_order_for_col(
        bars,
        df=df,
        model_order=model_order,
        mask_order=mask_order,
        emotion_order=emotion_order,
    )

    for test_condition in df["test_condition"].unique():
        df_test = df[df["test_condition"] == test_condition].copy()

        for split_value in split_order:
            title = f"{split_by.capitalize()}: {split_value} | {test_condition}"

            output_path = None
            if output_dir is not None:
                mean_suffix = "noMean" if mean_mode is None else f"mean_{mean_mode}"
                safe_test = _safe_name(test_condition)
                safe_split = _safe_name(split_value)

                output_path = (
                    output_dir /
                    f"splitBy_{split_by}_{safe_split}_x_{x_axis}_bars_{bars}_{mean_suffix}_{safe_test}.png"
                )

            _plot_grouped_bars(
                df=df_test,
                fixed_col=split_by,
                fixed_value=split_value,
                x_col=x_axis,
                group_col=bars,
                x_order=x_order,
                group_order=bar_order,
                title=title,
                output_path=output_path,
                figsize=figsize,
            )

    return df


n_class = {
    "Anger": 162,
    "Disgust": 160,
    "Fear": 74,
    "Happiness": 1185,
    "Neutral": 680,
    "Sadness": 478,
    "Surprise": 329,
}
df_mean_balanced = plot_mask_breakdown_bars(
    json_path="results_with_mask_breakdown.json",
    n_class=n_class,
    output_dir="mask_breakdown_plots",
    split_by="emotion",
    mean_mode="native",
    x_axis="mask",
    bars="model",
)

plt.show(block=True)
