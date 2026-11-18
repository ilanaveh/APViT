import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from scipy.stats import binomtest
# from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIG
# ============================================================

CLASS_NAMES = [
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
    "Angry",
    "Disgust",
    "Fear",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)

MIN_RELEVANCE_SUBSET_N = 5
N_BOOTSTRAP = 5000
RANDOM_SEED = 42

MODEL_A = 'RAF_blur8_pretrained0-8_freeze_fix_projs'
MODEL_B = 'RAF_blur8_freeze'
# ============================================================
# LABEL UTILITIES
# ============================================================

def labels_to_int(labels):
    """
    Convert string labels to integer class indices.

    If labels are already integers, they are returned as integers.
    """
    labels = np.asarray(labels)

    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(int)

    return np.array(
        [CLASS_TO_IDX[x] for x in labels],
        dtype=int
    )


def get_class_counts(y_true):
    return np.array([
        np.sum(y_true == i)
        for i in range(N_CLASSES)
    ])


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_inputs(
    y_true,
    y_pred_A,
    y_pred_B,
    relevance_A=None,
    relevance_B=None,
):
    assert len(y_true) == len(y_pred_A) == len(y_pred_B)

    if relevance_A is not None:
        assert relevance_A.shape[0] == len(y_true)

    if relevance_B is not None:
        assert relevance_B.shape[0] == len(y_true)

    if relevance_A is not None and relevance_B is not None:
        assert relevance_A.shape == relevance_B.shape


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def plot_class_counts(y_true):
    counts = get_class_counts(y_true)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.bar(CLASS_NAMES, counts)

    ax.set_ylabel("Number of test images")
    ax.set_title("Test-set class distribution")
    ax.tick_params(axis="x", rotation=45)

    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(n),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


# ============================================================
# CONFUSION MATRICES
# ============================================================

def compute_confusion_matrices(y_true, y_pred):
    """
    Returns:
        counts:
            Raw confusion matrix.

        normalized:
            Row-normalized confusion matrix:
            P(predicted=j | true=i)
    """
    counts = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(N_CLASSES),
    )

    row_sums = counts.sum(axis=1, keepdims=True)

    normalized = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums != 0,
    )

    return counts, normalized


def plot_confusion_matrix(
    cm,
    title,
    normalized=True,
    cmap="Blues",
    vmin=None,
    vmax=None,
):
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        cm,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(title)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):

            if normalized:
                text = f"{100 * cm[i, j]:.1f}%"
            else:
                text = str(int(cm[i, j]))

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_confusion_difference_pp(delta_cm):
    """
    Plot:
        100 * (Model B normalized CM - Model A normalized CM)

    Positive diagonal:
        Model B improved.

    Negative off-diagonal:
        Model B reduced that confusion.
    """
    delta_pp = 100 * delta_cm

    max_abs = np.max(np.abs(delta_pp))

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        delta_pp,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(
        "Change in confusion rate: Model B − Model A"
    )

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j,
                i,
                f"{delta_pp[i, j]:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage points")

    plt.tight_layout()
    plt.show()


# ============================================================
# BOOTSTRAP CONFUSION-MATRIX DIFFERENCES
# ============================================================

def bootstrap_confusion_difference(
    y_true,
    y_pred_A,
    y_pred_B,
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_SEED,
):
    """
    Paired bootstrap, performed separately within each true class.

    This preserves:
    - the pairing between Model A and Model B predictions
    - the class sample sizes
    """
    rng = np.random.default_rng(seed)

    boot_deltas = np.zeros(
        (n_boot, N_CLASSES, N_CLASSES),
        dtype=np.float32,
    )

    indices_by_class = [
        np.where(y_true == c)[0]
        for c in range(N_CLASSES)
    ]

    for b in range(n_boot):

        for true_class, idx in enumerate(indices_by_class):

            n = len(idx)

            sampled_idx = rng.choice(
                idx,
                size=n,
                replace=True,
            )

            pred_A = y_pred_A[sampled_idx]
            pred_B = y_pred_B[sampled_idx]

            for predicted_class in range(N_CLASSES):

                rate_A = np.mean(
                    pred_A == predicted_class
                )
                rate_B = np.mean(
                    pred_B == predicted_class
                )

                boot_deltas[
                    b,
                    true_class,
                    predicted_class
                ] = rate_B - rate_A

    _, cm_A = compute_confusion_matrices(
        y_true,
        y_pred_A,
    )

    _, cm_B = compute_confusion_matrices(
        y_true,
        y_pred_B,
    )

    observed = cm_B - cm_A

    ci_low = np.percentile(
        boot_deltas,
        2.5,
        axis=0,
    )

    ci_high = np.percentile(
        boot_deltas,
        97.5,
        axis=0,
    )

    return observed, ci_low, ci_high


def plot_confusion_difference_with_ci(
    delta,
    ci_low,
    ci_high,
):
    """
    Adds * to cells whose 95% bootstrap CI excludes zero.
    """
    delta_pp = 100 * delta

    reliable = (
        (ci_low > 0) |
        (ci_high < 0)
    )

    max_abs = np.max(np.abs(delta_pp))

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        delta_pp,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(
        "Model B − Model A\n"
        "* = 95% bootstrap CI excludes zero"
    )

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):

            star = "*" if reliable[i, j] else ""

            ax.text(
                j,
                i,
                f"{delta_pp[i, j]:+.1f}{star}",
                ha="center",
                va="center",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage-point difference")

    plt.tight_layout()
    plt.show()


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

def compare_per_class_accuracy(
    y_true,
    y_pred_A,
    y_pred_B,
):
    """
    Paired per-class comparison using the exact McNemar equivalent:
    a two-sided binomial test on discordant predictions.

    Holm correction is applied across the 7 emotions.
    """
    rows = []

    for c, class_name in enumerate(CLASS_NAMES):

        mask = y_true == c

        correct_A = y_pred_A[mask] == c
        correct_B = y_pred_B[mask] == c

        n = mask.sum()

        acc_A = correct_A.mean()
        acc_B = correct_B.mean()

        A_wrong_B_correct = np.sum(
            (~correct_A) & correct_B
        )

        A_correct_B_wrong = np.sum(
            correct_A & (~correct_B)
        )

        n_discordant = (
            A_wrong_B_correct +
            A_correct_B_wrong
        )

        if n_discordant > 0:

            p = binomtest(
                A_wrong_B_correct,
                n=n_discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue

        else:
            p = 1.0

        rows.append({
            "emotion": class_name,
            "n": n,
            "accuracy_A": acc_A,
            "accuracy_B": acc_B,
            "difference_B_minus_A":
                acc_B - acc_A,
            "A_wrong_B_correct":
                A_wrong_B_correct,
            "A_correct_B_wrong":
                A_correct_B_wrong,
            "p_raw": p,
        })

    df = pd.DataFrame(rows)

    # df["p_holm"] = multipletests(
    #     df["p_raw"],
    #     method="holm",
    # )[1]

    return df


def plot_per_class_recall(cm_A, cm_B):
    recall_A = np.diag(cm_A)
    recall_B = np.diag(cm_B)

    x = np.arange(N_CLASSES)
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width / 2,
        100 * recall_A,
        width,
        label="Model A",
    )

    ax.bar(
        x + width / 2,
        100 * recall_B,
        width,
        label="Model B",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Recall (%)")
    ax.set_title(
        "Per-emotion recognition performance"
    )
    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# CONFUSION-TRANSITION TABLES
# ============================================================

def make_confusion_difference_table(
    y_true,
    cm_A,
    cm_B,
    ci_low=None,
    ci_high=None,
):
    """
    One row per true-class / predicted-class combination.
    """
    rows = []

    class_counts = get_class_counts(y_true)

    for true_c in range(N_CLASSES):

        for pred_c in range(N_CLASSES):

            row = {
                "true_emotion":
                    CLASS_NAMES[true_c],
                "predicted_emotion":
                    CLASS_NAMES[pred_c],
                "n_true_class":
                    class_counts[true_c],
                "rate_A":
                    cm_A[true_c, pred_c],
                "rate_B":
                    cm_B[true_c, pred_c],
                "difference":
                    cm_B[true_c, pred_c]
                    - cm_A[true_c, pred_c],
            }

            if ci_low is not None:
                row["ci_low"] = (
                    ci_low[true_c, pred_c]
                )
                row["ci_high"] = (
                    ci_high[true_c, pred_c]
                )

            rows.append(row)

    df = pd.DataFrame(rows)

    df["abs_difference"] = np.abs(
        df["difference"]
    )

    return df.sort_values(
        "abs_difference",
        ascending=False,
    ).reset_index(drop=True)


def transition_table(
    y_true,
    y_pred_A,
    y_pred_B,
):
    """
    Counts exact changes in predictions between the two models.

    Example:
        True Fear
        Model A -> Surprise
        Model B -> Fear
    """
    rows = []

    for true_c in range(N_CLASSES):

        true_mask = y_true == true_c

        for pred_A_c in range(N_CLASSES):

            for pred_B_c in range(N_CLASSES):

                if pred_A_c == pred_B_c:
                    continue

                mask = (
                    true_mask &
                    (y_pred_A == pred_A_c) &
                    (y_pred_B == pred_B_c)
                )

                n = mask.sum()

                if n == 0:
                    continue

                rows.append({
                    "true":
                        CLASS_NAMES[true_c],
                    "model_A_prediction":
                        CLASS_NAMES[pred_A_c],
                    "model_B_prediction":
                        CLASS_NAMES[pred_B_c],
                    "n":
                        n,
                    "B_fixes_A_error":
                        (
                            pred_A_c != true_c
                            and
                            pred_B_c == true_c
                        ),
                    "A_fixes_B_error":
                        (
                            pred_A_c == true_c
                            and
                            pred_B_c != true_c
                        ),
                })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "n",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# RELEVANCE NORMALIZATION
# ============================================================

def normalize_relevance_positive_mass(
    relevance,
    eps=1e-12,
):
    """
    Keep only positive relevance and normalize each image
    so that total positive relevance sums to 1.

    Use this when the scientific question is:
        "Where is the model allocating its relevance?"

    If negative relevance is meaningful for your attribution
    method, analyze positive and negative relevance separately
    instead of using this function.
    """
    rel = np.maximum(
        relevance,
        0,
    )

    totals = rel.sum(
        axis=(1, 2),
        keepdims=True,
    )

    return rel / (totals + eps)


# ============================================================
# RELEVANCE VISUALIZATION
# ============================================================

def plot_relevance_pair(
    maps_A,
    maps_B,
    title,
):
    """
    Compare mean relevance on exactly the same images.
    """
    if len(maps_A) == 0:
        print(
            f"No images available for: {title}"
        )
        return

    mean_A = maps_A.mean(axis=0)
    mean_B = maps_B.mean(axis=0)

    difference = mean_B - mean_A

    common_max = max(
        np.max(mean_A),
        np.max(mean_B),
    )

    diff_max = np.max(
        np.abs(difference)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    im0 = axes[0].imshow(
        mean_A,
        cmap="inferno",
        vmin=0,
        vmax=common_max,
    )

    axes[0].set_title("Model A")
    axes[0].axis("off")

    axes[1].imshow(
        mean_B,
        cmap="inferno",
        vmin=0,
        vmax=common_max,
    )

    axes[1].set_title("Model B")
    axes[1].axis("off")

    im2 = axes[2].imshow(
        difference,
        cmap="RdBu_r",
        vmin=-diff_max,
        vmax=diff_max,
    )

    axes[2].set_title("B − A")
    axes[2].axis("off")

    fig.colorbar(
        im0,
        ax=axes[:2],
        shrink=0.75,
        label="Normalized relevance",
    )

    fig.colorbar(
        im2,
        ax=axes[2],
        shrink=0.75,
        label="Relevance difference",
    )

    fig.suptitle(
        f"{title}\nN = {len(maps_A)}"
    )

    plt.show()


def plot_relevance_for_true_class(
    true_class,
    y_true,
    rel_A,
    rel_B,
):
    c = CLASS_TO_IDX[true_class]

    mask = y_true == c

    plot_relevance_pair(
        rel_A[mask],
        rel_B[mask],
        title=f"True emotion: {true_class}",
    )


# ============================================================
# BEHAVIORAL SUBSETS
# ============================================================

def get_behavior_masks(
    y_true,
    y_pred_A,
    y_pred_B,
    true_class,
):
    """
    Split a true emotion into:
        both correct
        only A correct
        only B correct
        both wrong
    """
    c = CLASS_TO_IDX[true_class]

    true_mask = y_true == c

    correct_A = y_pred_A == c
    correct_B = y_pred_B == c

    return {
        "Both correct":
            true_mask &
            correct_A &
            correct_B,

        "Only A correct":
            true_mask &
            correct_A &
            (~correct_B),

        "Only B correct":
            true_mask &
            (~correct_A) &
            correct_B,

        "Both wrong":
            true_mask &
            (~correct_A) &
            (~correct_B),
    }


def behavioral_subset_counts(
    y_true,
    y_pred_A,
    y_pred_B,
):
    rows = []

    for c, emotion in enumerate(CLASS_NAMES):

        true_mask = y_true == c

        correct_A = y_pred_A == c
        correct_B = y_pred_B == c

        rows.append({
            "emotion":
                emotion,
            "total":
                np.sum(true_mask),
            "both_correct":
                np.sum(
                    true_mask &
                    correct_A &
                    correct_B
                ),
            "A_only_correct":
                np.sum(
                    true_mask &
                    correct_A &
                    (~correct_B)
                ),
            "B_only_correct":
                np.sum(
                    true_mask &
                    (~correct_A) &
                    correct_B
                ),
            "both_wrong":
                np.sum(
                    true_mask &
                    (~correct_A) &
                    (~correct_B)
                ),
        })

    return pd.DataFrame(rows)


def plot_behavioral_relevance(
    true_class,
    y_true,
    y_pred_A,
    y_pred_B,
    rel_A,
    rel_B,
    min_n=MIN_RELEVANCE_SUBSET_N,
):
    masks = get_behavior_masks(
        y_true,
        y_pred_A,
        y_pred_B,
        true_class,
    )

    for group_name, mask in masks.items():

        n = np.sum(mask)

        print(
            f"{true_class} — "
            f"{group_name}: N={n}"
        )

        if n < min_n:
            print(
                "Skipping map because "
                "the subset is very small."
            )
            continue

        plot_relevance_pair(
            rel_A[mask],
            rel_B[mask],
            title=(
                f"{true_class}: "
                f"{group_name}"
            ),
        )


# ============================================================
# EXACT PREDICTION TRANSITIONS
# ============================================================

def exact_transition_mask(
    y_true,
    y_pred_A,
    y_pred_B,
    true_emotion,
    pred_A_emotion,
    pred_B_emotion,
):
    """
    Example:
        true_emotion = Fear
        pred_A_emotion = Surprise
        pred_B_emotion = Fear
    """
    true_c = CLASS_TO_IDX[
        true_emotion
    ]

    pred_A_c = CLASS_TO_IDX[
        pred_A_emotion
    ]

    pred_B_c = CLASS_TO_IDX[
        pred_B_emotion
    ]

    return (
        (y_true == true_c) &
        (y_pred_A == pred_A_c) &
        (y_pred_B == pred_B_c)
    )


def plot_exact_transition(
    y_true,
    y_pred_A,
    y_pred_B,
    rel_A,
    rel_B,
    true_emotion,
    pred_A_emotion,
    pred_B_emotion,
    min_n=MIN_RELEVANCE_SUBSET_N,
):
    mask = exact_transition_mask(
        y_true,
        y_pred_A,
        y_pred_B,
        true_emotion,
        pred_A_emotion,
        pred_B_emotion,
    )

    n = mask.sum()

    print(
        f"True {true_emotion}: "
        f"A -> {pred_A_emotion}, "
        f"B -> {pred_B_emotion}, "
        f"N={n}"
    )

    if n < min_n:
        print(
            "Subset is very small; "
            "interpret descriptively."
        )

    if n == 0:
        return

    plot_relevance_pair(
        rel_A[mask],
        rel_B[mask],
        title=(
            f"True {true_emotion}: "
            f"A -> {pred_A_emotion}, "
            f"B -> {pred_B_emotion}"
        ),
    )


# ============================================================
# MACRO-AVERAGED RELEVANCE
# ============================================================

def macro_average_relevance(
    relevance,
    y_true,
):
    """
    Give every emotion equal weight, irrespective of
    how many test images it contains.
    """
    class_means = []

    for c in range(N_CLASSES):

        mask = y_true == c

        class_means.append(
            relevance[mask].mean(axis=0)
        )

    class_means = np.stack(
        class_means
    )

    return class_means.mean(axis=0)


def plot_macro_relevance(
    macro_A,
    macro_B,
):
    diff = macro_B - macro_A

    vmax = max(
        macro_A.max(),
        macro_B.max(),
    )

    dmax = np.max(
        np.abs(diff)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    im = axes[0].imshow(
        macro_A,
        cmap="inferno",
        vmin=0,
        vmax=vmax,
    )

    axes[0].set_title("Model A")
    axes[0].axis("off")

    axes[1].imshow(
        macro_B,
        cmap="inferno",
        vmin=0,
        vmax=vmax,
    )

    axes[1].set_title("Model B")
    axes[1].axis("off")

    im_diff = axes[2].imshow(
        diff,
        cmap="RdBu_r",
        vmin=-dmax,
        vmax=dmax,
    )

    axes[2].set_title("B − A")
    axes[2].axis("off")

    fig.colorbar(
        im,
        ax=axes[:2],
        shrink=0.75,
    )

    fig.colorbar(
        im_diff,
        ax=axes[2],
        shrink=0.75,
    )

    fig.suptitle(
        "Macro-averaged relevance\n"
        "(each emotion weighted equally)"
    )

    plt.show()


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis(
    y_true,
    y_pred_A,
    y_pred_B,
    relevance_A=None,
    relevance_B=None,
):
    """
    Run the main confusion/performance analysis.

    Relevance analysis is run only if relevance maps are supplied.
    """

    # --------------------------------------------------------
    # Prepare labels
    # --------------------------------------------------------

    y_true = labels_to_int(y_true)
    y_pred_A = labels_to_int(y_pred_A)
    y_pred_B = labels_to_int(y_pred_B)

    if relevance_A is not None:
        relevance_A = np.asarray(
            relevance_A,
            dtype=np.float32,
        )

    if relevance_B is not None:
        relevance_B = np.asarray(
            relevance_B,
            dtype=np.float32,
        )

    validate_inputs(
        y_true,
        y_pred_A,
        y_pred_B,
        relevance_A,
        relevance_B,
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n============================")
    print("CLASS COUNTS")
    print("============================")

    for emotion, n in zip(
        CLASS_NAMES,
        get_class_counts(y_true),
    ):
        print(f"{emotion}: {n}")

    plot_class_counts(y_true)

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    cm_A_counts, cm_A = (
        compute_confusion_matrices(
            y_true,
            y_pred_A,
        )
    )

    cm_B_counts, cm_B = (
        compute_confusion_matrices(
            y_true,
            y_pred_B,
        )
    )

    delta_cm = cm_B - cm_A

    plot_confusion_matrix(
        cm_A,
        "Model A — row-normalized confusion matrix",
        normalized=True,
        vmin=0,
        vmax=1,
    )

    plot_confusion_matrix(
        cm_B,
        "Model B — row-normalized confusion matrix",
        normalized=True,
        vmin=0,
        vmax=1,
    )

    plot_confusion_matrix(
        cm_A_counts,
        "Model A — raw counts",
        normalized=False,
    )

    plot_confusion_matrix(
        cm_B_counts,
        "Model B — raw counts",
        normalized=False,
    )

    plot_confusion_difference_pp(
        delta_cm
    )

    # --------------------------------------------------------
    # Balanced accuracy
    # --------------------------------------------------------

    balanced_accuracy_A = np.mean(
        np.diag(cm_A)
    )

    balanced_accuracy_B = np.mean(
        np.diag(cm_B)
    )

    print("\n============================")
    print("BALANCED ACCURACY")
    print("============================")

    print(
        f"Model A: "
        f"{100 * balanced_accuracy_A:.2f}%"
    )

    print(
        f"Model B: "
        f"{100 * balanced_accuracy_B:.2f}%"
    )

    print(
        f"Difference B - A: "
        f"{100 * (balanced_accuracy_B - balanced_accuracy_A):+.2f} pp"
    )

    plot_per_class_recall(
        cm_A,
        cm_B,
    )

    # --------------------------------------------------------
    # Paired per-class comparison
    # --------------------------------------------------------

    performance_df = (
        compare_per_class_accuracy(
            y_true,
            y_pred_A,
            y_pred_B,
        )
    )

    print("\n============================")
    print("PER-CLASS PAIRED COMPARISON")
    print("============================")

    performance_print = (
        performance_df.copy()
    )

    for col in [
        "accuracy_A",
        "accuracy_B",
        "difference_B_minus_A",
    ]:
        performance_print[col] *= 100

    print(
        performance_print.round({
            "accuracy_A": 1,
            "accuracy_B": 1,
            "difference_B_minus_A": 1,
            "p_raw": 4,
            # "p_holm": 4,
        }).to_string(index=False)
    )

    # --------------------------------------------------------
    # Bootstrap confusion differences
    # --------------------------------------------------------

    print("\nRunning paired bootstrap...")

    delta_boot, ci_low, ci_high = (
        bootstrap_confusion_difference(
            y_true,
            y_pred_A,
            y_pred_B,
        )
    )

    plot_confusion_difference_with_ci(
        delta_boot,
        ci_low,
        ci_high,
    )

    # --------------------------------------------------------
    # Rank confusion changes
    # --------------------------------------------------------

    confusion_change_df = (
        make_confusion_difference_table(
            y_true,
            cm_A,
            cm_B,
            ci_low,
            ci_high,
        )
    )

    print("\n============================")
    print("LARGEST CONFUSION CHANGES")
    print("============================")

    print(
        confusion_change_df.head(20)
        .to_string(index=False)
    )

    off_diagonal_df = (
        confusion_change_df[
            confusion_change_df[
                "true_emotion"
            ]
            !=
            confusion_change_df[
                "predicted_emotion"
            ]
        ]
    )

    print("\n============================")
    print("LARGEST OFF-DIAGONAL CHANGES")
    print("============================")

    print(
        off_diagonal_df.head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Exact transition counts
    # --------------------------------------------------------

    transition_df = transition_table(
        y_true,
        y_pred_A,
        y_pred_B,
    )

    print("\n============================")
    print("MOST COMMON MODEL DISAGREEMENTS")
    print("============================")

    print(
        transition_df.head(30)
        .to_string(index=False)
    )

    print("\n============================")
    print("MODEL B FIXES MODEL A")
    print("============================")

    print(
        transition_df[
            transition_df[
                "B_fixes_A_error"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n============================")
    print("MODEL A FIXES MODEL B")
    print("============================")

    print(
        transition_df[
            transition_df[
                "A_fixes_B_error"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Behavioral subset counts
    # --------------------------------------------------------

    subset_counts_df = (
        behavioral_subset_counts(
            y_true,
            y_pred_A,
            y_pred_B,
        )
    )

    print("\n============================")
    print("BEHAVIORAL SUBSET COUNTS")
    print("============================")

    print(
        subset_counts_df
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Relevance analysis
    # --------------------------------------------------------

    if (
        relevance_A is not None
        and
        relevance_B is not None
    ):

        print("\n============================")
        print("RELEVANCE ANALYSIS")
        print("============================")

        rel_A_norm = (
            normalize_relevance_positive_mass(
                relevance_A
            )
        )

        rel_B_norm = (
            normalize_relevance_positive_mass(
                relevance_B
            )
        )

        # --------------------------------------------
        # Macro-average relevance
        # --------------------------------------------

        macro_A = macro_average_relevance(
            rel_A_norm,
            y_true,
        )

        macro_B = macro_average_relevance(
            rel_B_norm,
            y_true,
        )

        plot_macro_relevance(
            macro_A,
            macro_B,
        )

        # --------------------------------------------
        # Mean relevance for each true emotion
        # --------------------------------------------

        for emotion in CLASS_NAMES:

            plot_relevance_for_true_class(
                emotion,
                y_true,
                rel_A_norm,
                rel_B_norm,
            )

        # --------------------------------------------
        # Behavioral subsets for every emotion
        # --------------------------------------------

        for emotion in CLASS_NAMES:

            plot_behavioral_relevance(
                emotion,
                y_true,
                y_pred_A,
                y_pred_B,
                rel_A_norm,
                rel_B_norm,
            )

    # --------------------------------------------------------
    # Return useful outputs
    # --------------------------------------------------------

    return {
        "cm_A_counts":
            cm_A_counts,
        "cm_B_counts":
            cm_B_counts,
        "cm_A":
            cm_A,
        "cm_B":
            cm_B,
        "delta_cm":
            delta_cm,
        "bootstrap_ci_low":
            ci_low,
        "bootstrap_ci_high":
            ci_high,
        "performance_df":
            performance_df,
        "confusion_change_df":
            confusion_change_df,
        "transition_df":
            transition_df,
        "subset_counts_df":
            subset_counts_df,
    }


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    metadata = pd.read_csv('/home/projects/bagon/ilanaveh/code/Transformer-Explainability/compare_models/attention_maps/'
                           'NATIVE_RESOLUTION/WITH_POOLING/metadata_all_models_native.csv')

    only_pred_maps = metadata.loc[metadata['target_label'] == metadata['pred_label']].reset_index(drop=True)
    only_pred_maps_A = only_pred_maps.loc[only_pred_maps['model'] == MODEL_A].reset_index(drop=True)
    only_pred_maps_B = only_pred_maps.loc[only_pred_maps['model'] == MODEL_B].reset_index(drop=True)
    assert (only_pred_maps_A['true_label'] == only_pred_maps_B['true_label']).all()
    y_true = only_pred_maps_A['true_label']
    y_pred_A = only_pred_maps_A['pred_label']
    y_pred_B = only_pred_maps_B['pred_label']
    maps_A = [np.load('/home/projects/bagon/ilanaveh/code/Transformer-Explainability/compare_models/' + mp) for mp in only_pred_maps_A['map_path']]
    maps_B = [np.load('/home/projects/bagon/ilanaveh/code/Transformer-Explainability/compare_models/' + mp) for mp in only_pred_maps_B['map_path']]
    # --------------------------------------------------------
    # Replace this section with your actual loading code.
    #
    # Required:
    #
    # y_true
    # y_pred_A
    # y_pred_B
    #
    # Optional:
    #
    # relevance_A
    # relevance_B
    #
    # Expected relevance shape:
    #
    #     (N, H, W)
    #
    # Example:
    #
    # y_true = np.load("y_true.npy")
    # y_pred_A = np.load("pred_model_A.npy")
    # y_pred_B = np.load("pred_model_B.npy")
    #
    # relevance_A = np.load("relevance_model_A.npy")
    # relevance_B = np.load("relevance_model_B.npy")
    # --------------------------------------------------------

    # raise RuntimeError(
    #     "Replace the loading section in __main__ "
    #     "with your actual data, then call run_analysis()."
    # )

    # Example:
    #
    results = run_analysis(
        y_true=y_true,
        y_pred_A=y_pred_A,
        y_pred_B=y_pred_B,
        relevance_A=maps_A,
        relevance_B=maps_B,
    )
    #
    #
    # Example of a targeted transition afterwards:

    rel_A_norm = normalize_relevance_positive_mass(
        maps_A
    )

    rel_B_norm = normalize_relevance_positive_mass(
        maps_B
    )

    plot_exact_transition(
        y_true=labels_to_int(y_true),
        y_pred_A=labels_to_int(y_pred_A),
        y_pred_B=labels_to_int(y_pred_B),
        rel_A=rel_A_norm,
        rel_B=rel_B_norm,
        true_emotion="Sad",
        pred_A_emotion="Sad",
        pred_B_emotion="Happy",
    )