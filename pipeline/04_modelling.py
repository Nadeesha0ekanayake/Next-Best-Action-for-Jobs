"""Stage 04 — Modelling (multinomial logistic regression).

Trains a 4-class model (null / Cold / Require Follow Up / Urgent) on the feature
matrix from stage 03, evaluates it, interprets coefficients, and scores every
job. Class weights are balanced to handle the imbalance. Logic unchanged from
the original; I/O swapped to local CSV and charts saved to files.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

from common import CHARTS, load_interim, save_interim

sns.set_theme(style="whitegrid", font_scale=1.0)
TARGET_NAMES = ["NoAction", "Cold", "Follow Up", "Urgent"]
CLASS_COLORS = ["#90EE90", "#87CEEB", "#FFD700", "#FF4444"]


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    df = load_interim("features")
    feature_cols = load_interim("feature_list")["feature_name"].tolist()

    X = df[feature_cols].copy()
    for col in X.columns[X.isnull().any().values]:
        X[col] = X[col].fillna(-1 if "days_since" in col else 0)
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0,
                               max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print("Classification report (test):")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES, digits=3))
    print(f"Accuracy: {acc:.3f}")

    # --- Confusion matrix chart ---
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    axes[0].set_title("Confusion Matrix (counts)"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=axes[1],
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    axes[1].set_title("Confusion Matrix (row-normalised = recall)"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    fig.suptitle("Model Performance — Confusion Matrix", fontweight="bold")
    fig.tight_layout(); fig.savefig(CHARTS / "04_confusion_matrix.png", dpi=140); plt.close(fig)

    # --- ROC curves (OvR) ---
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(8, 7))
    auc_scores = {}
    for i, (label, color) in enumerate(zip(TARGET_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        auc_scores[label] = auc
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(CHARTS / "04_roc_curves.png", dpi=140); plt.close(fig)
    macro_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
    print(f"Macro AUC: {macro_auc:.3f}")

    # --- Coefficients ---
    coef_df = pd.DataFrame(model.coef_, columns=feature_cols, index=TARGET_NAMES).T
    coef_df["abs_max"] = coef_df.abs().max(axis=1)
    coef_df = coef_df.sort_values("abs_max", ascending=False)

    top = coef_df.head(18).drop(columns="abs_max")
    fig, axes = plt.subplots(1, 4, figsize=(20, 9), sharey=True)
    for i, (cls, color) in enumerate(zip(TARGET_NAMES, CLASS_COLORS)):
        s = top[cls].sort_values()
        s.plot(kind="barh", ax=axes[i], edgecolor="black", linewidth=0.5,
               color=[color if v > 0 else "#C0C0C0" for v in s.values])
        axes[i].axvline(0, color="black", lw=0.5); axes[i].set_title(cls, fontweight="bold")
        axes[i].set_xlabel("Coefficient")
    fig.suptitle("Top feature coefficients by class", fontweight="bold")
    fig.tight_layout(); fig.savefig(CHARTS / "04_coefficients_by_class.png", dpi=140); plt.close(fig)

    # --- Score all jobs ---
    X_all = df[feature_cols].copy()
    for col in X_all.columns[X_all.isnull().any().values]:
        X_all[col] = X_all[col].fillna(-1 if "days_since" in col else 0)
    all_proba = model.predict_proba(scaler.transform(X_all))
    all_pred = model.predict(scaler.transform(X_all))
    cat_map = {0: "NoAction", 1: "Cold", 2: "Require Follow Up", 3: "Urgent"}

    scored = df[["job_key", "account_id", "job_id", "status", "v1_category", "target"]].copy()
    scored["predicted_category"] = pd.Series(all_pred).map(cat_map)
    scored["prob_null"], scored["prob_cold"] = all_proba[:, 0], all_proba[:, 1]
    scored["prob_followup"], scored["prob_urgent"] = all_proba[:, 2], all_proba[:, 3]
    scored["confidence"] = all_proba.max(axis=1)
    for col in ["days_since_created", "days_since_last_activity", "money_at_risk",
                "total_entity_count", "completeness_score", "lifecycle_stage",
                "has_emergency_language", "has_overdue_invoice", "fe_total_action_count", "fe_has_any_action"]:
        if col in df.columns:
            scored[col] = df[col].values

    agreement = (scored["v1_category"] == scored["predicted_category"]).mean()
    print(f"v1 vs v2 agreement: {agreement:.1%}")

    save_interim(scored, "scored_jobs")
    coef_out = coef_df.drop(columns="abs_max").reset_index().rename(columns={"index": "feature"})
    coef_out.columns = [c.replace(" ", "_") for c in coef_out.columns]
    save_interim(coef_out, "model_coefficients")
    save_interim(pd.DataFrame([{
        "model_type": "LogisticRegression", "class_weight": "balanced", "C": 1.0,
        "n_features": len(feature_cols), "n_train": len(X_train), "n_test": len(X_test),
        "accuracy": float(acc), "macro_auc": float(macro_auc),
    }]), "model_metadata")
    print("Charts: 04_confusion_matrix.png, 04_roc_curves.png, 04_coefficients_by_class.png")


if __name__ == "__main__":
    main()
