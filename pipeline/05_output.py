"""Stage 05 — Output & deliverables.

Takes the scored jobs from stage 04, assigns the single best NEXT ACTION per job
from its predicted category + entity state, summarises money at risk and
per-account breakdowns, and saves the final deliverable tables + charts. Logic
unchanged from the original; I/O swapped to local CSV.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import CHARTS, load_interim, save_interim

sns.set_theme(style="whitegrid", font_scale=1.0)
CAT_ORDER = ["Urgent", "Require Follow Up", "Cold", "NoAction"]
CAT_COLORS = ["#FF4444", "#FFD700", "#87CEEB", "#90EE90"]


def assign_next_action(row):
    cat = row.get("predicted_category", "NoAction")
    if cat == "NoAction":
        return pd.Series({"next_action_type": None, "next_action_entity": None,
                          "next_action_reason": "No action needed"})
    rules = [
        ("has_overdue_invoice", "ChasePayment", "Invoice", "Overdue invoice needs payment follow-up"),
        ("work_done_no_invoice", "CreateInvoice", "Invoice", "Work completed but no invoice created"),
        ("draft_invoice_unsent", "SendInvoice", "Invoice", "Draft invoice exists but not sent"),
        ("approved_no_next_step", "BookAppointment", "Appointment", "Quote approved but no appointment booked"),
        ("draft_quote_unsent", "SendQuote", "Quote", "Draft quote exists but not sent"),
        ("estimate_no_quote", "CreateQuote", "Quote", "Estimate sent but no quote created"),
    ]
    for flag, atype, entity, reason in rules:
        if row.get(flag, 0) == 1:
            return pd.Series({"next_action_type": atype, "next_action_entity": entity, "next_action_reason": reason})
    if row.get("status_is_confirmed", 0) == 1 and row.get("has_appointment", 0) == 0:
        return pd.Series({"next_action_type": "BookAppointment", "next_action_entity": "Appointment",
                          "next_action_reason": "Job confirmed but no appointment booked"})
    if row.get("has_sent_quote", 0) == 1 and row.get("has_approved_quote", 0) == 0:
        return pd.Series({"next_action_type": "FollowUpQuote", "next_action_entity": "Quote",
                          "next_action_reason": "Quote sent but no response yet"})
    if row.get("has_quote", 0) == 0 and cat in ("Urgent", "Require Follow Up"):
        return pd.Series({"next_action_type": "CreateQuote", "next_action_entity": "Quote",
                          "next_action_reason": "No quote exists — create one to progress the job"})
    return pd.Series({"next_action_type": "ReviewJob", "next_action_entity": "Job",
                      "next_action_reason": "General review recommended"})


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    scored = load_interim("scored_jobs")
    feats = load_interim("features")

    ctx = ["job_key", "has_quote", "has_invoice", "has_appointment", "has_estimate",
           "has_draft_quote", "has_sent_quote", "has_approved_quote", "has_draft_invoice",
           "has_sent_invoice", "has_overdue_invoice", "has_paid_invoice", "has_future_appointment",
           "work_done_no_invoice", "approved_no_next_step", "estimate_no_quote",
           "draft_quote_unsent", "draft_invoice_unsent", "status_is_confirmed", "status_indicates_done"]
    scored = scored.merge(feats[[c for c in ctx if c in feats.columns]], on="job_key", how="left")

    actions = scored.apply(assign_next_action, axis=1)
    scored = pd.concat([scored, actions], axis=1)
    print("Next action distribution:")
    print(scored["next_action_type"].value_counts(dropna=False).to_string())

    # ---- Category distribution ----
    cat_counts = scored["predicted_category"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cat_counts.reindex(CAT_ORDER).plot(kind="bar", ax=axes[0], color=CAT_COLORS, edgecolor="black", linewidth=0.5)
    axes[0].set_title("v2 predicted category distribution"); axes[0].set_ylabel("Jobs")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].pie(cat_counts.reindex(CAT_ORDER).values, labels=CAT_ORDER, autopct="%1.1f%%",
                colors=CAT_COLORS, startangle=90)
    axes[1].set_title("Category proportions")
    fig.suptitle("NBA — job category distribution", fontweight="bold")
    fig.tight_layout(); fig.savefig(CHARTS / "05_category_distribution.png", dpi=140); plt.close(fig)

    # ---- v1 vs v2 ----
    v1 = scored["v1_category"].value_counts().reindex(CAT_ORDER).fillna(0)
    v2 = cat_counts.reindex(CAT_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CAT_ORDER))
    ax.bar(x - 0.2, v1.values, 0.4, label="v1 (rules)", color="#C0C0C0", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, v2.values, 0.4, label="v2 (model)", color=CAT_COLORS, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CAT_ORDER, rotation=45); ax.set_ylabel("Jobs")
    ax.set_title("v1 (rule-based) vs v2 (model) category distribution", fontweight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(CHARTS / "05_v1_vs_v2.png", dpi=140); plt.close(fig)

    # ---- Next actions ----
    ac = scored["next_action_type"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ac.plot(kind="barh", ax=ax, color=sns.color_palette("Set2", len(ac)), edgecolor="black", linewidth=0.5)
    ax.set_title("Recommended next actions", fontweight="bold"); ax.set_xlabel("Jobs")
    fig.tight_layout(); fig.savefig(CHARTS / "05_next_actions.png", dpi=140); plt.close(fig)

    # ---- Money at risk ----
    if "money_at_risk" in scored.columns:
        mar = scored.groupby("predicted_category")["money_at_risk"].sum().reindex(CAT_ORDER)
        fig, ax = plt.subplots(figsize=(8, 5))
        mar.plot(kind="bar", ax=ax, color=CAT_COLORS, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Total money at risk ($)"); ax.set_title("Money at risk by category", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(CHARTS / "05_money_at_risk.png", dpi=140); plt.close(fig)
        total = scored[scored["predicted_category"].isin(["Urgent", "Require Follow Up"])]["money_at_risk"].sum()
        print(f"Total money at risk (actionable): ${total:,.0f}")

    # ---- Per-account summary ----
    acct = scored.groupby("account_id").agg(
        total_jobs=("job_key", "count"),
        urgent_count=("predicted_category", lambda x: (x == "Urgent").sum()),
        followup_count=("predicted_category", lambda x: (x == "Require Follow Up").sum()),
        cold_count=("predicted_category", lambda x: (x == "Cold").sum()),
        total_money_at_risk=("money_at_risk", "sum"),
        avg_confidence=("confidence", "mean")).reset_index()
    acct = acct.sort_values("total_money_at_risk", ascending=False)

    out_cols = ["job_key", "account_id", "job_id", "status", "predicted_category",
                "confidence", "prob_null", "prob_cold", "prob_followup", "prob_urgent",
                "next_action_type", "next_action_entity", "next_action_reason",
                "money_at_risk", "days_since_created", "days_since_last_activity",
                "total_entity_count", "completeness_score", "lifecycle_stage",
                "has_emergency_language", "has_overdue_invoice", "fe_total_action_count", "v1_category"]
    save_interim(scored[[c for c in out_cols if c in scored.columns]], "final_output")
    save_interim(acct, "account_summary")
    print("Charts: 05_category_distribution.png, 05_v1_vs_v2.png, 05_next_actions.png, 05_money_at_risk.png")


if __name__ == "__main__":
    main()
