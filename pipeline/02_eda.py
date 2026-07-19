"""Stage 02 — Exploratory data analysis.

Profiles the extracted data and saves the key EDA charts: entity coverage, job
staleness, the lifecycle conversion funnel, front-end engagement, and the v1
label class balance. Logic unchanged; charts are saved to files instead of shown
inline.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import TODAY, CHARTS, load_interim

sns.set_theme(style="whitegrid", font_scale=1.0)


def safe_days_since(series, reference=TODAY):
    parsed = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
    return (reference - parsed).dt.days


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    jobs = load_interim("jobs")
    quotes = load_interim("quotes")
    invoices = load_interim("invoices")
    appts = load_interim("appointments")
    estimates = load_interim("estimates")
    fe = load_interim("fe_actions")
    labels = load_interim("v1_labels")

    for frame, flag in [(quotes, "has_quotes"), (invoices, "has_invoices"),
                        (appts, "has_appointments"), (estimates, "has_estimates")]:
        jobs[flag] = jobs["job_key"].isin(set(frame["job_key"])) if len(frame) else False
    jobs["has_any_entity"] = jobs[["has_quotes", "has_invoices", "has_appointments", "has_estimates"]].any(axis=1)
    jobs["days_since_last_event"] = safe_days_since(jobs["last_job_event_ts"])

    # ---- Entity coverage ----
    fig, ax = plt.subplots(figsize=(9, 5))
    types = ["Quotes", "Invoices", "Appointments", "Estimates"]
    counts = [jobs["has_quotes"].sum(), jobs["has_invoices"].sum(),
              jobs["has_appointments"].sum(), jobs["has_estimates"].sum()]
    bars = ax.bar(types, counts, color=["#FF8C00", "#1F4E79", "#2CA02C", "#9467BD"], edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{c:,}\n({c/len(jobs)*100:.0f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Jobs"); ax.set_title(f"Jobs with each entity type (of {len(jobs):,})")
    ghost = (~jobs["has_any_entity"]).sum()
    ax.text(0.98, 0.95, f"Ghost leads (no entities): {ghost:,} ({ghost/len(jobs)*100:.0f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="grey"))
    fig.tight_layout(); fig.savefig(CHARTS / "02_entity_coverage.png", dpi=140); plt.close(fig)

    # ---- Staleness buckets ----
    bins = [0, 3, 7, 14, 30, 60, 90, 180, 365, float("inf")]
    bl = ["0-3d", "4-7d", "8-14d", "15-30d", "31-60d", "61-90d", "91-180d", "181-365d", "365d+"]
    buckets = pd.cut(jobs["days_since_last_event"], bins=bins, labels=bl, right=True).value_counts().reindex(bl)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2CA02C", "#2CA02C", "#FFD700", "#FFD700", "#FF8C00", "#FF4444", "#FF4444", "#C0C0C0", "#C0C0C0"]
    buckets.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Days since last event"); ax.set_ylabel("Jobs")
    ax.set_title("Job staleness buckets (aligned with category thresholds)")
    plt.xticks(rotation=45)
    fig.tight_layout(); fig.savefig(CHARTS / "02_staleness_buckets.png", dpi=140); plt.close(fig)

    # ---- Conversion funnel ----
    total = len(jobs)
    q_sent = quotes[quotes["quote_status"].str.lower().isin(["sent", "issued", "approved", "rejected"])]["job_key"].nunique() if len(quotes) else 0
    q_appr = quotes[quotes["is_approved"] == True]["job_key"].nunique() if len(quotes) else 0  # noqa: E712
    i_paid = invoices[(invoices["invoice_status"].str.lower() == "paid") | (invoices["paid_timestamp"].notna())]["job_key"].nunique() if len(invoices) else 0
    stages = ["All jobs", "Quote created", "Quote sent", "Quote approved", "Appointment", "Invoice", "Invoice paid"]
    vals = [total, jobs["has_quotes"].sum(), q_sent, q_appr, jobs["has_appointments"].sum(), jobs["has_invoices"].sum(), i_paid]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(stages[::-1], vals[::-1], color=sns.color_palette("YlOrRd_r", len(stages)), edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + 3, b.get_y() + b.get_height() / 2, f"{v:,} ({v/total*100:.0f}%)", va="center", fontsize=9)
    ax.set_xlabel("Jobs"); ax.set_title("Job lifecycle conversion funnel", fontweight="bold")
    fig.tight_layout(); fig.savefig(CHARTS / "02_conversion_funnel.png", dpi=140); plt.close(fig)

    # ---- FE engagement ----
    if len(fe):
        per_acct = fe.groupby("account_id").size()
        cat_counts = fe["action_category"].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cat_counts.plot(kind="barh", ax=axes[0], color=sns.color_palette("Set2", len(cat_counts)), edgecolor="black", linewidth=0.5)
        axes[0].set_title("FE action category distribution"); axes[0].set_xlabel("Events")
        axes[1].hist(per_acct, bins=25, color="#2CA02C", edgecolor="black", linewidth=0.5)
        axes[1].axvline(per_acct.median(), color="red", linestyle="--", label=f"Median: {per_acct.median():.0f}")
        axes[1].set_xlabel("FE actions per account"); axes[1].set_ylabel("Accounts"); axes[1].legend()
        axes[1].set_title("FE action volume per account")
        fig.suptitle("Front-end tradie engagement", fontweight="bold")
        fig.tight_layout(); fig.savefig(CHARTS / "02_fe_engagement.png", dpi=140); plt.close(fig)

    # ---- v1 label class balance ----
    lc = labels["v1_category"].value_counts()
    palette = {"Urgent": "#FF4444", "Require Follow Up": "#FFD700", "Cold": "#87CEEB", "NoAction": "#90EE90"}
    fig, ax = plt.subplots(figsize=(8, 5))
    lc.plot(kind="bar", ax=ax, color=[palette.get(c, "#999") for c in lc.index], edgecolor="black", linewidth=0.5)
    for i, v in enumerate(lc.values):
        ax.text(i, v + 2, f"{v:,}\n({v/len(labels)*100:.0f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Jobs"); ax.set_title("v1 label distribution (training target)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(CHARTS / "02_label_balance.png", dpi=140); plt.close(fig)

    print(f"Jobs {len(jobs):,} | Ghost leads {ghost:,} ({ghost/len(jobs)*100:.0f}%)")
    print("Charts: 02_entity_coverage, 02_staleness_buckets, 02_conversion_funnel, 02_fe_engagement, 02_label_balance")


if __name__ == "__main__":
    main()
