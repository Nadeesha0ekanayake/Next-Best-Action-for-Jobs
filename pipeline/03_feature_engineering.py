"""Stage 03 — Feature engineering.

Builds a model-ready matrix (one row per job) with ~40 features across 7 groups:
recency, frequency, financial, lifecycle, behavioural, BE engagement, FE
engagement — plus the v1 label as the training target. Logic is unchanged from
the original; only the I/O (warehouse -> local CSV) differs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import TODAY, TODAY_STR, CHARTS, load_interim, save_interim

sns.set_theme(style="whitegrid", font_scale=1.0)


def safe_days_since(series, reference=TODAY):
    parsed = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
    return (reference - parsed).dt.days


def build_features():
    df_jobs = load_interim("jobs")
    df_quotes = load_interim("quotes")
    df_invoices = load_interim("invoices")
    df_appointments = load_interim("appointments")
    df_estimates = load_interim("estimates")
    df_fe = load_interim("fe_actions")
    df_labels = load_interim("v1_labels")

    feat = df_jobs[["job_key", "account_id", "job_id", "status"]].copy()

    # ---- Group 1: Recency ----
    feat["days_since_created"] = safe_days_since(df_jobs["job_created_timestamp"])
    feat["days_since_last_job_event"] = safe_days_since(df_jobs["last_job_event_ts"])

    ts_parts = [df_jobs[["job_key", "last_job_event_ts"]].rename(columns={"last_job_event_ts": "ts"})]
    for frame, col in [(df_quotes, "last_quote_event_ts"), (df_invoices, "last_invoice_event_ts"),
                       (df_appointments, "last_appt_event_ts"), (df_estimates, "last_estimate_event_ts")]:
        if len(frame):
            ts_parts.append(frame.groupby("job_key")[col].max().reset_index().rename(columns={col: "ts"}))
    all_ts = pd.concat(ts_parts, ignore_index=True)
    all_ts["ts"] = pd.to_datetime(all_ts["ts"], errors="coerce", utc=True).dt.tz_localize(None)
    last_act = all_ts.groupby("job_key")["ts"].max().reset_index()
    last_act["days_since_last_activity"] = (TODAY - last_act["ts"]).dt.days
    feat = feat.merge(last_act[["job_key", "days_since_last_activity"]], on="job_key", how="left")

    if len(df_quotes):
        sq = df_quotes[df_quotes["quote_status"].str.lower().isin(["sent", "issued", "approved", "rejected"])]
        if len(sq):
            g = sq.groupby("job_key")["issued_timestamp"].max().reset_index()
            g["days_since_quote_sent"] = safe_days_since(g["issued_timestamp"])
            feat = feat.merge(g[["job_key", "days_since_quote_sent"]], on="job_key", how="left")
    if "days_since_quote_sent" not in feat:
        feat["days_since_quote_sent"] = np.nan

    bins = [0, 3, 7, 14, 30, 60, 90, 180, 365, float("inf")]
    feat["staleness_bucket"] = pd.cut(feat["days_since_last_activity"], bins=bins,
                                      labels=[1, 2, 3, 4, 5, 6, 7, 8, 9], right=True).astype(float)

    # ---- Group 2: Frequency ----
    def count_per_job(frame, key, out):
        if len(frame):
            g = frame.groupby("job_key")[key].nunique().reset_index().rename(columns={key: out})
            return feat.merge(g, on="job_key", how="left")
        feat[out] = 0
        return feat

    feat = count_per_job(df_quotes, "quote_key", "quote_count")
    feat = count_per_job(df_invoices, "invoice_key", "invoice_count")
    feat = count_per_job(df_appointments, "appointment_key", "appointment_count")
    feat = count_per_job(df_estimates, "estimate_key", "estimate_count")
    for c in ["quote_count", "invoice_count", "appointment_count", "estimate_count"]:
        feat[c] = feat[c].fillna(0).astype(int)
    feat["total_entity_count"] = feat[["quote_count", "invoice_count", "appointment_count", "estimate_count"]].sum(axis=1)

    # ---- Group 3: Financial ----
    if len(df_quotes):
        qf = df_quotes.groupby("job_key").agg(max_quote_total=("total", "max"),
                                              sum_quote_total=("total", "sum"),
                                              has_approved_quote=("is_approved", "any")).reset_index()
        qf["has_approved_quote"] = qf["has_approved_quote"].astype(int)
        feat = feat.merge(qf, on="job_key", how="left")
    if len(df_invoices):
        inf = df_invoices.groupby("job_key").agg(max_invoice_total=("total", "max"),
                                                 sum_invoice_total=("total", "sum"),
                                                 has_paid_invoice=("paid_timestamp", lambda x: x.notna().any()),
                                                 has_overdue_invoice=("is_overdue", "any")).reset_index()
        inf["has_paid_invoice"] = inf["has_paid_invoice"].astype(int)
        inf["has_overdue_invoice"] = inf["has_overdue_invoice"].astype(int)
        feat = feat.merge(inf, on="job_key", how="left")
        ov = df_invoices[df_invoices["is_overdue"] == True]  # noqa: E712
        if len(ov):
            feat = feat.merge(ov.groupby("job_key")["total"].sum().reset_index().rename(columns={"total": "overdue_amount"}), on="job_key", how="left")
        unpaid = df_invoices[(df_invoices["invoice_status"].str.lower().isin(["sent", "issued", "overdue"])) & (df_invoices["paid_timestamp"].isna())]
        if len(unpaid):
            feat = feat.merge(unpaid.groupby("job_key")["total"].sum().reset_index().rename(columns={"total": "unpaid_invoice_amount"}), on="job_key", how="left")
    for c in ["max_quote_total", "sum_quote_total", "has_approved_quote", "max_invoice_total",
              "sum_invoice_total", "has_paid_invoice", "has_overdue_invoice", "overdue_amount", "unpaid_invoice_amount"]:
        if c not in feat:
            feat[c] = 0
        feat[c] = feat[c].fillna(0)
    if len(df_quotes):
        su = df_quotes[(df_quotes["quote_status"].str.lower().isin(["sent", "issued"])) & (~df_quotes["is_approved"].fillna(False))]
        if len(su):
            feat = feat.merge(su.groupby("job_key")["total"].sum().reset_index().rename(columns={"total": "sent_quote_amount"}), on="job_key", how="left")
    if "sent_quote_amount" not in feat:
        feat["sent_quote_amount"] = 0.0
    feat["sent_quote_amount"] = feat["sent_quote_amount"].fillna(0)
    feat["money_at_risk"] = feat["unpaid_invoice_amount"] + feat["sent_quote_amount"]
    for c in ["max_quote_total", "sum_quote_total", "max_invoice_total", "sum_invoice_total", "money_at_risk", "overdue_amount"]:
        feat[f"log_{c}"] = np.log1p(feat[c].clip(lower=0))

    # ---- Group 4: Lifecycle ----
    feat["has_quote"] = (feat["quote_count"] > 0).astype(int)
    feat["has_invoice"] = (feat["invoice_count"] > 0).astype(int)
    feat["has_appointment"] = (feat["appointment_count"] > 0).astype(int)
    feat["has_estimate"] = (feat["estimate_count"] > 0).astype(int)
    feat["has_any_entity"] = (feat["total_entity_count"] > 0).astype(int)
    feat["completeness_score"] = (feat["has_estimate"] + feat["has_quote"]
                                  + feat["days_since_quote_sent"].notna().astype(int)
                                  + feat["has_approved_quote"] + feat["has_appointment"]
                                  + feat["has_invoice"] + feat["has_paid_invoice"])

    def stage(r):
        for flag, val in [("has_paid_invoice", 6), ("has_invoice", 5), ("has_appointment", 4),
                          ("has_approved_quote", 3), ("has_quote", 2), ("has_estimate", 1)]:
            if r[flag]:
                return val
        return 0
    feat["lifecycle_stage"] = feat.apply(stage, axis=1)

    for frame, statuses, out in [(df_quotes, ["draft", "created"], "has_draft_quote"),
                                 (df_quotes, ["sent", "issued"], "has_sent_quote"),
                                 (df_invoices, ["draft", "created"], "has_draft_invoice"),
                                 (df_invoices, ["sent", "issued", "overdue"], "has_sent_invoice")]:
        col = "quote_status" if "quote" in out else "invoice_status"
        if len(frame):
            g = frame.groupby("job_key")[col].apply(lambda x: int(x.str.lower().isin(statuses).any())).reset_index().rename(columns={col: out})
            feat = feat.merge(g, on="job_key", how="left")
        if out not in feat:
            feat[out] = 0
        feat[out] = feat[out].fillna(0).astype(int)

    # ---- Group 5: Behavioural ----
    kws = ["leak", "flooding", "urgent", "asap", "broken", "no hot water", "safety",
           "emergency", "burst", "flood", "damage", "dangerous"]
    text = (df_jobs["job_title"].fillna("") + " " + df_jobs["job_description"].fillna("")).str.lower()
    feat["has_emergency_language"] = text.apply(lambda t: int(any(k in t for k in kws)))
    sl = df_jobs["status"].fillna("").str.lower()
    feat["status_indicates_done"] = sl.isin(["done", "work completed", "invoiced", "completed"]).astype(int)
    feat["status_is_cancelled"] = sl.isin(["cancelled", "canceled"]).astype(int)
    feat["status_is_new"] = (sl == "new").astype(int)
    feat["status_is_quoted"] = (sl == "quoted").astype(int)
    feat["status_is_confirmed"] = (sl == "confirmed").astype(int)
    feat["status_mismatch"] = ((sl == "new") & (feat["has_any_entity"] == 1)).astype(int)
    if len(df_appointments):
        fut = df_appointments[pd.to_datetime(df_appointments["start_date"], errors="coerce").astype(str) > TODAY_STR]
        feat["has_future_appointment"] = feat["job_key"].isin(set(fut["job_key"])).astype(int)
    else:
        feat["has_future_appointment"] = 0
    feat["work_done_no_invoice"] = ((feat["status_indicates_done"] == 1) & (feat["has_invoice"] == 0)).astype(int)
    feat["approved_no_next_step"] = ((feat["has_approved_quote"] == 1) & (feat["has_appointment"] == 0) & (feat["has_invoice"] == 0)).astype(int)
    feat["estimate_no_quote"] = ((feat["has_estimate"] == 1) & (feat["has_quote"] == 0)).astype(int)
    feat["draft_quote_unsent"] = ((feat["has_draft_quote"] == 1) & (feat["has_sent_quote"] == 0)).astype(int)
    feat["draft_invoice_unsent"] = ((feat["has_draft_invoice"] == 1) & (feat["has_sent_invoice"] == 0)).astype(int)

    # ---- Group 6: BE engagement ----
    feat["has_customer"] = df_jobs["customer_id"].notna().astype(int)
    feat["estimate_to_quote"] = ((feat["has_estimate"] == 1) & (feat["has_quote"] == 1)).astype(int)
    feat["quote_to_invoice"] = ((feat["has_quote"] == 1) & (feat["has_invoice"] == 1)).astype(int)
    ls = df_jobs["lead_source"].fillna("unknown").str.lower()
    for src in ls.value_counts().head(5).index:
        feat[f"lead_source_{src}"] = (ls == src).astype(int)

    # ---- Group 7: FE engagement ----
    if len(df_fe):
        df_fe["account_id"] = df_fe["account_id"].astype(str)
        fe = df_fe.groupby("account_id").agg(
            fe_total_action_count=("action_category", "size"),
            fe_distinct_action_categories=("action_category", "nunique"),
            fe_first=("action_date", "min"), fe_last=("action_date", "max")).reset_index()
        fe["fe_last"] = pd.to_datetime(fe["fe_last"], errors="coerce")
        fe["fe_first"] = pd.to_datetime(fe["fe_first"], errors="coerce")
        fe["fe_days_since_last_action"] = (TODAY - fe["fe_last"]).dt.days
        span = (fe["fe_last"] - fe["fe_first"]).dt.days + 1
        fe["fe_action_velocity"] = fe["fe_total_action_count"] / span.clip(lower=1)
        fe = fe.drop(columns=["fe_first", "fe_last"])
        feat["account_id"] = feat["account_id"].astype(str)
        feat = feat.merge(fe, on="account_id", how="left")
        feat["fe_has_any_action"] = (feat["fe_total_action_count"].fillna(0) > 0).astype(int)
    for c in [c for c in feat.columns if c.startswith("fe_")]:
        feat[c] = feat[c].fillna(0)

    # ---- Target ----
    feat = feat.merge(df_labels[["job_key", "v1_category", "v1_money_at_risk"]], on="job_key", how="left")
    feat["target"] = feat["v1_category"].map({"NoAction": 0, "Cold": 1, "Require Follow Up": 2, "Urgent": 3})

    meta_cols = {"job_key", "account_id", "job_id", "status", "v1_category",
                 "v1_money_at_risk", "target", "sent_quote_amount"}
    feature_cols = [c for c in feat.columns if c not in meta_cols and feat[c].dtype in ("int64", "float64", "int32", "float32")]
    for c in feature_cols:
        if feat[c].isnull().any():
            feat[c] = feat[c].fillna(0 if feat[c].dtype.kind == "i" else feat[c].median())
    return feat, feature_cols


def charts(feat, feature_cols):
    CHARTS.mkdir(parents=True, exist_ok=True)
    corr = feat[feature_cols + ["target"]].corr()["target"].drop("target").sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top = corr.head(20)
    top.plot(kind="barh", ax=ax, color=["#FF4444" if v > 0 else "#1F4E79" for v in top.values],
             edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Top 20 features correlated with urgency target")
    ax.set_xlabel("Correlation with target")
    fig.tight_layout(); fig.savefig(CHARTS / "03_feature_target_correlation.png", dpi=140); plt.close(fig)

    cols = corr.head(15).index.tolist() + ["target"]
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(feat[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Feature correlation heatmap (top 15 + target)")
    fig.tight_layout(); fig.savefig(CHARTS / "03_feature_correlation_heatmap.png", dpi=140); plt.close(fig)


def main():
    feat, feature_cols = build_features()
    print(f"Feature matrix: {feat.shape[0]:,} jobs x {len(feature_cols)} features")
    charts(feat, feature_cols)
    save_interim(feat, "features")
    save_interim(pd.DataFrame({"feature_name": feature_cols}), "feature_list")
    print("Charts: 03_feature_target_correlation.png, 03_feature_correlation_heatmap.png")


if __name__ == "__main__":
    main()
