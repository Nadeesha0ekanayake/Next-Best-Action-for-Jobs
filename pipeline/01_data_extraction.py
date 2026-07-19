"""Stage 01 — Data extraction + v1 rule-based labels.

METHODOLOGY / QUERIES (production):
    In production this stage pulls the latest state of each entity for a cohort
    of tradie accounts from the warehouse, e.g.:

        SELECT entity_id, account_id, job_id, job_title, status, ...
        FROM   your_catalog.silver.jobs_changed
        WHERE  account_id IN (:cohort)
          AND  is_last_entity_revision = true

    with equivalent queries for invoices, quotes, appointments, estimates, and a
    gold-layer front-end action-events table. Table names here are GENERIC
    PLACEHOLDERS (see common.SOURCE_TABLES); the real cohort of account IDs is
    NOT included. This runnable version reads the synthetic CSVs from stage 00.

It then applies the v1 rule-based categorisation (Urgent / Require Follow Up /
Cold / NoAction) that becomes the training target for the model in stage 04.
"""

import numpy as np
import pandas as pd

from common import TODAY, TODAY_STR, load_raw, save_interim


def safe_days_since(date_val, reference=TODAY):
    if date_val is None or (isinstance(date_val, float) and pd.isna(date_val)):
        return None
    try:
        dt = pd.to_datetime(date_val, utc=True).tz_localize(None)
        return (reference - dt).days
    except Exception:
        return None


def has_urgency_language(text):
    if not text:
        return False
    kws = ["leak", "flooding", "urgent", "asap", "broken", "no hot water",
           "safety", "emergency", "burst", "flood", "damage", "dangerous"]
    return any(kw in str(text).lower() for kw in kws)


def determine_work_done(job_row, job_invoices, job_appointments):
    status = str(job_row.get("status") or "").lower()
    if status in ("done", "work completed", "invoiced", "completed"):
        return "yes"
    for _, apt in job_appointments.iterrows():
        if "completed" in str(apt.get("last_appt_event_type") or "").lower():
            return "yes"
    if len(job_appointments) == 0 and status in ("new", "quoted", "confirmed", ""):
        return "no"
    return "unclear"


def add_key(df, id_col, name):
    df[name] = df["account_id"].astype(str) + "|" + df[id_col].astype(str)
    df["job_key"] = df["account_id"].astype(str) + "|" + df["job_id"].astype(str)
    return df


def build_labels(df_jobs, df_quotes, df_invoices, df_appointments, df_estimates):
    inv_by_job = df_invoices.groupby("job_key") if len(df_invoices) else None
    qt_by_job = df_quotes.groupby("job_key") if len(df_quotes) else None
    apt_by_job = df_appointments.groupby("job_key") if len(df_appointments) else None
    est_by_job = df_estimates.groupby("job_key") if len(df_estimates) else None
    empty = pd.DataFrame()

    def grp(g, k):
        if g is None:
            return empty
        try:
            return g.get_group(k)
        except KeyError:
            return empty

    records = []
    for _, job in df_jobs.iterrows():
        jk = job["job_key"]
        job_inv, job_qt = grp(inv_by_job, jk), grp(qt_by_job, jk)
        job_apt, job_est = grp(apt_by_job, jk), grp(est_by_job, jk)

        status_lower = str(job.get("status") or "").strip().lower()
        title, desc = str(job.get("job_title") or ""), str(job.get("job_description") or "")
        created_days = safe_days_since(job.get("job_created_timestamp"))

        is_demo = any(len(x) and x["job_is_demo"].any() for x in (job_qt, job_apt, job_est)) \
            or "demo" in (title + desc).lower() or "test" in (title + desc).lower()
        has_engagement = any(len(x) for x in (job_qt, job_inv, job_apt, job_est))

        sent_q = job_qt[job_qt["quote_status"].str.lower().isin(["sent", "issued"])] if len(job_qt) else empty
        draft_q = job_qt[job_qt["quote_status"].str.lower().isin(["draft", "created"])] if len(job_qt) else empty
        appr_q = job_qt[job_qt["is_approved"] == True] if len(job_qt) else empty  # noqa: E712

        sent_inv = job_inv[job_inv["invoice_status"].str.lower().isin(["sent", "issued", "overdue"])] if len(job_inv) else empty
        draft_inv = job_inv[job_inv["invoice_status"].str.lower().isin(["draft", "created"])] if len(job_inv) else empty
        paid_inv = job_inv[(job_inv["invoice_status"].str.lower() == "paid") | (job_inv["paid_timestamp"].notna())] if len(job_inv) else empty
        overdue_inv = job_inv[job_inv["is_overdue"] == True] if len(job_inv) else empty  # noqa: E712
        future_appts = job_apt[job_apt["start_date"].astype(str) > TODAY_STR] if len(job_apt) else empty

        all_ts = [str(job.get("last_job_event_ts"))] if job.get("last_job_event_ts") else []
        for frame, col in [(job_inv, "last_invoice_event_ts"), (job_qt, "last_quote_event_ts"),
                           (job_apt, "last_appt_event_ts"), (job_est, "last_estimate_event_ts")]:
            all_ts += [str(v) for v in frame.get(col, pd.Series(dtype=object)).dropna()]
        last_activity = max(all_ts) if all_ts else None
        days_inactive = safe_days_since(last_activity) if last_activity else (created_days or 999)
        days_inactive = 999 if days_inactive is None else days_inactive

        money_at_risk = 0.0
        for _, inv in job_inv.iterrows():
            if str(inv.get("invoice_status") or "").lower() in ("sent", "issued", "overdue") and pd.isna(inv.get("paid_timestamp")):
                money_at_risk += float(inv.get("total") or 0)
        for _, qt in job_qt.iterrows():
            if str(qt.get("quote_status") or "").lower() in ("sent", "issued") and not qt.get("is_approved"):
                money_at_risk += float(qt.get("total") or 0)

        work_done = determine_work_done(job, job_inv, job_apt)
        category, reasoning = None, ""

        if is_demo:
            reasoning = "Demo or test job"
        elif status_lower in ("cancelled", "canceled"):
            reasoning = "Job cancelled"
        elif len(paid_inv) and not len(overdue_inv) and not len(sent_inv):
            reasoning = "Fully paid"
        else:
            # URGENT
            if len(overdue_inv) and work_done == "yes":
                category, reasoning = "Urgent", "Overdue invoice, work completed"
            if category is None and work_done == "yes" and not len(job_inv):
                category, reasoning = "Urgent", "Work done, no invoice"
            if category is None and has_urgency_language(desc + " " + title) and created_days is not None and created_days < 14 \
                    and status_lower not in ("done", "completed", "cancelled", "canceled", "invoiced"):
                category, reasoning = "Urgent", "Emergency language, recent job"
            if category is None and len(sent_q):
                for _, q in sent_q.iterrows():
                    if not q.get("is_approved") and not q.get("is_deleted"):
                        sd = safe_days_since(q.get("issued_timestamp") or q.get("last_quote_event_ts"))
                        if sd is not None and 3 <= sd <= 14:
                            category, reasoning = "Urgent", f"Quote sent {sd}d ago, warm lead"
                            break
            if category is None and len(appr_q) and not len(job_apt) and not len(job_inv):
                category, reasoning = "Urgent", "Approved quote, no appointment/invoice"

            # REQUIRE FOLLOW UP
            if category is None and len(overdue_inv) and work_done in ("no", "unclear") and 14 <= days_inactive <= 60:
                category, reasoning = "Require Follow Up", "Overdue invoice, work status unclear"
            if category is None and len(draft_q) and not len(sent_q):
                category, reasoning = "Require Follow Up", "Draft quote never sent"
            if category is None and len(sent_q):
                for _, q in sent_q.iterrows():
                    if not q.get("is_approved") and not q.get("is_deleted"):
                        sd = safe_days_since(q.get("issued_timestamp") or q.get("last_quote_event_ts"))
                        if sd is not None and 14 < sd <= 60:
                            category, reasoning = "Require Follow Up", f"Quote sent {sd}d ago, no response"
                            break
            if category is None and status_lower == "confirmed" and not len(job_apt):
                category, reasoning = "Require Follow Up", "Confirmed, no appointment"
            if category is None and len(draft_inv) and not len(sent_inv):
                category, reasoning = "Require Follow Up", "Draft invoice not sent"
            if category is None and has_engagement and 14 <= days_inactive <= 60 \
                    and status_lower not in ("done", "completed", "cancelled", "canceled", "invoiced", "paid"):
                category, reasoning = "Require Follow Up", f"Active job quiet {days_inactive}d"
            if category is None and len(job_est) and not len(job_qt):
                category, reasoning = "Require Follow Up", "Estimate exists, no quote created"

            # COLD
            if category is None and days_inactive >= 60 and status_lower not in ("done", "completed", "cancelled", "canceled", "paid"):
                category, reasoning = "Cold", f"Inactive {days_inactive}d"
            if category is None and not has_engagement and created_days is not None and created_days >= 30 \
                    and status_lower not in ("done", "completed", "cancelled", "canceled"):
                category, reasoning = "Cold", f"Ghost lead, {created_days}d old"

            if category is None:
                reasoning = "Job complete" if status_lower in ("done", "completed", "invoiced", "paid") else "Insufficient data"

        if category is None:
            money_at_risk = 0.0

        records.append({
            "account_id": job["account_id"], "job_id": job["job_id"], "job_key": jk,
            "v1_category": category if category else "NoAction", "v1_reasoning": reasoning,
            "v1_money_at_risk": round(money_at_risk, 2), "v1_is_demo": is_demo,
            "v1_has_engagement": has_engagement,
        })

    return pd.DataFrame(records)


def inject_label_noise(df, rng, rate=0.12):
    """Flip a fraction of labels to an ADJACENT class to emulate real-world
    labelling ambiguity. Without this the synthetic labels are a deterministic
    function of the features, so the downstream model would be trivially perfect.
    Seeded and synthetic-only — documented so the example stays honest.
    """
    cats = ["NoAction", "Cold", "Require Follow Up", "Urgent"]
    order = {c: i for i, c in enumerate(cats)}
    flip = rng.random(len(df)) < rate
    new = df["v1_category"].tolist()
    for i in range(len(df)):
        if flip[i]:
            cur = order.get(new[i], 0)
            step = int(rng.choice([-1, 1]))
            new[i] = cats[min(3, max(0, cur + step))]
    df = df.copy()
    df["v1_category"] = new
    return df


def main():
    df_jobs = add_key(load_raw("jobs"), "job_id", "job_id_key")
    df_quotes = add_key(load_raw("quotes"), "quote_id", "quote_key")
    df_invoices = add_key(load_raw("invoices"), "invoice_id", "invoice_key")
    df_appointments = add_key(load_raw("appointments"), "appointment_id", "appointment_key")
    df_estimates = add_key(load_raw("estimates"), "estimate_id", "estimate_key")
    df_fe_actions = load_raw("fe_actions")

    print(f"Jobs {len(df_jobs):,} | Quotes {len(df_quotes):,} | Invoices {len(df_invoices):,} | "
          f"Appts {len(df_appointments):,} | Estimates {len(df_estimates):,} | FE {len(df_fe_actions):,}")

    df_labels = build_labels(df_jobs, df_quotes, df_invoices, df_appointments, df_estimates)
    df_labels = inject_label_noise(df_labels, np.random.default_rng(0), rate=0.12)
    print("\nv1 label distribution:")
    print(df_labels["v1_category"].value_counts().to_string())

    for df, name in [(df_jobs, "jobs"), (df_quotes, "quotes"), (df_invoices, "invoices"),
                     (df_appointments, "appointments"), (df_estimates, "estimates"),
                     (df_fe_actions, "fe_actions"), (df_labels, "v1_labels")]:
        save_interim(df, name)


if __name__ == "__main__":
    main()
