"""Stage 00 — Generate a SYNTHETIC source dataset.

Nothing here comes from any real system. It fabricates the six source tables the
NBA pipeline consumes — jobs, quotes, invoices, appointments, estimates, and
front-end (FE) action events — with the same columns and realistic
relationships, so the downstream stages (01-05) run end-to-end on fake data.

Design: each synthetic tradie account gets an "engagement" level; each job gets a
"progress" latent (how far down the lifecycle it went). Entity presence, statuses,
amounts, and staleness all derive from those, so the rule-based labels and the
model behave like they would on real data.
"""

import numpy as np
import pandas as pd

from common import RAW, TODAY, days_ago_str

SEED = 42
N_ACCOUNTS = 120

STATUSES = ["New", "Quoted", "Confirmed", "Scheduled", "Work Completed", "Invoiced", "Cancelled"]
LEAD_SOURCES = ["marketplace", "direct", "referral", "repeat", "external"]
FE_CATEGORIES = [
    "Workflow Management", "Quote Sent", "Invoice Sent", "Estimate Sent",
    "Appointment Created", "New Job Created", "Note Added", "Payment Receipt Sent",
]
EMERGENCY_SNIPPETS = ["burst pipe leak", "urgent no hot water", "emergency flooding",
                      "broken safety switch", "asap dangerous wiring"]
NORMAL_SNIPPETS = ["bathroom renovation", "deck build", "fence repair", "kitchen tiling",
                   "garden landscaping", "roof gutter clean", "aircon install"]


def main():
    rng = np.random.default_rng(SEED)

    # Synthetic account ids (NOT real) — engagement drives everything downstream.
    account_ids = [f"ACC{n:05d}" for n in range(1, N_ACCOUNTS + 1)]
    engagement = dict(zip(account_ids, rng.beta(1.8, 2.5, N_ACCOUNTS)))

    jobs, quotes, invoices, appts, estimates, fe = [], [], [], [], [], []
    job_counter = quote_counter = inv_counter = apt_counter = est_counter = 0

    for acc in account_ids:
        eng = engagement[acc]
        n_jobs = 1 + rng.poisson(2 + eng * 8)
        for _ in range(n_jobs):
            job_counter += 1
            job_id = job_counter
            created_days = float(rng.integers(1, 200))
            # progress: how far the job advanced (higher engagement -> further)
            progress = np.clip(rng.beta(1.5, 3.0) + 0.3 * eng, 0, 1)
            emergency = rng.random() < 0.05
            desc = rng.choice(EMERGENCY_SNIPPETS) if emergency else rng.choice(NORMAL_SNIPPETS)

            # staleness: recent if in progress, old if abandoned
            last_activity_days = float(rng.integers(0, int(created_days) + 1))
            if progress < 0.25:
                last_activity_days = float(rng.integers(30, int(created_days) + 30))

            has_estimate = rng.random() < (0.15 + 0.3 * progress)
            has_quote = rng.random() < (0.2 + 0.6 * progress)
            has_invoice = has_quote and rng.random() < (0.1 + 0.7 * progress)
            has_appt = has_quote and rng.random() < (0.3 * progress)
            cancelled = rng.random() < 0.06

            status = rng.choice(STATUSES, p=[.28, .2, .12, .08, .12, .12, .08]) if not cancelled else "Cancelled"

            jobs.append({
                "entity_id": job_id, "account_id": acc, "job_id": job_id,
                "job_title": desc.title(), "job_description": desc,
                "job_created_timestamp": days_ago_str(created_days),
                "lead_source": rng.choice(LEAD_SOURCES), "lead_id": 900000 + job_id,
                "status": status, "customer_id": (700000 + job_id) if rng.random() < 0.7 else None,
                "customer_source": "app",
                "last_job_event_ts": days_ago_str(last_activity_days),
                "last_job_event_type": "JobUpdated",
            })

            base_amt = float(rng.gamma(2.0, 400) + 100)

            if has_estimate:
                est_counter += 1
                estimates.append({
                    "estimate_id": est_counter, "account_id": acc, "job_id": job_id,
                    "estimate_job_description": desc, "cost": round(base_amt * 0.9, 2),
                    "duration": int(rng.integers(1, 10)),
                    "estimate_created_ts": days_ago_str(created_days - 1),
                    "estimate_updated_ts": days_ago_str(last_activity_days),
                    "job_is_demo": False,
                    "last_estimate_event_ts": days_ago_str(last_activity_days),
                    "last_estimate_event_type": "EstimateUpdated",
                    "entity_revision": int(rng.integers(1, 4)),
                })

            if has_quote:
                quote_counter += 1
                approved = rng.random() < (0.2 + 0.5 * progress)
                qstatus = rng.choice(["Draft", "Sent", "Approved", "Rejected"],
                                     p=[.25, .45, .2, .1])
                if approved:
                    qstatus = "Approved"
                sent_days = last_activity_days + rng.integers(0, 20)
                quotes.append({
                    "quote_id": quote_counter, "account_id": acc, "job_id": job_id,
                    "issued_timestamp": days_ago_str(sent_days) if qstatus != "Draft" else None,
                    "valid_until_timestamp": days_ago_str(max(0, sent_days - 30)),
                    "is_approved": bool(approved), "is_deleted": False,
                    "total": round(base_amt, 2), "quote_status": qstatus, "is_finalized": True,
                    "job_is_demo": False,
                    "last_quote_event_ts": days_ago_str(last_activity_days),
                    "entity_revision": int(rng.integers(1, 4)),
                })

            if has_invoice:
                inv_counter += 1
                paid = rng.random() < (0.3 + 0.5 * progress)
                overdue = (not paid) and rng.random() < 0.4
                istatus = "Paid" if paid else ("Overdue" if overdue else rng.choice(["Draft", "Sent"], p=[.3, .7]))
                invoices.append({
                    "invoice_id": inv_counter, "account_id": acc, "job_id": job_id,
                    "issued_timestamp": days_ago_str(last_activity_days + 5) if istatus != "Draft" else None,
                    "paid_timestamp": days_ago_str(last_activity_days) if paid else None,
                    "due_timestamp": days_ago_str(max(0, last_activity_days - 14)),
                    "total": round(base_amt * 1.05, 2), "invoice_status": istatus,
                    "is_finalized": True, "is_overdue": bool(overdue), "invoice_type": "standard",
                    "last_invoice_event_ts": days_ago_str(last_activity_days),
                })

            if has_appt:
                apt_counter += 1
                future = rng.random() < 0.3
                start_days = -float(rng.integers(1, 14)) if future else float(rng.integers(1, 60))
                appts.append({
                    "appointment_id": apt_counter, "account_id": acc, "job_id": job_id,
                    "start_date": days_ago_str(start_days), "end_date": days_ago_str(start_days),
                    "created_date": days_ago_str(created_days - 2), "job_is_demo": False,
                    "last_appt_event_ts": days_ago_str(last_activity_days),
                    "last_appt_event_type": "Scheduled" if future else "Completed",
                    "entity_revision": int(rng.integers(1, 3)),
                })

        # FE action events for the account (account-level engagement signal)
        n_fe = rng.poisson(eng * 40)
        for _ in range(n_fe):
            act_days = float(rng.integers(0, 120))
            fe.append({
                "account_id": acc,
                "action_timestamp": days_ago_str(act_days),
                "action_date": days_ago_str(act_days),
                "event_name": "interaction",
                "action_category": rng.choice(FE_CATEGORIES),
            })

    RAW.mkdir(parents=True, exist_ok=True)
    tables = {
        "jobs": jobs, "quotes": quotes, "invoices": invoices,
        "appointments": appts, "estimates": estimates, "fe_actions": fe,
    }
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        df.to_csv(RAW / f"{name}.csv", index=False)
        print(f"  wrote raw/{name}.csv  ({len(df):,} rows, {df.shape[1]} cols)")

    print(f"\nSynthetic dataset generated: {N_ACCOUNTS} accounts, {job_counter:,} jobs")


if __name__ == "__main__":
    main()
