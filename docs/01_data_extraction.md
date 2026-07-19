# 01 — Data Extraction + v1 Labels

Pulls and consolidates all data the pipeline needs, then generates the v1
rule-based category labels that become the model's training target.

## Data sources

In production this reads the latest state of each entity from the warehouse.
Table names below are **generic placeholders** (see `pipeline/common.py`); the
runnable version reads the synthetic CSVs from stage 00.

| # | Layer | Table (placeholder) | Captures |
|---|-------|---------------------|----------|
| 1 | Silver (back-end) | `your_catalog.silver.jobs_changed` | Job entity state |
| 2 | Silver (back-end) | `your_catalog.silver.invoices_changed` | Invoice entity state |
| 3 | Silver (back-end) | `your_catalog.silver.quotes_changed_events` | Quote revisions |
| 4 | Silver (back-end) | `your_catalog.silver.appointments_changed_events` | Appointment revisions |
| 5 | Silver (back-end) | `your_catalog.silver.estimates_changed_events` | Estimate revisions |
| 6 | Gold (front-end) | `your_catalog.gold.accounts_and_events_modular` | Tradie UI actions |

**Critical rule:** entity IDs are only unique **within an account**, so every key
is composite: `(account_id, entity_id)`.

## v1 rule-based categorisation

Each job is categorised by interpretable heuristics. The rules are checked in
priority order (Urgent → Require Follow Up → Cold → NoAction):

| Category | Example triggers |
|----------|------------------|
| **Urgent** | Overdue invoice with work done · work done but no invoice · warm quote sent 3–14d ago · approved quote with no next step · emergency language on a recent job |
| **Require Follow Up** | Quote sent 14–60d ago, no response · draft quote never sent · confirmed job with no appointment · active job gone quiet 14–60d · estimate with no quote |
| **Cold** | Inactive 60+ days · ghost lead 30+ days old with no engagement · quote unanswered 60+ days |
| **NoAction** | Fully paid · cancelled · demo/test · complete |

A **money-at-risk** figure (unpaid sent invoices + unapproved sent quotes) is
attached to every actionable job.

> The example injects a small amount of documented, seeded label noise so the
> downstream model isn't trivially perfect on synthetic data.

See [`pipeline/01_data_extraction.py`](../pipeline/01_data_extraction.py).
