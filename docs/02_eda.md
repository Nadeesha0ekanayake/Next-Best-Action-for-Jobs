# 02 — Exploratory Data Analysis

Profiles the extracted data and surfaces the patterns that inform feature
engineering and modelling.

## What it covers

- Data profiling — shape, types, null rates, distributions
- Job status & lifecycle (status often lags the real entity state)
- Entity coverage — which jobs have quotes, invoices, appointments, estimates
- Financial distributions — quote / invoice amounts
- Time analysis — job age, staleness, event gaps
- Conversion funnel — lead → quote → approved → appointment → invoice → paid
- Front-end engagement — tradie platform-usage patterns
- v1 label analysis — class balance, label vs features

## Key findings

1. **Ghost leads dominate.** A large share of jobs have zero entities (no quote,
   invoice, appointment, or estimate) — the core of the Cold / NoAction classes.
2. **Class imbalance.** "Require Follow Up" is the minority class → the model
   needs balanced class weights.
3. **Staleness is the strongest signal.** `days_since_last_activity` cleanly
   separates categories — Urgent jobs are recent, Cold jobs are old.
4. **Entity presence is binary but powerful.** Simply having a quote or invoice
   distinguishes engaged jobs from ghost leads.
5. **Status lags reality.** Many "New" jobs already have entities — features must
   cross-reference entities, not trust status alone.
6. **Financial amounts span orders of magnitude** → log-transform for the model.
7. **FE engagement varies by account** — action count is a strong account-level
   engagement proxy.
8. **The funnel has a big early drop-off** — most leads never get a quote.

## Implications for feature engineering

| Finding | Features to build |
|---------|-------------------|
| Staleness separates categories | `days_since_last_activity`, `days_since_created`, `staleness_bucket` |
| Entity presence matters | `has_quote`, `has_invoice`, `entity_count`, `lifecycle_stage` |
| Financial signals | `log_quote_total`, `money_at_risk`, `has_overdue_invoice` |
| Status lags | `status_mismatch` (status = New but has entities) |
| FE engagement | `fe_total_action_count`, `fe_action_velocity` |
| Class imbalance | `class_weight="balanced"` in the model |

Charts: `charts/02_entity_coverage.png`, `02_staleness_buckets.png`,
`02_conversion_funnel.png`, `02_fe_engagement.png`, `02_label_balance.png`.

See [`pipeline/02_eda.py`](../pipeline/02_eda.py).
