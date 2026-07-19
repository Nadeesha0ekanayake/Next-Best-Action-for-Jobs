# 05 — Output & Deliverables

Takes the scored jobs from stage 04, assigns the single most impactful **next
action**, calculates money at risk, and produces the final deliverables.

## Next-action mapping

The action is chosen from the predicted category plus the job's entity state, in
priority order:

| Scenario | Action | Entity |
|----------|--------|--------|
| Overdue invoice | `ChasePayment` | Invoice |
| Work done, no invoice | `CreateInvoice` | Invoice |
| Draft invoice, not sent | `SendInvoice` | Invoice |
| Approved quote, no next step | `BookAppointment` | Appointment |
| Confirmed job, no appointment | `BookAppointment` | Appointment |
| Draft quote, not sent | `SendQuote` | Quote |
| Sent quote awaiting response | `FollowUpQuote` | Quote |
| Estimate exists, no quote | `CreateQuote` | Quote |
| No quote (Urgent / Follow Up) | `CreateQuote` | Quote |
| General review needed | `ReviewJob` | Job |
| NoAction category | *(none)* | — |

## Deliverables

- **`final_output`** — every job with predicted category, confidence,
  probabilities, next action, money at risk, and context features
- **`account_summary`** — per-account rollup (urgent/follow-up/cold counts, total
  money at risk, average confidence), sorted by money at risk

Charts: `charts/05_category_distribution.png`, `05_v1_vs_v2.png`,
`05_next_actions.png`, `05_money_at_risk.png`.

## Takeaways

1. **Staleness is king** — `days_since_last_activity` is the strongest predictor.
2. **Entity presence matters** — ghost leads dominate the Cold / NoAction classes.
3. **Financial signals amplify urgency** — overdue invoices and high money-at-risk
   push jobs toward Urgent.
4. **FE engagement is a differentiator** — active tradies have fewer cold jobs.
5. **The model generalises beyond rules** — it learns interactions rigid
   thresholds miss; the disagreement cases are where it adds value.

### How it could be used
- Surface `next_action_type` as contextual prompts in the UI ("Send this quote").
- Sort Urgent jobs by `money_at_risk` to show where the biggest revenue is at stake.
- Nudge ghost leads (zero entities, 30+ days) with a "Create Quote" prompt.

See [`pipeline/05_output.py`](../pipeline/05_output.py).
