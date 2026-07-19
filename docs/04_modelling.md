# 04 — Modelling (Multinomial Logistic Regression)

Trains a 4-class model on the feature matrix from stage 03, evaluates it,
interprets coefficients, and scores every job with category probabilities.

## Approach

| Choice | Value | Why |
|--------|-------|-----|
| Model | Multinomial logistic regression | Interpretable, proper multi-class probabilities |
| Target | 4-class ordinal (NoAction=0, Cold=1, Follow Up=2, Urgent=3) | Matches the v1 categories |
| Split | 70 / 30, stratified | Preserve class balance across train/test |
| Class weights | `balanced` | Handle the "Require Follow Up" minority class |
| Regularisation | L2, `C=1.0` | Guard against overfitting on correlated features |
| Scaling | `StandardScaler`, fit on train only | Logistic regression needs standardised inputs |

## Evaluation

- **Classification report** — per-class precision / recall / F1
- **Confusion matrix** — counts and row-normalised (recall) — `charts/04_confusion_matrix.png`
- **ROC curves** — one-vs-rest per class + macro AUC — `charts/04_roc_curves.png`
- **Coefficients** — which features drive each class — `charts/04_coefficients_by_class.png`

On the synthetic dataset the model reaches ~0.89 accuracy and ~0.96 macro AUC.
(Real-world performance depends on the data; the point of v2 is that it learns
feature *interactions* the rigid v1 rules miss.)

## Scoring

Every job is scored with a predicted category, a probability per class, and a
confidence (max probability). The **v1-vs-v2 disagreements** are the most
interesting cases — where the model catches nuance the rules cannot.

See [`pipeline/04_modelling.py`](../pipeline/04_modelling.py).
