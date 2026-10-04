# Model decisions

## Target and preparation

The target is delivered concrete volume summed for a plant, date, and assignment
hour. Numeric ticket predictors are averaged for that same key, with a one-to-one
merge to the target. Ticket volume, ticket count, and average ticket volume are
excluded from predictors. Observed active hours form the evaluation population;
empty hours are not added.

`preparation.py` includes the original pipeline's numeric preparation rules: median filling,
linear-interpolated percentile bounds, clipping, median/IQR scaling, a centered
fallback for zero IQR, six-decimal rounding, and cyclical encodings. Statistics
come from training rows. Non-winsorized continuous fields also clip to training
minima/maxima, as in the original pipeline. Missingness indicators remain predictors.

The runnable experiment starts from the already prepared plant-hour
replica. It does not apply ticket preprocessing again to its robust-scaled values.
No raw confidential tickets, lookup tables, coordinates, capacities, product-code
parsers, or fitted preprocessing statistics are included. The generic numeric
utility and aggregation function can operate on enriched masked tickets; they do
not reconstruct the private joins or establish end-to-end raw-data equivalence.
The cycle-pressure feature is already present in the replica; its original
capacity-derived calculation is not rerun here.

## Chronology and feature representation

The CSV's train, validation, and test labels and replica dates are retained.
Rows are ordered by date, assignment hour, and masked plant label before fitting,
matching the original pipeline's date/hour/plant order. Median filling uses only
the training period, including before cross-validation. No split is shuffled.

The replica uses neutral English feature names, masked plant labels and product
families, and its own calendar. It has a different feature representation from
the original: extra explicit calendar inputs and plant indicators are retained;
private plant codes and the original year indicator are absent. The feature list
is saved in `results/replica_metrics.json`. Predictions therefore reproduce this
replica experiment, not the original fitted estimator.

The aggregation helper accepts explicit assignment hours instead of recovering
hours by truncating an angle from sine/cosine ticket features. That avoids
roundoff-based hour recovery in the original pipeline. Plant labels are explicit rather
than recovered from private one-hot codes. The target sum and predictor means
are unchanged. The prepared replica needs neither recovery step.

## Model comparison and selection

The model grid preserves the modeling source:

- Historical median: a global training-target median, not a per-plant profile.
- Ridge: alpha in 0.1, 1, 10, 50.
- Elastic Net: alpha in 0.001, 0.01, 0.1, 1; mixing ratio in 0.2, 0.5, 0.8;
  maximum iterations 20000.
- Random Forest: 300 trees; depth in 8, 12, unrestricted; minimum leaf size in
  3, 8; square-root feature sampling; seed 42.

Validation MAE chooses the best run within each family. The original pipeline fixes Random
Forest as the final family. This package preserves that decision and fails if
Random Forest does not also have the lowest validation MAE. It refits the chosen
forest on training plus validation rows and evaluates that estimator on test.
Only that family receives a held-out score. 

Cross-validation uses five expanding `TimeSeriesSplit` folds within training.
MAE is averaged across folds, with sample standard deviation (`ddof=1`). Folds
split ordered rows and may divide a date. Preprocessing is shared across folds,
so this is not a fold-local preprocessing evaluation.

MAE, RMSE, and sMAPE retain the original pipeline formulas. sMAPE divides absolute error by
half the sum of absolute actual and predicted values, then averages and multiplies
by 100. A zero/zero pair contributes zero. Masked division implements that same
case without a divide-by-zero warning. Predictions are not clipped.

## Outputs and limits of fidelity

The CLI writes metrics, the complete family comparison, test predictions, and a
comparison figure. Estimators are retrained in memory and never serialized.
Runtime measurements and machine-specific metadata are omitted. The original pipeline's
notebook, feature-importance plot, scatter plot, error-by-hour plot, inference
schema, and model binaries are not generated. This keeps the runnable comparison
small and independent of application inference.

The replica preserves the checked best/worst validation ranking, Random Forest's
advantage over linear models, and held-out MAE and sMAPE within 30% of the real
model's results. The full validation family order also matches the real
model's order in this run. Those checks do not establish equality of coefficients,
feature distributions, per-plant errors, individual predictions, RMSE, uncertainty
coverage, operational performance, or privacy guarantees.

The public experiment uses Python 3.14, the only interpreter verified locally,
and pins its direct dependencies. It does not claim compatibility with the
original experiment's interpreter or serialized artifacts. Flask, client-demand
modeling, dashboards, deployment, and administrative retraining are outside scope.
