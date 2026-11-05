# Frozen numerical results

`experiment` contains the complete A/B/C results for all three fixed partitions:
individual predictions, all negative curve R² cases, exact physical-time metrics,
source-block bootstrap, three-model-seed stability, and algorithm rankings.
`selection` contains training-internal candidate rankings and oracle diagnostics.
The oracle cases use measured training responses only for diagnosis, never deployment.

`production` contains the final retained C result, later consistency-calibration comparisons,
and five-output SHAP. The tested additional calibrators were rejected in all three partitions;
the deployed function is identity_current_C. Rows labelled final and C in those comparisons
can therefore coincide; they are not additional independent experiments.

Table_5 and Table_6b names are computational table identifiers from the original experiment,
not claims about the numbering in every subsequent manuscript. Figure source filenames
are similarly historical. The original CSV values and actual negative scores are retained.

A is the historical architecture refitted on a nested 131-record training subset;
B is the same architecture freshly fitted on all 158 records; C is the optimized
four-anchor/derived-output structure fitted on the same 158 records. Shared holdouts make
B-A and C-B paired comparisons. Do not compare scores on different split seeds as a
paired algorithm improvement. The three partitions are reported separately, not combined.

Curve R², MAE, RMSE, and NRMSE integrate the piecewise-linear error over the union of
measured and predicted active supports, with one-sided boundaries for nonzero threshold
endpoints. Negative R² is not clipped. Nonnegative force is a separate numerical property.

The primary C median physical-time curve R² is 0.420561 (validation) and 0.515544 (test),
with 4 and 6 negative cases. Not all five-output R² values exceed 0.80; the derived area
validation R² is negative. C-B source-bootstrap intervals do not establish consistent
significant improvement. See the full CSVs rather than interpreting a single aggregate
score as design reliability or unseen-source generalization.

SHAP explains the actual five outputs on log (Fmax, Td, I) and logit (Astar, tp_ratio)
scales. It recomputes derived features after perturbing raw inputs, uses training background
records, and retains additivity checks. Contributions describe model dependence, not causality.
