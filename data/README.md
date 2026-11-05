# Impact force–time history dataset for reinforced concrete members

## Two distinct history representations

`curves.npz`, `force_histories_long.csv`, and `force_histories/<curve_id>.csv` contain
the identical 38,403 model-ready points for all 199 archive records. NPZ offsets use
half-open indexing: history i occupies `[offsets[i], offsets[i+1])`. Each per-history
CSV includes curve_id, point_index, time_ms, and force_kN. No new smoothing, clipping,
upsampling, or digitization was performed for this release.

`present10_origin_long.csv` contains the complete 500-point plotting worksheet for
each of the ten present-study histories. Its time column is **time_s**, in seconds;
the force column is force_kN. Sheet and column names preserve the Origin mapping.
These values are not claimed to be raw sensor acquisition signals. The processed
contact windows and five descriptors are separately documented in `audit`.

The original own Origin project is in `source/FIG3.opju`. S600 uses the confirmed
4.21 m/s input; the 4.02 graph-legend alias is preserved in the evidence. The earlier
four reference curves are not additional dataset members or an external validation set.

## Records and partitions

`archive_199.csv` preserves the frozen inputs, targets, class labels, and missing values.
The `split` column in this archive is a preparation placeholder. **Always join on curve_id
with `splits/<seed>.csv` for the actual final partition.** Do not use legacy_split as
the final 8:1:1 assignment. Eligible records total 198; B3-7-P remains quarantined.

Each seed 20261102, 20261103, and 20261104 assigns 158 training, 20 validation, and
20 test records. The nested_A field defines the 131-record A control. Known associated
specimens are kept together; these are not publication-isolated partitions. Inner folds
refer only to training records. The three holdout sets overlap and must not be pooled as
60 independent tests.

Roles: primary = 169 ordinary-RC experimental records including the own ten;
numerical_aux = 18 numerical records; shape_aux = 11 special-material/member records;
quarantine = one excluded record. The inherited auxiliary names are classification labels:
eligible training records were jointly supervised in this experiment, not restricted to
pretraining. `record_training_uses.csv` records the actual use of each record.

## Inputs, targets, and units

The ordered 17 inputs are v, m, b, h, H0, d, c, N, fc_reported, fyl, dl, Asl, fyt,
dt, Asv, s, corrosion_pct. See `input_dictionary.csv`. Source and specimen IDs, role
labels, measured anchors, and force ordinates are not deployment inputs.

Dimensions are mm; mass kg; velocity m/s; axial force kN; strengths MPa; steel areas
mm²; corrosion percent. The metadata column fc_reported retains the source's cube,
cylinder, or unspecified strength definition, not a universal converted cylinder strength.
The predictor symbol f′c is a display convention; consult strength_type before physical reuse.

Fmax is kN, Td is ms, I is kN·ms, Astar = I/(Fmax Td), and tp_ratio is dimensionless.
One kN·ms equals one N·s. Time zero is the modeling contact-window origin.
Missing optional values are retained, not replaced by arbitrary zeros. The stored
training-only imputer is part of each trained component.

## Sources and quality

`source_mapping.csv` is keyed by curve_id and carries the current frozen class and
source identity. Source labels and reference numbers refer to the archived manuscript
bibliography, not a new numbering system. `source_references.csv` supplies matching
bibliographic identities. A source without a confirmed DOI is left without one.
`audit/inherited_data_corrections.csv` records values, evidence locations, and reasons.
Historical split labels in inherited audit tables are not final split assignments.
Local PDF paths have been replaced with basenames; PDF hashes and evidence pages remain.

The source evidence manifest contains inherited DOI/source conflicts; match the
source mapping, actual title, and PDF hash rather than inferring a source from a filename.
Neither source identity checks nor release checks establish complete pointwise verification
of the 189 inherited curves. Unresolved strength definitions, contact/support metadata,
curve identity, and repeated damage remain visible in the original audit evidence.

Own data and original curation: CC BY 4.0, fffaang (2026). Reused-source rights are
retained; see `../DATA_LICENSE.md`. Article PDFs are not included.
