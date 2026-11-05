# Prediction interface and outputs

Run `python app/app.py` from the repository root after installing
`requirements-inference.txt`. The application binds only to 127.0.0.1 and checks
Host, Origin, request size, and a local session token. It is a local research
application, not an authenticated multi-user internet service. Do not expose its port publicly.

Enter v (m/s), m (kg), b and h (mm), and fc_reported (MPa). Other inputs may remain
missing: H0, effective depth d, cover c, axial force N, longitudinal yield fyl,
bar diameter dl, total longitudinal area Asl, stirrup yield fyt, stirrup diameter dt,
effective stirrup area Asv, spacing s, and corrosion_pct. See `data/input_dictionary.csv`
for all units. Missing inputs are disclosed when training-fitted imputation is used.
The tool does not infer clear span from total length or convert cube to cylinder strength.

The default complete example, C2.5-L-2167-4.71, is an archived training case from
Sun et al. (2024), DOI 10.1016/j.engstruct.2024.118416. Its effective depth and nominal
steel areas use documented geometry. The examples are not independent validation cases.

The single-case CLI accepts a JSON object with `name` and `inputs`. Batch CSV requires
case_id and the 17 exact input column names, with 1–100 unique cases. If any row is
invalid, the entire batch is rejected; no cases are silently dropped.

Exports contain final five-output descriptors, native piecewise-linear force-history knots,
curve-extracted descriptors, consistency-projection diagnostics, input notices, and model
SHA-256. Time is ms; force kN; impulse kN·ms. Curve CSV additionally includes seconds and
normalized coordinates. The explicit peak knot is retained. Source and measured-response
fields are not accepted as user inputs.

The frozen production path uses four independent ensembles and derives Astar. The independently
fitted Astar benchmark retained in the model is not substituted for the derived output.
No fitting is performed by the app. The NumPy float64 forward adapter is kept separate from
the PyTorch training implementation and verified against stored outputs. Both use the same
unchanged saved estimator bytes. Only load the trusted distributed model; joblib/pickle
files can execute code and are unsafe from untrusted sources.

Windows: download the portable ZIP from the [v1.0.1 Release](https://github.com/fffaang/impact-force-time-history-prediction/releases/tag/v1.0.1), retain the adjacent `_internal`
folder, and launch `Start_Predictor.cmd`. The executable is unsigned. Closing the browser
does not stop the local service; use its stop control. Check the asset and file checksums.

To rebuild on Windows x64, use the exact standard CPython 3.12.14 archive and SHA-256 in
`provenance/windows_runtime.json`. Verify the archive against its upstream checksum file
before extracting it. Create a fresh virtual environment from that interpreter, without
system packages, unset `PYTHONPATH` and `PYTHONHOME`, and set `PYTHONNOUSERSITE=1`.

```shell
python -m pip install --require-hashes -r requirements-build-windows.lock
python scripts/build_windows.py --output output/windows_build
```

The builder verifies the runtime files, exact dependency versions, and absence of extra
packages before staging the unchanged model. It retains third-party notices and records
runtime, dependency-lock, and model hashes. The v1.0.1 application was built through this
entry point and independently checked against all 198 primary predictions and 40 heldout
histories. Build timestamps can differ, so a later build is not claimed to have an identical
EXE hash. Dataset version 1.0.0 and its separate data/model assets are unchanged.
