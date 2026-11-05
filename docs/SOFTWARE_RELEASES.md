# Software releases

## v1.0.1 / ImpactCurve 1.1.1

The Windows application is rebuilt in an isolated standard CPython 3.12.14 environment.
The build checks upstream runtime hashes, fixes all wheel versions and hashes, and
rejects unexpected installed packages. Dependency notices and original signatures remain
intact. The local interface, input schema, and scientific prediction code are unchanged.

Verification covers the executable's embedded Python modules, native dependency files,
package contents, the complete 17-input example, HTTP input boundaries, batch and CSV
exports, and all 198 primary predictions, including all 40 validation/test histories.
Numerical tolerance is relative 1e-9 and absolute 1e-8. The saved model, original data,
fixed partitions, results, and SHAP values retain their earlier hashes.

The application is unsigned. Upstream dependency signatures are preserved; one dependency
certificate chain could not be fully validated on the build host, but its bytes match
the checksum-verified official wheel. Use the published SHA-256 checksums and your
organization's software security policy.

Dataset version 1.0.0 and its three large control-model archives remain available in the
original dataset release. The superseded Windows asset is withdrawn only after the new
download has passed integrity and executable tests. Existing tags and commit history
are not rewritten.
