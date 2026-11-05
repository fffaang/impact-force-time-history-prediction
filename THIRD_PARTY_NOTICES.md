# Third-party notices

Original code and model weights are licensed in `LICENSE`; dataset rights and source
attribution are addressed separately in `DATA_LICENSE.md`.

The prediction application uses NumPy 2.1.3, SciPy 1.14.1, pandas 2.2.3,
scikit-learn 1.5.2, joblib 1.4.2, and threadpoolctl 3.7.0. Their license notices,
including available numerical-runtime notices, are retained under `third_party/licenses`.
The Windows distribution also contains Python 3.12.14 and a PyInstaller 6.16.0 bootloader.
The bootloader's distribution exception is retained with its license.

Local browser assets are Plotly.js 3.1.1 (MIT) and Lucide 0.468.0 (ISC, with Feather
MIT attribution). Their original license comments and texts are preserved. Additional
license texts document packages from the scientific build environment; their presence
does not imply that every package is a dependency of the CPU prediction application.

Scientific training additionally uses the packages pinned in `requirements-training.txt`.
Those packages are obtained from their official distributions and retain their own licenses.
No third-party article PDF, article illustration, or external font file is distributed.
