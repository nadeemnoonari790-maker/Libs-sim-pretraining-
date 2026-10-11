# Data

| Item | Source | How to obtain |
|---|---|---|
| `chemicam_open_database/` (dataset 1) | ChemiCam open database, Takahashi et al. (2020), Deep-Sea Research Part I 158:103232, doi:10.1016/j.dsr.2020.103232 | `git clone https://github.com/ocean-perception/chemicam_open_database data/chemicam_open_database` (or set `CHEMICAM_DIR`) |
| `chemlibs/` (dataset 2) | Mount Holyoke Multi-Parameter LIBS Reference Database, NASA PDS, Dyar MD & Lepore KH, doi:10.17189/b2aj-cz96 | `python tools/fetch_chemlibs.py` (downloads and reduces ChemLIBS Mars-atmosphere 5 mJ spectra, locations 1-2, plus `libs_metadata.xlsx`) |
| `atomic_lines.db` | SQLite line list (wavelength, A_ki, E_i, E_k, g_i, g_k), energy levels and ionisation potentials derived from the NIST Atomic Spectra Database | Included |

The raw spectra are not redistributed here; please obtain them from, and cite, the original sources.

**TODO before release:** document exactly how `atomic_lines.db` was compiled from NIST ASD (version, query date, filters) — see the manuscript limitations.
