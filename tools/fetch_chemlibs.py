"""Download a subset of the Mount Holyoke ChemLIBS reference spectra (NASA PDS) and shrink them on the fly.

For every target (rock pellet) it takes the first N_LOC locations; each file's integrations are averaged to one
spectrum and resampled to a common 0.15 nm grid (241-905 nm; gaps between spectrometers = NaN).
Raw CSVs are NOT kept, so disk use stays small. Resumable: just run it again if it stops.

Source: Dyar MD, Lepore KH, Multi-Parameter LIBS Reference Database of Geological Materials Bundle (NASA PDS),
https://doi.org/10.17189/b2aj-cz96 . Please cite the data set when you use it.

Usage (run from the repository root):
    python tools/fetch_chemlibs.py --limit 12     # quick test: 12 files, ~1 minute
    python tools/fetch_chemlibs.py                # full run (~5,500 files, roughly 3.5 GB of traffic, 30-90 min)
Output: data/chemlibs/chemlibs_mars5mj.npz and data/chemlibs/libs_metadata.xlsx (progress file in the same folder).
Needs only numpy.
"""
import argparse, io, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import numpy as np

BASE = "https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-libs_reference_database/data_chemlibs/mars/5mj/"
GRID = np.arange(241.0, 905.01, 0.15, dtype=np.float64)
N_LOC = 2
META_URL = "https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-libs_reference_database/document/libs_metadata.xlsx"
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chemlibs")
UA = {"User-Agent": "libs-sim-pretraining data fetcher (research use)"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.h = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.h.append(v)


def get(url, tries=5, timeout=120):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def parse_csv(raw):
    """Return (wavelength, mean intensity over integration columns). Header rows are skipped robustly."""
    runs, cur = [], []
    for line in raw.decode("utf-8", "ignore").splitlines():
        parts = [p.strip() for p in re.split(r"[,\t;]", line) if p.strip() != ""]
        v = None
        if len(parts) >= 2:
            try:
                v = [float(p) for p in parts]
            except ValueError:
                v = None
        if v is None:
            if cur: runs.append(cur); cur = []
        else:
            cur.append(v)
    if cur: runs.append(cur)
    run = max(runs, key=len) if runs else []   # data block = longest contiguous numeric block (header rows are short)
    wl = [r[0] for r in run]; ys = [r[1:] for r in run]
    if len(wl) < 1000:
        raise ValueError(f"too few numeric rows: {len(wl)}")
    w = np.array(wl)
    m = min(len(y) for y in ys)
    Y = np.array([y[:m] for y in ys], dtype=np.float64)
    ok = np.isfinite(Y).all(axis=1) & np.isfinite(w)
    w, Y = w[ok], Y[ok]
    o = np.argsort(w); w, Y = w[o], Y[o]
    return w, Y.mean(axis=1), Y.shape[1]


def to_grid(w, y):
    out = np.interp(GRID, w, y, left=np.nan, right=np.nan)
    # blank the gap between spectrometers (np.interp would bridge it linearly)
    big = np.where(np.diff(w) > 3.0)[0]
    for i in big:
        out[(GRID > w[i]) & (GRID < w[i + 1])] = np.nan
    return out.astype(np.float32)


def work(name):
    raw = get(BASE + name)
    w, y, ncol = parse_csv(raw)
    return name, to_grid(w, y), float(w.min()), float(w.max()), ncol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    print("fetching directory listing ...", flush=True)
    html = get(BASE).decode("utf-8", "ignore")
    p = Links(); p.feed(html)
    files = sorted({h.split("/")[-1] for h in p.h if h.lower().endswith("_spect.csv")})
    print("csv files in folder:", len(files), flush=True)
    if not files:
        print("No csv links found. First 500 chars of listing:\n", html[:500]); sys.exit(1)
    # keep first N_LOC locations per target: filename ...ccs_mars_t01l01_<target>_spect.csv
    keep = []
    for f in files:
        m = re.search(r"_t(\d+)l(\d+)_(.+)_spect\.csv$", f, re.I)
        if m and int(m.group(2)) <= N_LOC:
            keep.append(f)
    print("selected (locations 1..%d):" % N_LOC, len(keep), flush=True)
    if a.limit:
        keep = keep[:a.limit]

    os.makedirs(OUTDIR, exist_ok=True)
    meta = os.path.join(OUTDIR, "libs_metadata.xlsx")
    if not os.path.exists(meta):
        open(meta, "wb").write(get(META_URL)); print("downloaded", meta, flush=True)
    prog = os.path.join(OUTDIR, "chemlibs_progress.npz")
    done = {}
    if os.path.exists(prog) and not a.limit:
        d = np.load(prog, allow_pickle=False)
        for n, s in zip(d["names"], d["spec"]):
            done[str(n)] = s
        print("resuming, already done:", len(done), flush=True)
    todo = [f for f in keep if f not in done]
    t0 = time.time(); nfail = 0; last_save = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(work, f): f for f in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            f = futs[fu]
            try:
                n, s, lo, hi, ncol = fu.result()
                done[n] = s
                if a.limit:
                    print(f"OK {n}: wl {lo:.1f}-{hi:.1f} nm, {ncol} integration cols, finite grid pts {int(np.isfinite(s).sum())}")
            except Exception as e:
                nfail += 1; print("FAIL", f, repr(e)[:150], flush=True)
            if i % 50 == 0:
                print(f"{i}/{len(todo)} done, {time.time() - t0:.0f}s, fails {nfail}", flush=True)
            if not a.limit and time.time() - last_save > 120:
                names = np.array(sorted(done)); np.savez(prog, names=names, spec=np.stack([done[k] for k in names]))
                last_save = time.time()
    if a.limit:
        print("TEST FINISHED, fails:", nfail); return
    names = np.array(sorted(done))
    spec = np.stack([done[k] for k in names])
    np.savez(prog, names=names, spec=spec)
    np.savez_compressed(os.path.join(OUTDIR, "chemlibs_mars5mj.npz"), names=names, spec=spec, grid=GRID.astype(np.float32))
    print("DONE. spectra:", len(names), "fails:", nfail, "->", os.path.join(OUTDIR, "chemlibs_mars5mj.npz"), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
