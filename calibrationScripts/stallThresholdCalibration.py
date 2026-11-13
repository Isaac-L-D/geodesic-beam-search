import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

PROJECT_ROOT = "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))

from engine import _silenceFortranStdout
from ODEClass import ODE, logDetails

HITS_PATH = os.path.join(PROJECT_ROOT, 'geodesicBeamSearch', 'output', 'allUniqueHits.csv')
SAMPLE_SIZE = 30
RUN_SPAN = 30000.0     # long enough to see sustained chaotic behavior, short enough to be cheap
CHUNK_SIZE = 1000.0    # candidate STALL_CHECK_INTERVAL - measuring arc-length growth per chunk


# Per-chunk arc-length growth over the whole run - this is exactly what a live stall-check
# would see: does growth ever stay near-zero for a full chunk (real stall) vs. does it dip low
# but recover (real chaos passing through a slow patch)?
def chunkedGrowth(args):
    idx, W, lle = args
    ode = ODE(W=W, doFitToCube=False)
    rng = np.random.default_rng(idx)
    initial = rng.uniform(-1, 1, size=3).tolist()
    log = logDetails()
    res = ode.integrate(initial_pos=initial, endTime=RUN_SPAN, dt=1.0, log=log)
    if res is None:
        return idx, lle, None
    _, y = res
    pts = y.T
    n = len(pts)
    stepsPerChunk = int(CHUNK_SIZE)  # dt=1.0, so 1 row per time unit
    nChunks = n // stepsPerChunk
    growths = []
    for c in range(nChunks):
        chunkPts = pts[c * stepsPerChunk: (c + 1) * stepsPerChunk + 1]
        growth = float(np.linalg.norm(np.diff(chunkPts, axis=0), axis=1).sum())
        growths.append(growth)
    return idx, lle, growths


def main():
    df = pd.read_csv(HITS_PATH)
    coefCols = [c for c in df.columns if c.startswith('dx_') or c.startswith('dy_') or c.startswith('dz_')]

    rng = np.random.default_rng(0)
    sampleIdx = rng.choice(len(df), size=min(SAMPLE_SIZE, len(df)), replace=False)
    tasks = []
    for idx in sampleIdx:
        W = df.iloc[idx][coefCols].to_numpy(dtype=np.float64).reshape(3, 27)
        W = W / np.linalg.norm(W)
        tasks.append((idx, W, float(df.iloc[idx]['LLEMean'])))

    print(f"measuring per-{int(CHUNK_SIZE)}-time-unit arc-length growth over {int(RUN_SPAN)} "
          f"time units for {len(tasks)} discovered hits...")
    with Pool(processes=8, initializer=_silenceFortranStdout) as pool:
        results = pool.map(chunkedGrowth, tasks)

    knownStalled = {1223, 591, 1876, 972, 1666}  # from investigateFailingHits.py (excl. the blowup case)

    print(f"\n{'idx':>6} {'LLE':>9} {'category':>10} {'minChunk':>10} {'maxChunk':>10} {'lastChunk':>10}")
    stalledMaxima = []
    normalMinima = []
    for idx, lle, growths in results:
        if growths is None:
            print(f"{idx:>6} {lle:>9.4f}   integrate() failed - skipped")
            continue
        category = 'STALLED' if idx in knownStalled else 'normal'
        minG, maxG, lastG = min(growths), max(growths), growths[-1]
        print(f"{idx:>6} {lle:>9.4f} {category:>10} {minG:>10.3f} {maxG:>10.3f} {lastG:>10.3f}")
        if category == 'STALLED':
            stalledMaxima.append(maxG)
        else:
            normalMinima.append(minG)

    print(f"\nSTALLED candidates: max per-chunk growth ever seen = {max(stalledMaxima):.4f} "
          f"(worst case among {len(stalledMaxima)} known-stalled hits)")
    if normalMinima:
        print(f"NORMAL candidates: min per-chunk growth ever seen = {min(normalMinima):.4f} "
              f"(worst dip among {len(normalMinima)} genuinely-progressing hits)")
        print(f"\nSeparation gap: [{max(stalledMaxima):.4f}, {min(normalMinima):.4f}] - "
              f"a threshold anywhere in this range cleanly separates the two groups in this sample.")


if __name__ == '__main__':
    main()
