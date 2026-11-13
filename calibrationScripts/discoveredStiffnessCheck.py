import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

PROJECT_ROOT = "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))

import attractorSimilarity
from attractorSimilarity import getAttractorPointsWithTimeout
from engine import _silenceFortranStdout

attractorSimilarity.ARC_LENGTH_TIMEOUT_SECONDS = 120
attractorSimilarity.ARC_LENGTH_TIMEOUT_RETRIES = 1

HITS_PATH = os.path.join(PROJECT_ROOT, 'geodesicBeamSearch', 'output', 'allUniqueHits.csv')
SAMPLE_SIZE = 30
# Small, fixed, cheap-if-normal target - not the real 700,000 production target. This is
# purely a relative stiffness probe: how long does a MODEST amount of arc length take for
# each discovered (always unit-norm) candidate, as a proxy for how expensive the real target
# would be at scale.
PROBE_ARC_LENGTH = 10000.0
PROBE_MAX_TIME_CAP = 500000.0


def probe(args):
    idx, W = args
    initial = np.random.default_rng(idx).uniform(-1, 1, size=3).tolist()
    t0 = time.time()
    points = getAttractorPointsWithTimeout(W, initial, PROBE_ARC_LENGTH, 5000, PROBE_MAX_TIME_CAP)
    elapsed = time.time() - t0
    return idx, elapsed, points is not None


def main():
    df = pd.read_csv(HITS_PATH)
    coefCols = [c for c in df.columns if c.startswith('dx_') or c.startswith('dy_') or c.startswith('dz_')]
    print(f"{len(df)} unique hits available, {len(coefCols)} coefficient columns")

    rng = np.random.default_rng(0)
    sampleIdx = rng.choice(len(df), size=min(SAMPLE_SIZE, len(df)), replace=False)
    tasks = []
    for idx in sampleIdx:
        W = df.iloc[idx][coefCols].to_numpy(dtype=np.float64).reshape(3, 27)
        W = W / np.linalg.norm(W)  # should already be ~unit norm, normalize defensively
        tasks.append((idx, W))

    print(f"probing {len(tasks)} random discovered hits, "
          f"arc length={PROBE_ARC_LENGTH}, time cap={PROBE_MAX_TIME_CAP}...")
    with Pool(processes=8, initializer=_silenceFortranStdout) as pool:
        results = pool.map(probe, tasks)

    times = np.array([e for _, e, ok in results if ok])
    nFailed = sum(1 for _, _, ok in results if not ok)
    print(f"\n{len(times)}/{len(tasks)} succeeded within the time cap, {nFailed} failed/rejected")
    if len(times) > 0:
        for p in [0, 10, 25, 50, 75, 90, 100]:
            print(f"  p{p}: {np.percentile(times, p):.2f}s")
        slow = sorted(times)[-5:]
        print(f"  slowest 5: {[f'{t:.1f}s' for t in slow]}")


if __name__ == '__main__':
    main()
