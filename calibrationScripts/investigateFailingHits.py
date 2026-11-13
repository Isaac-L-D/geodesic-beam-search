import os
import sys
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
from ODEClass import ODE, logDetails

attractorSimilarity.ARC_LENGTH_TIMEOUT_SECONDS = 120
attractorSimilarity.ARC_LENGTH_TIMEOUT_RETRIES = 1

HITS_PATH = os.path.join(PROJECT_ROOT, 'geodesicBeamSearch', 'output', 'allUniqueHits.csv')
SAMPLE_SIZE = 30
PROBE_ARC_LENGTH = 10000.0
PROBE_MAX_TIME_CAP = 500000.0


def probe(args):
    idx, W = args
    initial = np.random.default_rng(idx).uniform(-1, 1, size=3).tolist()
    points = getAttractorPointsWithTimeout(W, initial, PROBE_ARC_LENGTH, 5000, PROBE_MAX_TIME_CAP)
    return idx, points is not None


# Direct diagnostic run (not arc-length gated): integrate for a fixed, generous MODEL time and
# measure how much arc length actually accumulates, plus whether the trajectory blows up,
# settles (near-zero late-window speed), or genuinely progresses but just slowly.
def diagnose(args):
    idx, W, lle = args
    ode = ODE(W=W, doFitToCube=False)
    rng = np.random.default_rng(idx)
    initial = rng.uniform(-1, 1, size=3).tolist()
    log = logDetails()
    res = ode.integrate(initial_pos=initial, endTime=500000.0, dt=1.0, log=log)
    if res is None:
        return idx, lle, 'integrate() failed/blew up', None, None
    _, y = res
    pts = y.T
    totalArc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    # speed in the last 10% of the run vs the first 10% (post-warmup) - settling shows up as
    # late-window speed collapsing toward zero relative to early-window speed.
    n = len(pts)
    earlySpeed = np.linalg.norm(np.diff(pts[:n // 10], axis=0), axis=1).mean()
    lateSpeed = np.linalg.norm(np.diff(pts[-n // 10:], axis=0), axis=1).mean()
    verdict = 'settled (late speed collapsed)' if lateSpeed < 0.05 * earlySpeed else 'genuinely slow throughout'
    return idx, lle, verdict, totalArc, (earlySpeed, lateSpeed)


def main():
    df = pd.read_csv(HITS_PATH)
    coefCols = [c for c in df.columns if c.startswith('dx_') or c.startswith('dy_') or c.startswith('dz_')]

    rng = np.random.default_rng(0)
    sampleIdx = rng.choice(len(df), size=min(SAMPLE_SIZE, len(df)), replace=False)
    tasks = []
    for idx in sampleIdx:
        W = df.iloc[idx][coefCols].to_numpy(dtype=np.float64).reshape(3, 27)
        W = W / np.linalg.norm(W)
        tasks.append((idx, W))

    print("re-running the same probe to identify which indices failed...")
    with Pool(processes=8, initializer=_silenceFortranStdout) as pool:
        results = pool.map(probe, tasks)

    failedIdx = [idx for idx, ok in results if not ok]
    print(f"failed indices: {list(failedIdx)}")

    wByIdx = {idx: W for idx, W in tasks}
    diagTasks = [(idx, wByIdx[idx], float(df.iloc[idx]['LLEMean'])) for idx in failedIdx]

    print(f"\ndiagnosing {len(diagTasks)} failing hits directly (fixed 500,000 model-time integration)...")
    with Pool(processes=8, initializer=_silenceFortranStdout) as pool:
        diagResults = pool.map(diagnose, diagTasks)

    print(f"\n{'idx':>6} {'LLE':>10} {'verdict':>28} {'totalArc@500k':>15}")
    for idx, lle, verdict, totalArc, speeds in diagResults:
        arcStr = f"{totalArc:.1f}" if totalArc is not None else "N/A"
        print(f"{idx:>6} {lle:>10.4f} {verdict:>28} {arcStr:>15}")
        if speeds is not None:
            print(f"       early-window mean speed={speeds[0]:.6f}  late-window mean speed={speeds[1]:.6f}")


if __name__ == '__main__':
    main()
