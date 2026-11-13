import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool

PROJECT_ROOT = "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))

from engine import _silenceFortranStdout
from ODEClass import ODE

HITS_PATH = os.path.join(PROJECT_ROOT, 'geodesicBeamSearch', 'output', 'allUniqueHits.csv')
SAMPLE_SIZE = 30
RUN_SPAN = 30000.0
TAU = 500.0
WARMUP = 3 * TAU  # skip the initial charge-up period - m(0) is set to the initial instantaneous
# speed to minimize this, but early on m hasn't yet had time to reflect sustained behavior,
# so a low reading here isn't evidence of stalling, just startup transient.

known = {1223, 591, 1876, 972, 1666}  # settled; 1632 (blew up) excluded, same as before


def emaTrace(args):
    idx, W, lle = args
    ode = ODE(W=W, doFitToCube=False)
    rng = np.random.default_rng(idx)
    initial = rng.uniform(-1, 1, size=3).tolist()
    initialSpeed = np.linalg.norm(ode.eval(0, np.array(initial)))

    def rhs(t, s):
        v = ode.eval(t, s[:3])
        speed = np.linalg.norm(v)
        dm = (speed - s[4]) / TAU
        return np.array([v[0], v[1], v[2], speed, dm])

    from scipy.integrate import solve_ivp
    tEval = np.linspace(0, RUN_SPAN, 3000)
    sol = solve_ivp(rhs, (0, RUN_SPAN), initial + [0.0, initialSpeed], method='LSODA', t_eval=tEval)
    if not sol.success:
        return idx, lle, None, None
    mTrace = sol.y[4]
    postWarmup = mTrace[tEval >= WARMUP]
    if len(postWarmup) == 0:
        return idx, lle, None, None
    return idx, lle, float(postWarmup.min()), float(postWarmup.max())


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

    print(f"tracking EMA(speed, tau={TAU}) post-warmup range for {len(tasks)} discovered hits "
          f"over {int(RUN_SPAN)} time units...")
    with Pool(processes=8, initializer=_silenceFortranStdout) as pool:
        results = pool.map(emaTrace, tasks)

    stalledMaxima = []
    normalMinima = []
    print(f"\n{'idx':>6} {'LLE':>9} {'category':>10} {'post-warmup min m':>18} {'post-warmup max m':>18}")
    for idx, lle, mMin, mMax in results:
        if mMin is None:
            print(f"{idx:>6} {lle:>9.4f}   failed/skipped")
            continue
        category = 'STALLED' if idx in known else 'normal'
        print(f"{idx:>6} {lle:>9.4f} {category:>10} {mMin:>18.4f} {mMax:>18.4f}")
        if category == 'STALLED':
            stalledMaxima.append(mMax)
        else:
            normalMinima.append(mMin)

    print(f"\nSTALLED: max EMA(m) ever seen post-warmup = {max(stalledMaxima):.4f} "
          f"(worst case among {len(stalledMaxima)})")
    if normalMinima:
        print(f"NORMAL: min EMA(m) ever seen post-warmup = {min(normalMinima):.4f} "
              f"(worst dip among {len(normalMinima)})")
        print(f"\nSeparation gap: [{max(stalledMaxima):.4f}, {min(normalMinima):.4f}]")


if __name__ == '__main__':
    main()
