import os
import sys
import numpy as np
from multiprocessing import Pool

PROJECT_ROOT = "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))

import attractorSimilarity
from attractorSimilarity import (loadDatasetAttractors, getAttractorPointsWithTimeout,
                                  d2Descriptor, attractorDistance, TRANSIENT_FRACTION,
                                  N_OUTPUT_POINTS)
from engine import _silenceFortranStdout

# This is an exploratory calibration run, not production - large multipliers on already-fast
# attractors can need far more model time than the production MAX_TIME_CAP allows, and the
# subprocess wall-clock budget needs headroom to match. Neither constraint should be this
# loose in the final pipeline, just while searching for where convergence actually happens.
attractorSimilarity.ARC_LENGTH_TIMEOUT_SECONDS = 600
attractorSimilarity.ARC_LENGTH_TIMEOUT_RETRIES = 1

# Old-method arc lengths already measured for these (see calibrateArcLength.py output) -
# used as a per-attractor starting point since their "natural speed" varies ~700,000x, so a
# shared absolute search range would be badly wrong for either extreme.
CANDIDATES = {
    'bouali2': 1.2,
    'sprotti': 74.3,
    'lorenz': None,   # unknown old value - measured fresh below
    'dequanli': 873104.3,
}
MULTIPLIERS = [1, 3, 10, 30, 100, 300]
N_RUNS = 3   # smaller than production's 8 - enough to see the convergence trend, not final quality


HARD_MAX_TIME_CAP = 20000.0  # dense_output=True stores the full step history for the whole
# integration - past this many model-time units the interpolant itself gets big enough to OOM
# a worker (confirmed: dequanli x300 implied a ~720,000-unit budget and killed the process).
# A task that can't reach its target within this ceiling is reported as FAILED, which is itself
# useful calibration signal rather than something to work around.


def descriptorAt(name, W, targetArcLength, impliedSpeed, seedBase):
    # Generous model-time budget sized to the target itself, not the fixed production default -
    # 10x the naive "distance/speed" estimate, so a legitimately-slow-to-reach target doesn't
    # get mistaken for a rejection. Capped so a fast/outlier attractor's large multiplier can't
    # blow up worker memory.
    maxTimeCap = min(HARD_MAX_TIME_CAP, max(2000.0, (targetArcLength / impliedSpeed) * 10))
    allPoints = []
    for i in range(N_RUNS):
        rng = np.random.default_rng(seedBase + i)
        initial = rng.uniform(-1, 1, size=3).tolist()
        points = getAttractorPointsWithTimeout(W, initial, targetArcLength, N_OUTPUT_POINTS, maxTimeCap)
        if points is not None:
            startIdx = int(N_OUTPUT_POINTS * TRANSIENT_FRACTION)
            trimmed = points[startIdx:]
            if len(trimmed) >= 20:
                allPoints.append(trimmed)
    if not allPoints:
        return None
    pooled = np.concatenate(allPoints, axis=0)
    return d2Descriptor(pooled, np.random.default_rng(seedBase + 1000))


def task(args):
    name, W, targetArcLength, impliedSpeed, seedBase = args
    return (name, targetArcLength, seedBase), descriptorAt(name, W, targetArcLength, impliedSpeed, seedBase)


def main():
    known = loadDatasetAttractors()

    # Measure lorenz's old-method arc length fresh (not in the pre-measured set above).
    from ODEClass import ODE, logDetails
    ode = ODE(W=known['lorenz'], doFitToCube=False)
    rng = np.random.default_rng(0)
    res = ode.integrate(initial_pos=rng.uniform(-1, 1, size=3).tolist(), endTime=300.0, dt=0.01,
                         log=logDetails())
    _, y = res
    n = y.shape[1]
    pts = y[:, int(n * TRANSIENT_FRACTION):].T
    CANDIDATES['lorenz'] = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    print(f"lorenz old-method arc length: {CANDIDATES['lorenz']:.1f}")

    # Average speed implied by the old method's own run - baseLength was covered over
    # 300*(1-TRANSIENT_FRACTION) time units of post-transient integration.
    impliedSpeeds = {name: baseLength / (300.0 * (1 - TRANSIENT_FRACTION))
                      for name, baseLength in CANDIDATES.items()}

    tasks = []
    for name, baseLength in CANDIDATES.items():
        W = known[name]
        speed = impliedSpeeds[name]
        for m in MULTIPLIERS:
            targetLength = baseLength * m
            # two independent descriptors at the same target - gives the self-distance noise floor
            tasks.append((name, W, targetLength, speed, 10000 + hash((name, m, 'a')) % 10000))
            tasks.append((name, W, targetLength, speed, 20000 + hash((name, m, 'b')) % 10000))

    print(f"{len(tasks)} descriptor computations to run...")
    # Fewer workers than usual (22 -> 8): dense_output=True holds the full step history in
    # memory for the whole integration, and running that many long integrations concurrently
    # is what OOM-killed the process last time. Slower, but bounded.
    with Pool(processes=8, initializer=_silenceFortranStdout) as pool:
        results = dict(pool.imap_unordered(task, tasks))

    for name, baseLength in CANDIDATES.items():
        print(f"\n=== {name} (old-method arc length: {baseLength:.1f}) ===")
        prevDesc = None
        for m in MULTIPLIERS:
            targetLength = baseLength * m
            keyA = None
            keyB = None
            for key in results:
                if key[0] == name and key[1] == targetLength:
                    if keyA is None:
                        keyA = key
                    else:
                        keyB = key
            descA = results.get(keyA)
            descB = results.get(keyB)
            if descA is None or descB is None:
                print(f"  x{m:>4} (L={targetLength:>10.1f}): FAILED (one or both runs rejected)")
                continue
            selfDist = attractorDistance(descA, descB)
            changeFromPrev = attractorDistance(descA, prevDesc) if prevDesc is not None else float('nan')
            print(f"  x{m:>4} (L={targetLength:>10.1f}): self-distance={selfDist:.4f}  "
                  f"change-from-prev-L={changeFromPrev:.4f}")
            prevDesc = descA


if __name__ == '__main__':
    main()
