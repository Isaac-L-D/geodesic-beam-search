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

attractorSimilarity.ARC_LENGTH_TIMEOUT_SECONDS = 600
attractorSimilarity.ARC_LENGTH_TIMEOUT_RETRIES = 1

# newtonliepnik: slow/simple like bouali2 (old-method arc length 19.5 vs bouali2's 1.2) but
# NOT one of Phase 2's flagged-unreliable attractors - isolates whether bouali2's oversampling
# degradation is a general slow-attractor phenomenon or specific to bouali2 itself.
NAME = 'newtonliepnik'
BASE_LENGTH = 19.5
MULTIPLIERS = [1, 3, 10, 30, 100, 300]
N_RUNS = 3

HARD_MAX_TIME_CAP = 20000.0


def descriptorAt(W, targetArcLength, impliedSpeed, seedBase):
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
    targetArcLength, seedBase = args
    return (targetArcLength, seedBase), descriptorAt(W_GLOBAL, targetArcLength, SPEED_GLOBAL, seedBase)


def main():
    global W_GLOBAL, SPEED_GLOBAL
    known = loadDatasetAttractors()
    W_GLOBAL = known[NAME]
    SPEED_GLOBAL = BASE_LENGTH / (300.0 * (1 - TRANSIENT_FRACTION))

    tasks = []
    for m in MULTIPLIERS:
        targetLength = BASE_LENGTH * m
        tasks.append((targetLength, 10000 + hash((m, 'a')) % 10000))
        tasks.append((targetLength, 20000 + hash((m, 'b')) % 10000))

    print(f"{NAME} old-method arc length: {BASE_LENGTH}")
    print(f"{len(tasks)} descriptor computations to run...")
    with Pool(processes=8, initializer=_silenceFortranStdout) as pool:
        results = dict(pool.imap_unordered(task, tasks))

    prevDesc = None
    for m in MULTIPLIERS:
        targetLength = BASE_LENGTH * m
        keyA = None
        keyB = None
        for key in results:
            if key[0] == targetLength:
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
