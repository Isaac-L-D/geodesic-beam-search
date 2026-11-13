import os
import sys
import time
import numpy as np
from multiprocessing import Pool

PROJECT_ROOT = "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))

import attractorSimilarity
from attractorSimilarity import (loadDatasetAttractors, getAttractorPointsWithTimeout,
                                  d2Descriptor, attractorDistance, TRANSIENT_FRACTION,
                                  N_OUTPUT_POINTS, TARGET_ARC_LENGTH)
from engine import _silenceFortranStdout

attractorSimilarity.ARC_LENGTH_TIMEOUT_SECONDS = 600
attractorSimilarity.ARC_LENGTH_TIMEOUT_RETRIES = 1

# Generous first guess, being dialed in by observing how much model time unit-norm lorenz
# actually needs to reach TARGET_ARC_LENGTH - native lorenz needed ~7,200 (norm 31.5), and a
# unit-norm candidate in the same direction moves ~31.5x slower to cover the same arc length,
# implying ~227,000 - padded well above that since real wall-clock cost, not just the naive
# estimate, is what determines whether this is practical.
CANDIDATE_MAX_TIME_CAP = 2_000_000.0
N_RUNS = 8  # matches production N_RUNS_PER_DESCRIPTOR


def sampleDescriptor(args):
    W, maxTimeCap, seedBase = args
    rng = np.random.default_rng(seedBase)
    initial = rng.uniform(-1, 1, size=3).tolist()
    t0 = time.time()
    points = getAttractorPointsWithTimeout(W, initial, TARGET_ARC_LENGTH, N_OUTPUT_POINTS, maxTimeCap)
    elapsed = time.time() - t0
    if points is None:
        return None, elapsed
    startIdx = int(N_OUTPUT_POINTS * TRANSIENT_FRACTION)
    trimmed = points[startIdx:]
    if len(trimmed) < 20:
        return None, elapsed
    return trimmed, elapsed


def main():
    known = loadDatasetAttractors()
    nativeW = known['lorenz']
    unitW = nativeW / np.linalg.norm(nativeW)

    print(f"TARGET_ARC_LENGTH = {TARGET_ARC_LENGTH}")
    print(f"native lorenz norm: {np.linalg.norm(nativeW):.2f}")

    nativeTasks = [(nativeW, 20000.0, 30000 + i) for i in range(N_RUNS)]
    unitTasks = [(unitW, CANDIDATE_MAX_TIME_CAP, 40000 + i) for i in range(N_RUNS)]

    print(f"running {len(nativeTasks)} native + {len(unitTasks)} unit-norm runs...")
    with Pool(processes=8, initializer=_silenceFortranStdout) as pool:
        nativeResults = pool.map(sampleDescriptor, nativeTasks)
        unitResults = pool.map(sampleDescriptor, unitTasks)

    nativeTimes = [e for _, e in nativeResults]
    unitTimes = [e for _, e in unitResults]
    print(f"native run times: min={min(nativeTimes):.1f}s max={max(nativeTimes):.1f}s")
    print(f"unit-norm run times: min={min(unitTimes):.1f}s max={max(unitTimes):.1f}s")

    nativePoints = [p for p, _ in nativeResults if p is not None]
    unitPoints = [p for p, _ in unitResults if p is not None]
    print(f"native: {len(nativePoints)}/{N_RUNS} succeeded")
    print(f"unit-norm: {len(unitPoints)}/{N_RUNS} succeeded")

    if len(nativePoints) < 2 or len(unitPoints) < 2:
        print("not enough successful runs to compare - aborting")
        return

    rng = np.random.default_rng(0)
    half = len(nativePoints) // 2
    nativeDescA = d2Descriptor(np.concatenate(nativePoints[:half]), rng)
    nativeDescB = d2Descriptor(np.concatenate(nativePoints[half:]), rng)
    unitDesc = d2Descriptor(np.concatenate(unitPoints), rng)

    nativeSelfDist = attractorDistance(nativeDescA, nativeDescB)
    crossDist = attractorDistance(nativeDescA, unitDesc)
    print(f"\nnative lorenz self-distance (two halves): {nativeSelfDist:.4f}")
    print(f"native lorenz vs unit-norm lorenz-direction distance: {crossDist:.4f}")
    print(f"(previous reference-norm-scaling approach left this at 0.429)")


if __name__ == '__main__':
    main()
