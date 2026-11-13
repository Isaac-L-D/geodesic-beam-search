import os
import sys
import time
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))

from attractorSimilarity import (getAttractorPoints, d2Descriptor, attractorDistance,
                                  _silenceFortranStdout, TARGET_ARC_LENGTH, N_OUTPUT_POINTS)
from sharedConstants import W_COLUMNS

# N_RUNS_PER_DESCRIPTOR=8 exists to average out multistability (different initial conditions
# landing in different basins). Whether real discovered hits actually need that many runs - as
# opposed to known attractors, which is what nRuns=8 was originally justified against - has never
# been checked. Each run of the descriptor stage costs a full 700,000-arc-length integration, so
# if hits turn out to rarely be multistable, cutting nRuns would be a real, evidence-backed
# speedup to the single most expensive per-item cost in a production round.

HITS_PATH = os.path.join(PROJECT_ROOT, 'geodesicBeamSearch', 'output', 'allUniqueHits.csv')
N_CANDIDATES = 4
N_RUNS = 8
POOL_WORKERS = 22  # matches nproc - safe to use the full machine once the validation job has exited
OUT_PATH = os.path.join(SCRIPT_DIR, 'nRunsPerDescriptorCheck.log')

# Cascade-filter question: does a much cheaper, shorter-arc-length descriptor still rank
# candidates correctly relative to each other, even though it's known to be under-converged in
# an absolute sense (the original TARGET_ARC_LENGTH sweep found shorter lengths give a biased,
# not just noisier, descriptor)? If a candidate's coarse descriptor still lands closer to its OWN
# full descriptor than to any other candidate's, a conservative early-rejection filter is safe.
COARSE_ARC_LENGTH = TARGET_ARC_LENGTH / 20
COARSE_N_POINTS = N_OUTPUT_POINTS // 10


def _runTask(task):
    candIdx, runIdx, wFlat, seed, coarse = task
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if coarse:
        points = getAttractorPoints(np.array(wFlat).reshape(3, 27), rng,
                                     targetArcLength=COARSE_ARC_LENGTH, nPoints=COARSE_N_POINTS)
    else:
        points = getAttractorPoints(np.array(wFlat).reshape(3, 27), rng)
    elapsed = time.time() - t0
    return candIdx, runIdx, points, elapsed, coarse


def main():
    df = pd.read_csv(HITS_PATH)
    top = df.sort_values('LLEMean', ascending=False).head(N_CANDIDATES)

    tasks = []
    seed = 0
    wFlatByCand = {}
    for candIdx, (_, row) in enumerate(top.iterrows()):
        wFlat = row[W_COLUMNS].to_numpy(dtype=np.float64)
        wFlatByCand[candIdx] = wFlat
        for runIdx in range(N_RUNS):
            tasks.append((candIdx, runIdx, wFlat.tolist(), seed, False))
            seed += 1
        # One coarse (short-arc-length) run per candidate, for the cascade-filter check.
        tasks.append((candIdx, 'coarse', wFlat.tolist(), seed, True))
        seed += 1

    log = open(OUT_PATH, 'w')
    def logLine(s):
        print(s)
        log.write(s + "\n")
        log.flush()

    logLine(f"Testing {N_CANDIDATES} real hits (LLE={top['LLEMean'].tolist()}), {N_RUNS} runs each "
            f"(+1 coarse run at arc length {COARSE_ARC_LENGTH:.0f} vs full {TARGET_ARC_LENGTH:.0f}), "
            f"{POOL_WORKERS} parallel workers")

    runsByCand = {i: [None] * N_RUNS for i in range(N_CANDIDATES)}
    coarseByCand = {}
    with Pool(POOL_WORKERS, initializer=_silenceFortranStdout) as pool:
        for candIdx, runIdx, points, elapsed, coarse in pool.imap_unordered(_runTask, tasks):
            status = "OK" if points is not None else "FAILED/TIMEOUT"
            logLine(f"  cand {candIdx} run {runIdx}{' (coarse)' if coarse else ''}: "
                    f"{status} ({elapsed:.1f}s)")
            if coarse:
                coarseByCand[candIdx] = points
            else:
                runsByCand[candIdx][runIdx] = points

    rng = np.random.default_rng(999)
    fullPooledByCand = {}
    coarseDescByCand = {}
    for candIdx in range(N_CANDIDATES):
        runs = [p for p in runsByCand[candIdx] if p is not None and len(p) >= 20]
        logLine(f"\n=== Candidate {candIdx} (LLE={top['LLEMean'].iloc[candIdx]:.4f}): "
                f"{len(runs)}/{N_RUNS} runs succeeded ===")
        if len(runs) < 2:
            logLine("  Too few successful runs to compare.")
            continue

        perRunDesc = [d2Descriptor(r, rng) for r in runs]
        perRunDesc = [d for d in perRunDesc if d is not None]

        # Spread among the individual runs themselves - this IS the multistability question:
        # do different initial conditions actually land somewhere different?
        pairDists = [attractorDistance(perRunDesc[i], perRunDesc[j])
                     for i in range(len(perRunDesc)) for j in range(i + 1, len(perRunDesc))]
        logLine(f"  Pairwise distance between individual runs: mean={np.mean(pairDists):.4f}, "
                f"max={np.max(pairDists):.4f}")

        fullPooled = d2Descriptor(np.concatenate(runs, axis=0), rng)
        fullPooledByCand[candIdx] = fullPooled
        singleRun = perRunDesc[0]
        twoRunPooled = d2Descriptor(np.concatenate(runs[:2], axis=0), rng) if len(runs) >= 2 else None

        distFullVsSingle = attractorDistance(fullPooled, singleRun)
        logLine(f"  Full {len(runs)}-run pooled descriptor vs. single-run (run 0): {distFullVsSingle:.4f}")
        if twoRunPooled is not None:
            distFullVsTwo = attractorDistance(fullPooled, twoRunPooled)
            logLine(f"  Full {len(runs)}-run pooled descriptor vs. 2-run pooled: {distFullVsTwo:.4f}")

        coarsePts = coarseByCand.get(candIdx)
        if coarsePts is not None and len(coarsePts) >= 20:
            coarseDescByCand[candIdx] = d2Descriptor(coarsePts, rng)

    logLine(f"\nReference scale: known-attractor self-distance (same system, different run) is "
            f"typically ~0.01-0.05; NOVELTY_DISTANCE_THRESHOLD is currently 0.08. If the "
            f"full-vs-single/full-vs-2-run distances above are well within that self-distance "
            f"band, nRuns could likely be cut without changing novelty decisions. If pairwise "
            f"run distances are large (comparable to or above threshold), these candidates ARE "
            f"multistable and nRuns=8 is doing real work.")

    # Cascade-filter check: for each candidate with a coarse descriptor, is it still closer to
    # its OWN full descriptor than to any OTHER candidate's full descriptor? If yes across the
    # board, a conservative coarse-first reject filter is safe despite the coarse descriptor
    # being individually under-converged/biased - what matters for a filter is ranking, not
    # absolute agreement.
    logLine(f"\n=== Cascade-filter check: does the coarse (arc length {COARSE_ARC_LENGTH:.0f}) "
            f"descriptor still rank each candidate closest to ITSELF? ===")
    for candIdx, coarseDesc in coarseDescByCand.items():
        distToOwn = attractorDistance(coarseDesc, fullPooledByCand[candIdx])
        distToOthers = {j: attractorDistance(coarseDesc, fullPooledByCand[j])
                         for j in fullPooledByCand if j != candIdx}
        nearestOther = min(distToOthers, key=distToOthers.get) if distToOthers else None
        logLine(f"  cand {candIdx}: coarse-vs-own-full={distToOwn:.4f}  "
                f"nearest-OTHER-full={('cand ' + str(nearestOther) + ' @ ' + f'{distToOthers[nearestOther]:.4f}') if nearestOther is not None else 'n/a'}  "
                f"{'PASS (own is closer)' if nearestOther is None or distToOwn < distToOthers[nearestOther] else 'FAIL (confused with another candidate)'}")
    log.close()


if __name__ == '__main__':
    main()
