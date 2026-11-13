import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))

from attractorSimilarity import (loadDatasetAttractors, descriptorForW, attractorDistance,
                                  _silenceFortranStdout)
from sharedConstants import EXCLUDED_ATTRACTOR_NAMES

RESULTS_LOG_PATH = os.path.join(SCRIPT_DIR, 'curvatureValidation.txt')
DISTANCE_MATRIX_PATH = os.path.join(SCRIPT_DIR, 'curvatureValidationMatrix.csv')

# Same robust-outlier rule attractorSimilarity.py itself uses for the D2 baseline.
MAD_OUTLIER_THRESHOLD = 3.5


def logResult(line=""):
    print(line)
    with open(RESULTS_LOG_PATH, 'a') as f:
        f.write(line + "\n")


def modifiedZScores(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-12:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


# One task per (attractor, run) - two independent descriptorForW calls per attractor, at full
# production settings (nRuns=N_RUNS_PER_DESCRIPTOR=8, default targetArcLength/nPoints), mirroring
# exactly how a real discovery's descriptor gets built - not a reduced-cost stand-in.
def _task(args):
    name, run, W, seed = args
    rng = np.random.default_rng(seed)
    desc = descriptorForW(W, rng)
    return name, run, desc


def main():
    if os.path.exists(RESULTS_LOG_PATH):
        os.remove(RESULTS_LOG_PATH)
    known = loadDatasetAttractors()
    names = [n for n in known if n not in EXCLUDED_ATTRACTOR_NAMES]
    logResult(f"Loaded {len(names)} known attractors (excluding {len(EXCLUDED_ATTRACTOR_NAMES)} "
              f"already flagged unreliable for D2): {names}")

    tasks = []
    seed = 0
    for name in names:
        for run in ('A', 'B'):
            tasks.append((name, run, known[name], seed))
            seed += 1

    results = {}
    with Pool(initializer=_silenceFortranStdout) as pool:
        for name, run, desc in tqdm(pool.imap_unordered(_task, tasks), total=len(tasks),
                                     desc='Building curvature+D2 descriptors (production settings)'):
            results[(name, run)] = desc

    validNames = [n for n in names if results.get((n, 'A')) is not None
                  and results.get((n, 'B')) is not None
                  and results[(n, 'A')]['curvature'] is not None
                  and results[(n, 'B')]['curvature'] is not None
                  and results[(n, 'A')]['d2'] is not None
                  and results[(n, 'B')]['d2'] is not None]
    failed = [n for n in names if n not in validNames]
    if failed:
        logResult(f"\n{len(failed)} attractors failed to produce a full descriptor pair, dropped: {failed}")

    for label, key in [('Curvature', 'curvature'), ('D2', 'd2')]:
        logResult(f"\n=== {label} ===")
        selfDistances = {n: attractorDistance(results[(n, 'A')][key], results[(n, 'B')][key])
                          for n in validNames}
        for n in validNames:
            logResult(f"  self-distance {n}: {selfDistances[n]:.4f}")

        selfArray = np.array([selfDistances[n] for n in validNames])
        zScores = modifiedZScores(selfArray)
        reliableNames = [n for n, z in zip(validNames, zScores) if z <= MAD_OUTLIER_THRESHOLD]
        unreliableNames = [n for n, z in zip(validNames, zScores) if z > MAD_OUTLIER_THRESHOLD]
        if unreliableNames:
            logResult(f"\n  Unreliable for {label} (self-distance z > {MAD_OUTLIER_THRESHOLD}): {unreliableNames}")

        matrix = pd.DataFrame(index=reliableNames, columns=reliableNames, dtype=float)
        for n1 in reliableNames:
            for n2 in reliableNames:
                matrix.loc[n1, n2] = attractorDistance(results[(n1, 'A')][key], results[(n2, 'A')][key])
        matrix.to_csv(DISTANCE_MATRIX_PATH.replace('.csv', f'_{key}.csv'))

        crossDistances = [matrix.loc[n1, n2] for n1 in reliableNames for n2 in reliableNames if n1 != n2]
        meanSelf = float(np.mean([selfDistances[n] for n in reliableNames]))
        meanCross = float(np.mean(crossDistances))
        threshold = meanSelf + 0.25 * (meanCross - meanSelf)
        logResult(f"\n  {len(reliableNames)}/{len(validNames)} attractors reliable for {label}")
        logResult(f"  Mean self-distance: {meanSelf:.4f}")
        logResult(f"  Mean cross-distance: {meanCross:.4f}")
        logResult(f"  Separation ratio: {meanCross/meanSelf:.2f}x")
        logResult(f"  Novelty threshold (self + 0.25*(cross-self)): {threshold:.4f}")

    print(f"\nDone. Log: {RESULTS_LOG_PATH}")


if __name__ == '__main__':
    main()
