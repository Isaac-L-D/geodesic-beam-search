import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool

PROJECT_ROOT = "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))

from engine import _silenceFortranStdout, ALL_UNIQUE_HITS_PATH, RADII
from sharedConstants import W_COLUMNS

SAFETY_TIMEOUT = 120  # generous cap so a genuine stall doesn't block this measurement run
INTEGRATE_HELPER = os.path.join(PROJECT_ROOT, "geodesicBeamSearch", "integrateWorker.py")


# Reconstructed - localDensitySampler.py, which these originally lived in, no longer exists.
def capSeeds(seeds, n, rng):
    if len(seeds) <= n:
        return seeds
    idx = rng.choice(len(seeds), size=n, replace=False)
    return seeds[idx]


def buildPerturbationTasks(label, seeds, radii, nPerRadius, rng):
    tasks = []
    for seedIdx, seed in enumerate(seeds):
        seedW = seed.reshape(3, 27)
        for radius in radii:
            for _ in range(nPerRadius):
                direction = rng.normal(size=seedW.shape)
                direction /= np.linalg.norm(direction)
                perturbed = seedW + radius * direction
                perturbed /= np.linalg.norm(perturbed)
                tasks.append((label, seedIdx, radius, perturbed))
    return tasks


def timedIntegrate(WFlat):
    payload = json.dumps({"W": np.asarray(WFlat).flatten().tolist()})
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, INTEGRATE_HELPER],
            input=payload, capture_output=True, text=True,
            timeout=SAFETY_TIMEOUT, cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            return elapsed, 'crashed', None
        parsed = json.loads(result.stdout)
        return elapsed, 'completed', parsed['success']
    except subprocess.TimeoutExpired:
        return time.time() - t0, 'safety_timeout', None


def main():
    rng = np.random.default_rng(12345)
    allHits = pd.read_csv(ALL_UNIQUE_HITS_PATH)
    round2 = allHits[allHits['Round'] == 2]
    seeds = round2[W_COLUMNS].to_numpy(dtype=np.float64)
    seeds = capSeeds(seeds, 40, rng)
    print(f"using {len(seeds)} round-2 seeds")

    tasks = buildPerturbationTasks('Test', seeds, RADII, 4, rng)
    WFlats = [t[3].flatten() for t in tasks]
    print(f"{len(WFlats)} candidates to time, safety timeout={SAFETY_TIMEOUT}s")

    t0 = time.time()
    with Pool(processes=22, initializer=_silenceFortranStdout) as pool:
        results = list(pool.imap_unordered(timedIntegrate, WFlats))
    totalElapsed = time.time() - t0
    print(f"done in {totalElapsed:.1f}s wall time")

    df = pd.DataFrame(results, columns=['elapsed', 'outcome', 'success'])
    df.to_csv('integrateTimings.csv', index=False)

    print("\n=== outcome counts ===")
    print(df['outcome'].value_counts())

    normal = df[df['outcome'] == 'completed']['elapsed']
    print(f"\n=== elapsed time (s) for the {len(normal)} normally-completed candidates ===")
    for p in [50, 90, 95, 99, 99.5, 99.9, 100]:
        print(f"p{p}: {np.percentile(normal, p):.2f}s")
    print(f"max: {normal.max():.2f}s")

    stalled = df[df['outcome'] == 'safety_timeout']
    print(f"\n{len(stalled)} candidate(s) hit the {SAFETY_TIMEOUT}s safety timeout (genuine stalls)")


if __name__ == '__main__':
    main()
