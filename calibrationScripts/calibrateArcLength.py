import os
import sys
import numpy as np
from multiprocessing import Pool

PROJECT_ROOT = "/home/isaac-dash/Documents/Chaos Theory Project/SpaceOfKnownAttractors"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))

from ODEClass import ODE, logDetails
from attractorSimilarity import loadDatasetAttractors, TRANSIENT_FRACTION
from engine import _silenceFortranStdout

OLD_END_TIME = 300.0
OLD_DT = 0.01


# Reimplements the OLD (pre-arc-length, already-validated) fixed-time getAttractorPoints
# directly, since that function itself has been replaced - this measures "how much arc
# length did an already-good run cover" using the same method the project trusted before.
def oldMethodArcLength(args):
    name, W = args
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    ode = ODE(W=W, doFitToCube=False)
    initial = rng.uniform(-1, 1, size=3).tolist()
    log = logDetails()
    res = ode.integrate(initial_pos=initial, endTime=OLD_END_TIME, dt=OLD_DT, log=log)
    if res is None:
        return name, None
    _, y = res
    n = y.shape[1]
    startIdx = int(n * TRANSIENT_FRACTION)
    points = y[:, startIdx:].T  # (N, 3), post-transient only, matching what descriptors use
    if len(points) < 2:
        return name, None
    arcLength = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    return name, arcLength


def main():
    known = loadDatasetAttractors()
    names = list(known.keys())
    tasks = [(name, known[name]) for name in names]

    with Pool(processes=22, initializer=_silenceFortranStdout) as pool:
        results = dict(pool.imap(oldMethodArcLength, tasks))

    lengths = {name: L for name, L in results.items() if L is not None}
    print(f"{len(lengths)}/{len(names)} attractors integrated successfully")

    vals = np.array(list(lengths.values()))
    print(f"\narc length over a currently-validated {OLD_END_TIME}-time-unit run (post-transient):")
    for p in [0, 10, 25, 50, 60, 75, 90, 100]:
        print(f"  p{p}: {np.percentile(vals, p):.1f}")

    print("\nsmallest 5:")
    for name, L in sorted(lengths.items(), key=lambda x: x[1])[:5]:
        print(f"  {name}: {L:.1f}")
    print("largest 5:")
    for name, L in sorted(lengths.items(), key=lambda x: x[1])[-5:]:
        print(f"  {name}: {L:.1f}")


if __name__ == '__main__':
    main()
