import numpy as np
from scipy.spatial import cKDTree

# Finds the best RIGID alignment (rotation + translation, no scale - both clouds are pre-normalized
# to the same scale first) between two trajectory point clouds via Iterative Closest Point, then
# reports the residual mismatch after that best-fit alignment. This is closer to how a person
# actually compares two 3D shapes - mentally rotate one until it best matches the other, then judge
# what's left over - than a rotation-invariant SUMMARY STATISTIC (like curvature) that never
# explicitly finds the alignment at all.


def _normalize(points):
    centered = points - points.mean(axis=0)
    scale = np.sqrt((centered ** 2).sum(axis=1)).mean()
    return centered / scale


def _subsample(points, n, rng):
    if len(points) <= n:
        return points
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


# Kabsch algorithm: optimal rotation minimizing sum-of-squared-distances between corresponding
# (already-matched) point pairs. Determinant correction keeps this a proper rotation (no mirror
# flip) - see icpAlign's docstring for why reflections are handled separately, not silently allowed
# here.
def _bestRotation(source, target):
    H = source.T @ target
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1, 1, d])
    R = Vt.T @ correction @ U.T
    return R


def _icpOnce(source, target, maxIters=50, tol=1e-6):
    tree = cKDTree(target)
    current = source.copy()
    prevError = np.inf
    for _ in range(maxIters):
        dists, idx = tree.query(current)
        matched = target[idx]
        R = _bestRotation(current, matched)
        current = current @ R.T
        error = float(np.mean(dists ** 2))
        if abs(prevError - error) < tol:
            break
        prevError = error
    dists, _ = tree.query(current)
    return float(np.mean(dists)), current


# Runs ICP from several random initial rotations (it's only locally optimal from wherever it
# starts) and keeps the best result. allowReflection additionally tries a mirrored copy of the
# source, since a mirror-image shape often still reads as "the same" to a casual human glance even
# though it's not reachable by a proper rotation alone.
def icpAlign(pointsA, pointsB, nPoints=1500, nRestarts=8, allowReflection=True, seed=0):
    rng = np.random.default_rng(seed)
    A = _subsample(_normalize(pointsA), nPoints, rng)
    B = _subsample(_normalize(pointsB), nPoints, rng)

    candidates = [A]
    if allowReflection:
        mirrored = A.copy()
        mirrored[:, 0] *= -1
        candidates.append(mirrored)

    bestResidual, bestAligned = np.inf, None
    for source in candidates:
        for _ in range(nRestarts):
            randomR = _randomRotation(rng)
            residual, aligned = _icpOnce(source @ randomR.T, B)
            if residual < bestResidual:
                bestResidual, bestAligned = residual, aligned

    return bestResidual, bestAligned, B


def _randomRotation(rng):
    # Uniform random rotation via QR decomposition of a random Gaussian matrix.
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q