import numpy as np
import mahotas
from scipy.spatial import cKDTree
from renderSilhouette import renderAllViews, CAMERA_ANGLES

# Two-stage novelty-check cascade, replacing curvature as the primary decision driver (see
# attractorSimilarityTopN.py's descriptorForW). Validated on the 36 known attractors via a
# catalog-style retrieval test (self: true match present in the reference catalog; leave-one-out:
# true match removed, simulating a genuinely new discovery) - see
# perceptualSimilarity/cascadeValidationResults.npz / validateCascade.py for the numbers this is
# based on: Recall@5=97.2%, Top-1=88.9%, AUC=0.8958, FalsePositive=11.1%, FalseNegative=11.1%,
# each meaningfully better than curvature's own numbers under the identical test (AUC=0.8218,
# FalsePositive=22.2%, FalseNegative=36.1%).
#
#   1. Zernike moments (cheap, closed-form, O(1) per system): rank the whole reference catalog by
#      distance and take the K closest - this alone nearly matches curvature's cost.
#   2. ICP alignment (expensive, iterative): only run on those K candidates, return the best
#      (lowest) residual and which reference achieved it - the actual accept/reject distance.
#
# This keeps the expensive step bounded at K calls per candidate regardless of how large the
# reference catalog grows, rather than needing a full pairwise ICP search against everything.

ZERNIKE_DEGREE = 12
SHORTLIST_K = 5
ICP_RESTARTS = 8


def _toGray(img):
    return np.asarray(img.convert('L'), dtype=np.float64) / 255.0


# Zernike moment magnitudes - orthogonal polynomial basis over a disk; taking the magnitude (not
# phase) discards rotation information by construction. Closed-form, non-learned - same rigor
# category as curvature/D2.
def zernikeFeatures(img, degree=ZERNIKE_DEGREE):
    gray = (_toGray(img) * 255).astype(np.uint8)
    radius = min(gray.shape) // 2
    return np.array(mahotas.features.zernike_moments(gray, radius, degree=degree))


# points is (3, N) - renderAllViews'/renderSilhouette's convention, the OPPOSITE of
# attractorSimilarity.py's (N, 3). Callers holding (N,3) points (e.g. descriptorForW's `pooled`)
# must pass `pooled.T` here, never `pooled` directly - this exact mismatch has caused two real
# bugs already this project.
def zernikeDescriptor(points, angles=CAMERA_ANGLES):
    images = renderAllViews(points, angles)
    return [zernikeFeatures(img) for img in images]


def _zernikeDistance(featsA, featsB):
    return float(np.mean([np.linalg.norm(featsA[i] - featsB[i]) for i in range(len(featsA))]))


def zernikeDistanceBatch(desc, refDescs):
    return np.array([_zernikeDistance(desc, r) for r in refDescs])


# --- ICP alignment: finds the best rigid rotation between two point clouds, then reports the
# residual mismatch after that best-fit alignment. All of this expects (N, 3) points -
# attractorSimilarity.py's native convention, no transpose needed for anything below. ---

def _normalize(points):
    centered = points - points.mean(axis=0)
    scale = np.sqrt((centered ** 2).sum(axis=1)).mean()
    return centered / scale


def _subsample(points, n, rng):
    if len(points) <= n:
        return points
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


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


def _randomRotation(rng):
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def icpAlign(pointsA, pointsB, nPoints=1500, nRestarts=ICP_RESTARTS, allowReflection=True, seed=0):
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


# refNames/refDescs/refPoints share the same key order. Returns the lowest ICP residual found
# within the Zernike shortlist and which reference achieved it - Zernike only decides which K
# references are worth the expensive ICP step, it never itself produces the final distance.
def cascadeNearest(candidatePoints, candidateDesc, refNames, refDescs, refPoints,
                    k=SHORTLIST_K, nRestarts=ICP_RESTARTS):
    zernikeDists = zernikeDistanceBatch(candidateDesc, refDescs)
    shortlistIdx = np.argsort(zernikeDists)[:k]

    bestResidual, bestName = np.inf, None
    for i in shortlistIdx:
        residual, _, _ = icpAlign(candidatePoints, refPoints[refNames[i]], nRestarts=nRestarts)
        if residual < bestResidual:
            bestResidual, bestName = residual, refNames[i]
    return bestResidual, bestName, zernikeDists
