import numpy as np
from renderAndEmbed import renderAllViews, CAMERA_ANGLES
from momentSimilarity import zernikeFeatures
from icpAlignment import icpAlign

# Two-stage retrieval, motivated by validateMomentMethods.py/validatePerceptualMethods.py's
# real numbers on the 36 known attractors:
#   - Zernike moments (2D, multi-view): AUC 0.9921, cheap - one closed-form computation per
#     system, no pairwise search.
#   - icpAlign: AUC 0.9857, but better-calibrated at a hard threshold (FP 2.8% vs Zernike's
#     19.4%) - it actually finds the best-fit rigid alignment rather than summarizing shape,
#     at the cost of a random-restart search per PAIR, which doesn't scale once the reference
#     catalog holds thousands of discoveries.
# Zernike ranks nearly as well as ICP overall, so a short shortlist rarely drops the true
# nearest match; ICP then only has to run K times per candidate instead of N times, recovering
# ICP's calibration without paying its full O(N^2) cost against the whole catalog.
SHORTLIST_K = 5
ICP_RESTARTS = 8


def zernikeDescriptor(points, angles=CAMERA_ANGLES):
    images = renderAllViews(points, angles)
    return [zernikeFeatures(img) for img in images]


def _zernikeDistance(featsA, featsB):
    return float(np.mean([np.linalg.norm(featsA[i] - featsB[i]) for i in range(len(featsA))]))


def zernikeDistanceBatch(desc, refDescs):
    return np.array([_zernikeDistance(desc, r) for r in refDescs])


# refNames/refDescs/refPoints share the same key order. Returns the lowest ICP residual found
# within the shortlist (same scale/threshold as icpAlign's own validated FP/FN numbers) and
# which reference achieved it - Zernike only decides which K references are worth that
# expensive step, it never itself produces the final distance.
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