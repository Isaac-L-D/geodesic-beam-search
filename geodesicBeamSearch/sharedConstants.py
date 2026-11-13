import numpy as np


def invertFlatten(vec):
    return np.reshape(vec, (3, 27))


# Target working norm a (typically unit-norm) candidate gets scaled to before evaluation, so the
# fixed integration/LLE time budgets (tuned against a specific native coefficient scale, not the
# unit sphere) don't under-resolve small-norm candidates. 2.6 was chosen by sweeping scaleBy in
# {1,2,2.6,5,10,20,40} against 34 known-robustly-chaotic attractors: the chaotic/non-chaotic
# classification stayed flat at 97.1% detection across nearly the whole range, so the exact
# value barely matters for that - 2.6 sits comfortably in the plateau. Separately, the absolute
# LLE magnitude at this scale is heavily compressed relative to native-scale values (confirmed
# directly earlier tonight) - but that's a magnitude question, not a classification one, and
# doesn't affect the LLE>0 threshold this pipeline actually screens on.
EVAL_REFERENCE_NORM = 2.6

MONOMIAL_NAMES = ['1', 'z', 'z²', 'y', 'yz', 'yz²', 'y²', 'y²z', 'y²z²', 'x', 'xz', 'xz²', 'xy',
                   'xyz', 'xyz²', 'xy²', 'xy²z', 'xy²z²', 'x²', 'x²z', 'x²z²', 'x²y', 'x²yz',
                   'x²yz²', 'x²y²', 'x²y²z', 'x²y²z²']
W_COLUMNS = [f'{eq}_{m}' for eq in ('dx', 'dy', 'dz') for m in MONOMIAL_NAMES]

# From the attractorSimilarityValidation.txt attractorSimilarity validation (44 reliable attractors): mean distance
# between two independent runs of the SAME attractor - the baseline "noise floor" a discovery's
# distance from its nearest reference is judged against below. D2 doesn't drive the novelty
# decision (the Zernike+ICP cascade does - see runRound), but stays calibrated so it can
# independently flag disagreements with the cascade's verdict for review, rather than being
# dropped outright.
RELIABLE_MEAN_SELF_DISTANCE = 0.0242
_RELIABLE_MEAN_CROSS_DISTANCE = 0.1229  # mean distance between two DIFFERENT attractors, same validation run
# threshold = meanSelf + 0.25*(meanCross - meanSelf), biased toward not over-collapsing distinct
# discoveries into "already known".
NOVELTY_DISTANCE_THRESHOLD = 0.0488

# From attractorShapeAnalysis/curvatureValidation.txt (36 reliable attractors, full production
# settings - 8 replicas, real targetArcLength/nPoints, same descriptorForW call a real discovery
# uses). Same threshold formula as D2 above. Curvature is a secondary disagreement flag now too
# (see NOVELTY_DISTANCE_THRESHOLD_CASCADE in engine.py), not the novelty-decision
# driver.
RELIABLE_MEAN_SELF_DISTANCE_CURVATURE = 0.0616
_RELIABLE_MEAN_CROSS_DISTANCE_CURVATURE = 0.6706
NOVELTY_DISTANCE_THRESHOLD_CURVATURE = (RELIABLE_MEAN_SELF_DISTANCE_CURVATURE
    + 0.25 * (_RELIABLE_MEAN_CROSS_DISTANCE_CURVATURE - RELIABLE_MEAN_SELF_DISTANCE_CURVATURE))

# From attractorShapeAnalysis/attractorSimilarityValidation.txt and curvatureValidation.txt:
# these either fail to integrate at all, or their self-distance (two independent runs of the SAME
# attractor) is a population z-score outlier for whichever descriptor is in use - i.e. their own
# shape descriptor isn't stable/trustworthy under that method. Excluded from both seeding (a seed
# whose own identity is unstable is a bad thing to branch from) and the reference set used for the
# novelty check (an unstable reference could corrupt distinctness verdicts for anything found
# nearby).
EXCLUDED_ATTRACTOR_NAMES = {
    'atmosphericregime', 'genesiotesi', 'newtonliepnik', 'panxuzhou', 'shimizumorioka',
    'sprotti', 'wangsun',  # fail to integrate at all (attractorSimilarityValidation.txt)
    'sprottd', 'sprottjerk', 'sprotttorus',  # fail to integrate at all (curvatureValidation.txt)
    'bouali2', 'chenlee', 'valliselnino',  # D2 self-distance z-score outlier
    'liuchen', 'lorenz84', 'rayleighbenard', 'rucklidge', 'sprotte', 'tsucs2',
    # curvature self-distance z-score outlier (curvatureValidation.txt) - liuchen in particular is
    # extreme (self-distance 2.57 vs a typical ~0.06), not a borderline call.
}
