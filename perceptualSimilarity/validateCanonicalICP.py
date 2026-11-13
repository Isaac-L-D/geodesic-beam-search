import os
import sys
import time
import itertools
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'geodesicBeamSearch'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'attractorShapeAnalysis'))
os.chdir(PROJECT_ROOT)

from attractorSimilarity import loadDatasetAttractors
import sharedConstants as ls
from getTrajectory import getTrajectory
from icpAlignment import icpAlignCanonical

# Re-validates ICP using the PCA-canonicalized version (icpAlignCanonical) instead of the original
# random-8-restart icpAlign, against the same 36 known attractors - directly comparable to the
# icpSelf/icpCross already saved in validationResults.npz from the original version.

known = loadDatasetAttractors()
names = sorted(n for n in known if n not in ls.EXCLUDED_ATTRACTOR_NAMES)
print(f'{len(names)} known attractors')


def _integrateTask(args):
    i, name = args
    tA = getTrajectory(known[name], seed=1000 + i)
    tB = getTrajectory(known[name], seed=2000 + i)
    if tA is None or tB is None:
        return name, None
    return name, (tA[1], tB[1])


t0 = time.time()
trajectories = {}
failed = []
with Pool() as pool:
    tasks = list(enumerate(names))
    for name, result in tqdm(pool.imap(_integrateTask, tasks), total=len(tasks),
                              desc='Integrating known attractors (2 runs each)'):
        if result is None:
            failed.append(name)
        else:
            trajectories[name] = result
print(f'Integrated {len(trajectories)} attractors in {time.time()-t0:.1f}s'
      + (f' - failed: {failed}' if failed else ''))

reliableNames = sorted(trajectories.keys())


def _icpCanonSelfTask(name):
    runA, runB = trajectories[name]
    residual, _, _ = icpAlignCanonical(runA.T, runB.T, seed=42)
    return residual


def _icpCanonCrossTask(pair):
    nameA, nameB = pair
    residual, _, _ = icpAlignCanonical(trajectories[nameA][0].T, trajectories[nameB][0].T, seed=42)
    return residual


t0 = time.time()
with Pool() as pool:
    icpCanonSelf = list(tqdm(pool.imap(_icpCanonSelfTask, reliableNames), total=len(reliableNames),
                             desc='Canonical ICP self-distances'))
print(f'Canonical ICP self-distances ({len(icpCanonSelf)} pairs) in {time.time()-t0:.1f}s')

t0 = time.time()
crossPairs = list(itertools.combinations(reliableNames, 2))
with Pool() as pool:
    icpCanonCross = list(tqdm(pool.imap(_icpCanonCrossTask, crossPairs), total=len(crossPairs),
                              desc='Canonical ICP cross-distances'))
print(f'Canonical ICP cross-distances ({len(icpCanonCross)} pairs) in {time.time()-t0:.1f}s')


def analyze(label, selfD, crossD):
    selfD, crossD = np.array(selfD), np.array(crossD)
    meanSelf, meanCross = selfD.mean(), crossD.mean()
    combined = np.concatenate([selfD, crossD])
    ranks = np.argsort(np.argsort(combined)) + 1
    n1, n2 = len(selfD), len(crossD)
    rankSumCross = ranks[n1:].sum()
    auc = (rankSumCross - n2 * (n2 + 1) / 2) / (n1 * n2)
    threshold = meanSelf + 0.25 * (meanCross - meanSelf)
    falsePositive = (selfD > threshold).mean()
    falseNegative = (crossD <= threshold).mean()
    print(f'\n{label}:')
    print(f'  meanSelf={meanSelf:.4f}  meanCross={meanCross:.4f}  ratio={meanCross/meanSelf:.2f}x')
    print(f'  AUC = {auc:.4f}')
    print(f'  at threshold={threshold:.4f}: falsePositive={falsePositive:.1%}, falseNegative={falseNegative:.1%}')


analyze('Canonical ICP (PCA-aligned, 4 sign restarts)', icpCanonSelf, icpCanonCross)

old = np.load(os.path.join(SCRIPT_DIR, 'validationResults.npz'), allow_pickle=True)
analyze('Original ICP (8 random restarts) - for comparison', old['icpSelf'], old['icpCross'])

np.savez(os.path.join(SCRIPT_DIR, 'canonicalICPResults.npz'),
         names=reliableNames, icpCanonSelf=icpCanonSelf, icpCanonCross=icpCanonCross)
print('\nSaved to perceptualSimilarity/canonicalICPResults.npz')