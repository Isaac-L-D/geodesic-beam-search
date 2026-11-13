import os
import sys
import time
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
from noveltyCascade import zernikeDescriptor, zernikeDistanceBatch, SHORTLIST_K, ICP_RESTARTS
from icpAlignment import icpAlign

# Tests the cascade the way production actually uses it: a candidate compared against a whole
# reference CATALOG, not just one other trajectory - unlike validateMomentMethods.py/
# validatePerceptualMethods.py's plain pairwise self/cross. Two scenarios per known attractor:
#   - self:      runB queried against the full 36-reference catalog (runA of everyone,
#                including its own). Recall@K and top-1 correctness measure whether the Zernike
#                shortlist step ever drops the true match, and whether ICP within that shortlist
#                then correctly identifies it - not just whether the final number is small.
#   - leave-one-out: runB queried against the catalog with its OWN reference removed, so no
#                correct match exists - the real "is this genuinely new" case. The residual this
#                produces plays the role of a cross-distance in the AUC/threshold analysis below.

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


# No PyTorch anywhere here (renderAllViews + zernikeFeatures is pure matplotlib/skimage), so
# this is fork-parallelize-safe like the other moment-based validations.
def _descriptorTask(name):
    runA, runB = trajectories[name]
    return name, (zernikeDescriptor(runA), zernikeDescriptor(runB))


t0 = time.time()
zernikeDescs = {}
with Pool() as pool:
    for name, result in tqdm(pool.imap(_descriptorTask, reliableNames), total=len(reliableNames),
                              desc='Computing Zernike descriptors (2 runs each)'):
        zernikeDescs[name] = result
print(f'Computed descriptors for {len(zernikeDescs)} attractors in {time.time()-t0:.1f}s')

# Reference catalog = runA (points) + Zernike descriptor of runA, for every attractor.
refPointsA = {n: trajectories[n][0].T for n in reliableNames}
refDescA = {n: zernikeDescs[n][0] for n in reliableNames}


def _cascadeAgainstCatalog(candidatePoints, candidateDesc, catalogNames):
    refDescs = [refDescA[n] for n in catalogNames]
    zernikeDists = zernikeDistanceBatch(candidateDesc, refDescs)
    shortlistIdx = np.argsort(zernikeDists)[:SHORTLIST_K]
    shortlistNames = [catalogNames[i] for i in shortlistIdx]

    bestResidual, bestName = np.inf, None
    for refName in shortlistNames:
        residual, _, _ = icpAlign(candidatePoints, refPointsA[refName], nRestarts=ICP_RESTARTS)
        if residual < bestResidual:
            bestResidual, bestName = residual, refName
    return bestResidual, bestName, shortlistNames


def _queryTask(name):
    candidatePoints = trajectories[name][1].T
    candidateDesc = zernikeDescs[name][1]

    selfResidual, selfBestName, selfShortlist = _cascadeAgainstCatalog(
        candidatePoints, candidateDesc, reliableNames)

    looCatalog = [n for n in reliableNames if n != name]
    looResidual, looBestName, _ = _cascadeAgainstCatalog(
        candidatePoints, candidateDesc, looCatalog)

    return name, {
        'selfResidual': selfResidual,
        'selfRecall': name in selfShortlist,
        'selfTop1Correct': selfBestName == name,
        'looResidual': looResidual,
        'looBestName': looBestName,
    }


t0 = time.time()
results = {}
with Pool() as pool:
    for name, result in tqdm(pool.imap(_queryTask, reliableNames), total=len(reliableNames),
                              desc='Cascade: self + leave-one-out queries'):
        results[name] = result
print(f'Ran cascade queries for {len(results)} attractors in {time.time()-t0:.1f}s')

recallAtK = np.mean([r['selfRecall'] for r in results.values()])
top1Accuracy = np.mean([r['selfTop1Correct'] for r in results.values()])
print(f'\nRecall@{SHORTLIST_K} (true match survives the Zernike shortlist): {recallAtK:.1%}')
print(f'Top-1 accuracy (ICP within shortlist picks the true match): {top1Accuracy:.1%}')

wrongTop1 = [(n, r['looBestName']) for n, r in results.items() if not r['selfTop1Correct']]
if wrongTop1:
    print(f'Top-1 misses: {wrongTop1}')

selfD = np.array([r['selfResidual'] for r in results.values()])
crossD = np.array([r['looResidual'] for r in results.values()])


def analyze(label, selfD, crossD):
    meanSelf, meanCross = selfD.mean(), crossD.mean()
    combined = np.concatenate([selfD, crossD])
    ranks = np.argsort(np.argsort(combined)) + 1
    n1, n2 = len(selfD), len(crossD)
    auc = (ranks[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2)
    threshold = meanSelf + 0.25 * (meanCross - meanSelf)
    falsePositive = (selfD > threshold).mean()
    falseNegative = (crossD <= threshold).mean()
    print(f'\n{label}:')
    print(f'  meanSelf={meanSelf:.4f}  meanCross={meanCross:.4f}  ratio={meanCross/meanSelf:.2f}x')
    print(f'  AUC = {auc:.4f}')
    print(f'  at threshold={threshold:.4f}: falsePositive={falsePositive:.1%}, falseNegative={falseNegative:.1%}')


analyze('Cascade (Zernike shortlist -> ICP refine), catalog-style', selfD, crossD)

np.savez(os.path.join(SCRIPT_DIR, 'cascadeValidationResults.npz'),
         names=reliableNames, selfResidual=selfD, looResidual=crossD,
         selfRecall=[results[n]['selfRecall'] for n in reliableNames],
         selfTop1Correct=[results[n]['selfTop1Correct'] for n in reliableNames])
print('\nSaved to perceptualSimilarity/cascadeValidationResults.npz')