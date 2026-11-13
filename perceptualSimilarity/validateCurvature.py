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

from attractorSimilarity import loadDatasetAttractors, curvatureDescriptor, attractorDistanceBatch
import sharedConstants as ls
from getTrajectory import getTrajectory

# Head-to-head with validateCascade.py, applied to curvature - the CURRENT production novelty
# descriptor - using the identical retrieval scenario (self: true match present in the
# 36-reference catalog; leave-one-out: true match removed, simulating a genuinely new
# discovery), so the two are directly comparable rather than just similarly shaped. Curvature
# is single-stage (no shortlist+refine split), so "top-1" here is just the argmin over the whole
# catalog via the real attractorDistanceBatch production function.
#
# Uses the same getTrajectory-based trajectories as every other perceptualSimilarity/
# validation this session, not the full production arc-length integration pipeline - isolates
# the comparison to descriptor+distance method, not a different data-generation process.

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


# curvatureDescriptor expects points as (N, 3) - trajectories store the raw (3, N) shape
# getTrajectory returns, so transpose here (same convention icpAlign's callers use).
def _descriptorTask(name):
    runA, runB = trajectories[name]
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    descA = curvatureDescriptor(known[name], runA.T, rng)
    descB = curvatureDescriptor(known[name], runB.T, rng)
    return name, (descA, descB)


t0 = time.time()
curvDescs = {}
with Pool() as pool:
    for name, result in tqdm(pool.imap(_descriptorTask, reliableNames), total=len(reliableNames),
                              desc='Computing curvature descriptors (2 runs each)'):
        curvDescs[name] = result
print(f'Computed descriptors for {len(curvDescs)} attractors in {time.time()-t0:.1f}s')

refDescA = {n: curvDescs[n][0] for n in reliableNames}


def _queryAgainstCatalog(candidateDesc, catalogNames, k=5):
    refDescs = np.array([refDescA[n] for n in catalogNames])
    dists = attractorDistanceBatch(candidateDesc, refDescs)
    order = np.argsort(dists)
    topK = [catalogNames[i] for i in order[:k]]
    return dists[order[0]], catalogNames[order[0]], topK


def _queryTask(name):
    candidateDesc = curvDescs[name][1]

    selfDist, selfName, selfTop5 = _queryAgainstCatalog(candidateDesc, reliableNames)
    looCatalog = [n for n in reliableNames if n != name]
    looDist, looName, _ = _queryAgainstCatalog(candidateDesc, looCatalog)

    return name, {
        'selfResidual': selfDist,
        'selfRecall5': name in selfTop5,
        'selfTop1Correct': selfName == name,
        'looResidual': looDist,
        'looBestName': looName,
    }


t0 = time.time()
results = {}
with Pool() as pool:
    for name, result in tqdm(pool.imap(_queryTask, reliableNames), total=len(reliableNames),
                              desc='Curvature: self + leave-one-out queries'):
        results[name] = result
print(f'Ran curvature queries for {len(results)} attractors in {time.time()-t0:.1f}s')

recallAt5 = np.mean([r['selfRecall5'] for r in results.values()])
top1Accuracy = np.mean([r['selfTop1Correct'] for r in results.values()])
print(f'\nRecall@5: {recallAt5:.1%}')
print(f'Top-1 accuracy: {top1Accuracy:.1%}')

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


analyze('Curvature (current production descriptor), catalog-style', selfD, crossD)

np.savez(os.path.join(SCRIPT_DIR, 'curvatureCascadeComparisonResults.npz'),
         names=reliableNames, selfResidual=selfD, looResidual=crossD,
         selfRecall5=[results[n]['selfRecall5'] for n in reliableNames],
         selfTop1Correct=[results[n]['selfTop1Correct'] for n in reliableNames])
print('\nSaved to perceptualSimilarity/curvatureCascadeComparisonResults.npz')