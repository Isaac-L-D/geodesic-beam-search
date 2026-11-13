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
os.chdir(PROJECT_ROOT)  # ODEClass.py opens selectedStrangeAttractors.csv via a relative path

from attractorSimilarity import loadDatasetAttractors
import sharedConstants as ls
from getTrajectory import getTrajectory
from renderAndEmbed import renderAllViews, embedImages, cosineSimilarity, CAMERA_ANGLES
from hashSimilarity import hashImages, hashDistanceFromHashes
from icpAlignment import icpAlign

# Same validation methodology attractorShapeAnalysis/curvatureValidation.py used for curvature vs
# D2 (36 reliable known attractors, self-distance = two independent runs of the SAME attractor,
# cross-distance = different attractors, separation ratio = meanCross/meanSelf), applied here to
# the two perceptual-similarity methods from this session's investigation instead.

known = loadDatasetAttractors()
names = sorted(n for n in known if n not in ls.EXCLUDED_ATTRACTOR_NAMES)
print(f'{len(names)} known attractors')

# Independent per attractor, same as the ICP tasks below - `known` is a module-level global
# already populated before the Pool forks, so workers inherit it for free instead of needing it
# shipped through IPC. No PyTorch involved here (unlike the embedding step below), so fork-based
# Pool() is safe.
def _integrateTask(args):
    i, name = args
    tA = getTrajectory(known[name], seed=1000 + i)
    tB = getTrajectory(known[name], seed=2000 + i)
    if tA is None or tB is None:
        return name, None
    return name, (tA[1], tB[1])


t0 = time.time()
trajectories = {}  # name -> (runA_points, runB_points), each (3,N)
failed = []
with Pool() as pool:
    tasks = list(enumerate(names))
    for name, result in tqdm(pool.imap(_integrateTask, tasks), total=len(tasks),
                              desc='Integrating known attractors (2 runs each)'):
        if result is None:
            failed.append(name)
        else:
            trajectories[name] = result
print(f'Integrated {len(trajectories)} attractors (2 runs each) in {time.time()-t0:.1f}s'
      + (f' - failed: {failed}' if failed else ''))

reliableNames = sorted(trajectories.keys())

# --- Render once per trajectory, reused for both embedding and hashing (rendering, not the
# comparison itself, is the real per-trajectory cost) - each of embeddings/hashes is then a
# per-trajectory summary that's cheap to compare for every pair, no PyTorch in the hash path. ---
t0 = time.time()
embeddings = {}  # name -> (embA, embB), each (nAngles, 512)
hashes = {}  # name -> (hashA, hashB), each a list of nAngles imagehash objects
for name in tqdm(reliableNames, desc='Rendering + embedding + hashing (2 runs each)'):
    runA, runB = trajectories[name]
    imagesA = renderAllViews(runA, CAMERA_ANGLES)
    imagesB = renderAllViews(runB, CAMERA_ANGLES)
    embeddings[name] = (embedImages(imagesA), embedImages(imagesB))
    hashes[name] = (hashImages(imagesA), hashImages(imagesB))
print(f'Rendered + embedded + hashed {len(embeddings)} attractors (2 runs each) in {time.time()-t0:.1f}s')


def embDistance(embX, embY):
    sims = [cosineSimilarity(embX[i], embY[i]) for i in range(len(embX))]
    return 1.0 - float(np.mean(sims))


embSelf = [embDistance(embeddings[n][0], embeddings[n][1]) for n in reliableNames]
hashSelf = [hashDistanceFromHashes(hashes[n][0], hashes[n][1])[0] for n in reliableNames]

t0 = time.time()
embCross, hashCross = [], []
for nameA, nameB in itertools.combinations(reliableNames, 2):
    embCross.append(embDistance(embeddings[nameA][0], embeddings[nameB][0]))
    hashCross.append(hashDistanceFromHashes(hashes[nameA][0], hashes[nameB][0])[0])
print(f'Embedding + hash cross-distances ({len(embCross)} pairs) in {time.time()-t0:.1f}s')

# --- ICP: must be recomputed per pair (no reusable summary, unlike the embeddings above) - but
# every pair is independent of every other, so this is embarrassingly parallel. Worker functions
# read `trajectories` as a module-level global rather than taking it as an argument: Pool() forks
# on Linux, so worker processes inherit a copy-on-write view of it for free at fork time instead
# of needing to serialize and ship the whole dict through IPC on every single task.
def _icpSelfTask(name):
    runA, runB = trajectories[name]
    residual, _, _ = icpAlign(runA.T, runB.T, nRestarts=8, seed=42)
    return residual


def _icpCrossTask(pair):
    nameA, nameB = pair
    residual, _, _ = icpAlign(trajectories[nameA][0].T, trajectories[nameB][0].T, nRestarts=8, seed=42)
    return residual


t0 = time.time()
with Pool() as pool:
    icpSelf = list(tqdm(pool.imap(_icpSelfTask, reliableNames), total=len(reliableNames),
                         desc='ICP self-distances'))
print(f'ICP self-distances ({len(icpSelf)} pairs) in {time.time()-t0:.1f}s')

t0 = time.time()
crossPairs = list(itertools.combinations(reliableNames, 2))
with Pool() as pool:
    icpCross = list(tqdm(pool.imap(_icpCrossTask, crossPairs), total=len(crossPairs),
                          desc='ICP cross-distances'))
print(f'ICP cross-distances ({len(icpCross)} pairs) in {time.time()-t0:.1f}s')


def report(label, selfDists, crossDists):
    meanSelf, meanCross = np.mean(selfDists), np.mean(crossDists)
    print(f'\n{label}:')
    print(f'  meanSelf  = {meanSelf:.4f} (std {np.std(selfDists):.4f})')
    print(f'  meanCross = {meanCross:.4f} (std {np.std(crossDists):.4f})')
    print(f'  separation ratio (cross/self) = {meanCross/meanSelf:.2f}x')


report('Image-embedding distance (1 - cosine similarity)', embSelf, embCross)
report('pHash distance (normalized Hamming)', hashSelf, hashCross)
report('ICP alignment residual', icpSelf, icpCross)

np.savez(os.path.join(SCRIPT_DIR, 'validationResults.npz'),
         names=reliableNames, embSelf=embSelf, embCross=embCross,
         hashSelf=hashSelf, hashCross=hashCross, icpSelf=icpSelf, icpCross=icpCross)
print('\nSaved raw distances to perceptualSimilarity/validationResults.npz')