import numpy as np
import imagehash

# Another classical, non-learned alternative - perceptual hashing (pHash: downsampled DCT
# coefficients, robust to small pixel-level shifts unlike a literal pixel/crypto hash) reduces each
# image to a compact bit pattern, compared via Hamming distance. Pure numpy/PIL under the hood, no
# PyTorch, so fork-parallelize-safe like SSIM. Takes already-rendered images (see
# renderAndEmbed.renderAllViews) so rendering cost is shared, not paid again per method.

HASH_SIZE = 16  # imagehash default is 8 (64-bit hash); 16 (256-bit) keeps more shape detail


# Like embedImages in renderAndEmbed.py, the hash is a per-image summary that can be computed once
# per trajectory and cached - comparing two cached hash sets is then just cheap Hamming distance,
# no need to redo per pair (unlike ICP).
def hashImages(images, hashSize=HASH_SIZE):
    return [imagehash.phash(img, hash_size=hashSize) for img in images]


def hashDistanceFromHashes(hashesA, hashesB, hashSize=HASH_SIZE):
    maxBits = hashSize * hashSize
    perAngle = [(hashesA[i] - hashesB[i]) / maxBits for i in range(len(hashesA))]  # normalized to [0,1]
    return float(np.mean(perAngle)), perAngle


def hashDistance(imagesA, imagesB, hashSize=HASH_SIZE):
    return hashDistanceFromHashes(hashImages(imagesA, hashSize), hashImages(imagesB, hashSize), hashSize)