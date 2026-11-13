import numpy as np
from skimage.metrics import structural_similarity

# Classical, non-learned alternative to the ResNet-embedding approach in renderAndEmbed.py -
# structural similarity (SSIM) compares local luminance/contrast/structure patterns rather than
# raw pixels, the standard perceptual-quality metric before CNN-embedding methods (what LPIPS is
# built on) became common. No PyTorch involved - pure numpy/scipy, so unlike the embedding
# approach this is safe to fork-parallelize with multiprocessing.Pool without caveats. Takes
# already-rendered images (see renderAndEmbed.renderAllViews) so rendering cost is paid once and
# shared between whichever perceptual methods need it, not once per method.


def _toGray(img):
    return np.asarray(img.convert('L'), dtype=np.float64) / 255.0


def ssimDistance(imagesA, imagesB):
    perAngle = [structural_similarity(_toGray(a), _toGray(b), data_range=1.0)
                for a, b in zip(imagesA, imagesB)]
    return 1.0 - float(np.mean(perAngle)), perAngle