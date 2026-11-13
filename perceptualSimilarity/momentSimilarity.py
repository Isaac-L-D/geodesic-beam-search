import numpy as np
import mahotas

# Classical (non-learned) image-moment shape descriptor - a closed-form mathematical
# construction, the same rigor category as curvature/D2, not a trained model. Computed on the
# same rendered silhouette images as the other methods (see renderAndEmbed.renderAllViews), so
# rendering cost is shared, not paid again per method.

ZERNIKE_DEGREE = 12


def _toGray(img):
    return np.asarray(img.convert('L'), dtype=np.float64) / 255.0


# Zernike moment magnitudes - orthogonal polynomial basis over a disk; taking the magnitude
# (not phase) discards rotation information by construction.
def zernikeFeatures(img, degree=ZERNIKE_DEGREE):
    gray = (_toGray(img) * 255).astype(np.uint8)
    radius = min(gray.shape) // 2
    return np.array(mahotas.features.zernike_moments(gray, radius, degree=degree))


def _distance(featX, featY):
    return float(np.linalg.norm(featX - featY))


def momentDistances(imagesA, imagesB):
    zerA = [zernikeFeatures(img) for img in imagesA]
    zerB = [zernikeFeatures(img) for img in imagesB]
    return float(np.mean([_distance(zerA[i], zerB[i]) for i in range(len(imagesA))]))
