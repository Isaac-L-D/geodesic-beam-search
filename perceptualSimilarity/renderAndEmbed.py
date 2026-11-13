import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

# Renders a trajectory to a clean silhouette image from several fixed camera angles, then embeds
# each with a pretrained ImageNet CNN (ResNet18's penultimate layer - the standard "perceptual
# similarity" trick LPIPS and others are built on: features from a network trained for object
# recognition correlate with human visual similarity judgments far better than raw pixels do).
# Multiple angles because a single fixed camera view biases the comparison toward whatever
# orientation the two trajectories happen to share - see the rotation-invariance discussion this
# session had about curvature.

CAMERA_ANGLES = [(20, 30), (20, 120), (20, 210), (20, 300), (60, 45), (60, 225)]
IMAGE_SIZE = 224  # ResNet's expected input size

_model = None
_preprocess = None


def _loadModel():
    global _model, _preprocess
    if _model is not None:
        return
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    net = torchvision.models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()  # strip the classifier - keep the 512-dim pooled features
    net.eval()
    _model = net
    _preprocess = weights.transforms()


def renderTrajectoryImage(y, elev, azim):
    fig = plt.figure(figsize=(3, 3), dpi=IMAGE_SIZE // 3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[0], y[1], y[2], color='black', linewidth=0.6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def renderAllViews(y, angles=CAMERA_ANGLES):
    return [renderTrajectoryImage(y, elev, azim) for elev, azim in angles]


# Takes already-rendered images so callers comparing this against another method (e.g. SSIM) on
# the same trajectories only pay the matplotlib rendering cost once, not once per method.
def embedImages(images):
    _loadModel()
    batch = torch.stack([_preprocess(img) for img in images])
    with torch.no_grad():
        features = _model(batch)
    return features.numpy()  # (nAngles, 512)


def embedTrajectory(y, angles=CAMERA_ANGLES):
    return embedImages(renderAllViews(y, angles))


def cosineSimilarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Per-angle similarity plus the average - reported separately since a shape that matches well from
# some angles but not others is itself informative (e.g. mirror-image shapes, or a shared "main
# body" with a differently-oriented small feature).
def multiViewSimilarity(yA, yB, angles=CAMERA_ANGLES):
    featsA = embedTrajectory(yA, angles)
    featsB = embedTrajectory(yB, angles)
    perAngle = [cosineSimilarity(featsA[i], featsB[i]) for i in range(len(angles))]
    return {
        'meanSimilarity': float(np.mean(perAngle)),
        'maxSimilarity': float(np.max(perAngle)),
        'minSimilarity': float(np.min(perAngle)),
        'perAngle': perAngle,
    }