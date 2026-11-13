# Geodesic Beam Search

A modular method for discovering novel chaotic attractors in polynomial vector field space.

Starting from a curated set of known chaotic systems represented as unit vectors in an
81-dimensional coefficient space, Geodesic Beam Search iteratively expands a frontier of
"seed" systems by stepping out along equiangular tangent directions via exact geodesics,
discards candidates too close to already-cataloged territory, and keeps only the top-ranked
most chaotic new discoveries each round as the next round's seeds. The search framework is
decoupled from the two decision procedures it depends on &mdash; a **chaos oracle** (classifies
whether a candidate is genuinely chaotic and scores its fitness) and a **novelty oracle**
(decides whether a chaotic hit is geometrically distinct from everything cataloged so far)
&mdash; so either can be swapped out without touching the search mechanics.

Applying the method for 15 rounds, seeded from 36 known attractors, we evaluate roughly
394,000 candidate systems, identify about 27,500 chaotic hits, and confirm 12,200 as
geometrically distinct from every other known or discovered system.

Full method, results, and discussion: [`paper/geodesicBeamSearch.pdf`](paper/geodesicBeamSearch.pdf)

**Isaac Dash** (New York University), **Dr. William Hahn** (Florida Atlantic University)

## Data

The discovery catalog from the 15-round run &mdash; each hit's polynomial coefficients, fitness
score, and parent lineage &mdash; is in [`geodesicBeamSearch/output/allUniqueHits.csv`](geodesicBeamSearch/output/allUniqueHits.csv).

Per-round saved trajectories and explored tangent directions are too large for this repository
and are archived separately on Zenodo:

**[10.5281/zenodo.23856993](https://doi.org/10.5281/zenodo.23856993)**

## Repository structure

- **`geodesicBeamSearch/`** &mdash; the search implementation (`driver.py`, `engine.py`), its
  round-by-round output, and the diagnostic plots in this README's companion paper.
- **`paper/`** &mdash; LaTeX source and compiled PDF.
- **`perceptualSimilarity/`** &mdash; the novelty oracle (Zernike-moment/ICP shape-alignment)
  and its validation scripts.
- **`calibrationScripts/`** &mdash; one-off calibration/validation scripts for the chaos oracle
  and search parameters (arc-length, EMA thresholds, timeouts, etc.), kept for reproducibility.
- **`attractorShapeAnalysis/`** &mdash; shape-descriptor and similarity-metric code shared by
  the novelty oracle.
- **`ODEClass.py`**, **`DatasetPreprocessing.py`**, **`attractorDataset.csv`**,
  **`selectedStrangeAttractors.csv`** &mdash; the known-attractor dataset and the ODE
  integration class used throughout.

## Running the search

```bash
pip install -r requirements.txt
cd geodesicBeamSearch
python3 driver.py
```

This resumes from a checkpoint in `output/checkpoints/` if one exists, or starts a fresh
15-round run seeded from the known attractors otherwise.

## Dashboards

Three interactive [Dash](https://dash.plotly.com/) apps for browsing the data, each run with
`python3 <script>.py` and viewed at `localhost:8050`:

- **`makeDiscoveryDashboard.py`** &mdash; browse the discovery catalog: inspect any hit's
  trajectory, equation, and nearest cataloged neighbor.
- **`makeDisagreementDashboard.py`** &mdash; inspect cases where the chaos and novelty oracles'
  intermediate signals disagreed.
- **`makeDatasetDashboard.py`** &mdash; browse the curated known-attractor seed dataset.
