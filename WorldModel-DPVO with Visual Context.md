# WorldModel-DPVO with Visual Context

## Objective

Augment DPVO with a predictive world model that combines visual context, patch geometry, and temporal history.

The world model does **not replace DPVO matching**:

\[
\boxed{\text{predict} \rightarrow \text{measure} \rightarrow \text{correct}}
\]

- **World model:** predicts future visual-geometric state
- **DPVO tracker:** verifies predictions against the next image
- **Bundle adjustment:** produces the final geometrically consistent estimate

## Architecture

```text
Images up to time t
        │
        ├── DPVO encoder ──► local tracking features
        └── DINO/ResNet ───► contextual visual features
                              │
Past poses + depths + patch trajectories
                              │
                              ▼
                       World Model
                              │
           pose / patch / visibility / dynamics
                         predictions
                              │
                              ▼
Frame t+1 ─────────► DPVO correlation and update
                              │
                              ▼
                    Differentiable BA
                              │
                              ▼
               corrected pose, depth, and map
                              │
                              └──► world-model memory
```

## World-Model Input

For each patch \(i\), construct a visual-geometric token:

\[
q_i^t =
[
X_i^W,\,
p_i^t,\,
d_i^t,\,
f_i^{local},\,
f_i^{context},\,
v_i^t,\,
c_i^t
].
\]

Where:

- \(X_i^W\): estimated 3D world position
- \(p_i^t\): image location
- \(d_i^t\): depth
- \(f_i^{local}\): DPVO tracking feature
- \(f_i^{context}\): DINO or ResNet context feature
- \(v_i^t\): visibility state
- \(c_i^t\): confidence or uncertainty

The model also receives recent camera poses and global image tokens.

## Prediction

Using observations only up to time \(t\):

\[
\mathcal{W}_\theta
(F_{t-k:t},Q_{t-k:t},T_{t-k:t})
\rightarrow
\hat Q_{t+1},\Delta\hat T_{t+1}.
\]

Possible outputs include:

\[
\hat p_i^{t+1},\quad
\hat X_i^{t+1},\quad
P(v_i^{t+1}),\quad
P(\text{dynamic}),\quad
\hat f_i^{t+1},\quad
\Sigma_i.
\]

These outputs provide priors for DPVO tracking.

## Visual Features

Use two complementary representations:

- **DPVO features:** precise local correspondence and subpixel tracking
- **DINO/ResNet features:** semantic context, object identity, illumination robustness, and long-term re-identification

The contextual encoder should initially be pretrained and frozen.

## DPVO Correction

When frame \(t+1\) arrives, DPVO searches around the predicted location:

\[
\hat p_i^{WM},\quad \Sigma_i.
\]

The correlation and update networks determine the image-supported correction:

\[
p_i^{target}
=
\hat p_i^{WM}+\delta p_i.
\]

Bundle adjustment then optimizes camera poses and patch depths. World-model predictions remain uncertainty-weighted priors rather than hard constraints.

## Intended Benefit

The architecture targets cases where local frame-to-frame tracking fails:

- temporary occlusion
- dropped frames
- rapid motion
- motion blur
- dynamic objects
- illumination changes
- patch reappearance after long gaps

## Core Decomposition

\[
\boxed{\text{Visual encoder}=\text{scene appearance and context}}
\]

\[
\boxed{\text{DPVO state}=\text{geometric memory}}
\]

\[
\boxed{\text{World model}=\text{predictive temporal memory}}
\]

\[
\boxed{\text{Correlation + BA}=\text{measurement and correction}}
\]