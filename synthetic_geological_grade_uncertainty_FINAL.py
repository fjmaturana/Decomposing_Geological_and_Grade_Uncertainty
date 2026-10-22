"""
CONDITIONED SYNTHETIC CASE STUDY
Decomposing Geological and Grade Uncertainty in Mineral Resource Models


PURPOSE
-------
The experiment separates uncertainty in mineral-resource quantities into:

    U_grade(T)   = E_G[ Var(T | G, D) ]
    U_geology(T) = Var_G[ E(T | G, D) ]

using an explicitly nested design in which geological realisations form the
outer loop and conditional grade realisations form the inner loop.


WORKFLOW
--------
1. Generate a hidden folded tabular geological truth.
2. Generate a hidden spatially correlated copper-grade truth.
3. Sample the hidden truth with vertical synthetic drill sections at
   50, 100 and 200 m nominal spacing.
4. Generate geological realisations conditional on noisy upper and lower
   contact observations.
5. For every geological realisation, generate conditional grade realisations
   from the synthetic assay data.
6. Reclassify assay membership for every geological realisation so that
   grade conditioning depends on realised geology.
7. Calculate tonnes, average grade (% Cu) and contained metal (t Cu).
8. Apply the balanced finite-inner-simulation correction

       U_geo^c = S^2_{bar T} - U_grade / R.

9. Retain U_geo^c for inference and use max(0, U_geo^c) only for descriptive
   non-negative variance shares.
10. Produce factorial summaries, block-scale decompositions, sensitivity
    plots, cut-off curves, fixed-geology diagnostics and convergence outputs.

NOTES
-------------------------
Grade is stored numerically in percent Cu. Thus a grade value of 1.0 means
1.0% Cu. Physical contained copper is therefore calculated as

    tonnes Cu = tonnes * grade_pct_cu / 100.

No input files or command-line arguments are required.

Requirements:
    Python 3.10+
    numpy
    pandas
    matplotlib
"""

from pathlib import Path
import itertools
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Springer Nature artwork settings. DejaVu Sans is used explicitly because
# it embeds reliably in EPS/PDF output across Matplotlib, Ghostscript and
# pdfLaTeX workflows. The source lettering is deliberately larger than the
# minimum so that it remains within the journal's 8--12 pt recommendation
# after normal manuscript scaling.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11.0,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.1,
    "lines.markeredgewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 1. GLOBAL SETTINGS
# =============================================================================

SEED = 20260925
TRUTH_SEED = 20260924

NX = 48
NZ = 24
LX = 1200.0
LZ = 600.0

DX = LX / NX
DZ = LZ / NZ

X = np.linspace(DX / 2.0, LX - DX / 2.0, NX)
Z = np.linspace(DZ / 2.0, LZ - DZ / 2.0, NZ)
XX, ZZ = np.meshgrid(X, Z)

BLOCK_VOLUME = DX * DZ * 1.0
DENSITY = 2.7

BASE_CUTOFF = 0.50
CUTOFFS = np.arange(0.30, 1.21, 0.10)

N_GEO = 50
N_GRADE = 50

SPACINGS = (50.0, 100.0, 200.0)
BOUNDARY_SDS = (10.0, 30.0, 60.0)
CONTRASTS = (0.4, 0.8, 1.2)

BASELINE_SPACING = 100.0
BASELINE_BOUNDARY_SD = 30.0
BASELINE_CONTRAST = 0.8
BASELINE = (
    BASELINE_SPACING,
    BASELINE_BOUNDARY_SD,
    BASELINE_CONTRAST,
)

HALF_THICKNESS = 55.0
BACKGROUND_MEAN = 0.18
MIN_GRADE = 0.01

CONTACT_RANGE = 240.0
GRADE_RANGE_X = 220.0
GRADE_RANGE_Z = 90.0
GRADE_SD = 0.28

ASSAY_NOISE_SD = 0.025
CONTACT_OBS_NOISE_SD = 3.0

JITTER = 1.0e-8

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "synthetic_geological_grade_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. GENERAL UTILITIES
# =============================================================================

def exponential_covariance_1d(a, b, sill, range_):
    """One-dimensional exponential covariance using exp(-3h/range)."""
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    h = np.abs(a - b)
    return sill * np.exp(-3.0 * h / range_)


def exponential_covariance_2d(
    xa,
    za,
    xb,
    zb,
    sill,
    range_x,
    range_z,
):
    """Anisotropic two-dimensional exponential covariance."""
    xa = np.asarray(xa, dtype=float).reshape(-1, 1)
    za = np.asarray(za, dtype=float).reshape(-1, 1)
    xb = np.asarray(xb, dtype=float).reshape(1, -1)
    zb = np.asarray(zb, dtype=float).reshape(1, -1)

    h = np.sqrt(
        ((xa - xb) / range_x) ** 2
        + ((za - zb) / range_z) ** 2
    )
    return sill * np.exp(-3.0 * h)


def stable_cholesky(matrix, jitter=JITTER):
    """
    Numerically robust Cholesky factorisation.

    Increasing diagonal jitter is attempted first. An eigenvalue repair is
    used only as a final numerical safeguard.
    """
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)

    eye = np.eye(matrix.shape[0])
    scale = max(float(np.mean(np.diag(matrix))), 1.0)

    for multiplier in (1, 10, 100, 1000, 10000, 100000):
        try:
            return np.linalg.cholesky(
                matrix + eye * jitter * scale * multiplier
            )
        except np.linalg.LinAlgError:
            pass

    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, jitter * scale)
    repaired = (vectors * values) @ vectors.T
    repaired = 0.5 * (repaired + repaired.T)

    return np.linalg.cholesky(repaired)


def quantiles(values):
    """Return empirical P05, P50 and P95."""
    return np.quantile(
        np.asarray(values, dtype=float).ravel(),
        [0.05, 0.50, 0.95],
    )


# =============================================================================
# 3. HIDDEN SYNTHETIC TRUTH
# =============================================================================

def nominal_centreline(x):
    """Nominal folded centreline used by the geological model."""
    x = np.asarray(x, dtype=float)
    return (
        300.0
        + 65.0 * np.sin(2.0 * np.pi * x / 900.0)
        + 0.055 * (x - 600.0)
    )


def generate_hidden_truth():
    """
    Generate one fixed hidden truth.

    The hidden geological truth is a perturbed folded tabular body. The hidden
    grade truth is a spatially correlated standardised residual field that is
    combined with the scenario-specific mineralised/background mean contrast.
    """
    rng = np.random.default_rng(TRUTH_SEED)

    contact_cov = exponential_covariance_1d(
        X,
        X,
        sill=12.0 ** 2,
        range_=CONTACT_RANGE,
    )
    contact_departure = (
        stable_cholesky(contact_cov)
        @ rng.normal(size=NX)
    )

    true_centre = nominal_centreline(X) + contact_departure
    true_top = true_centre + HALF_THICKNESS
    true_bottom = true_centre - HALF_THICKNESS

    membership = (
        (ZZ <= true_top[np.newaxis, :])
        & (ZZ >= true_bottom[np.newaxis, :])
    )

    flat_x = XX.ravel()
    flat_z = ZZ.ravel()

    grade_cov = exponential_covariance_2d(
        flat_x,
        flat_z,
        flat_x,
        flat_z,
        sill=GRADE_SD ** 2,
        range_x=GRADE_RANGE_X,
        range_z=GRADE_RANGE_Z,
    )

    grade_residual = (
        stable_cholesky(grade_cov)
        @ rng.normal(size=flat_x.size)
    ).reshape(NZ, NX)

    return {
        "true_centre": true_centre,
        "true_top": true_top,
        "true_bottom": true_bottom,
        "membership": membership,
        "standard_grade_residual": grade_residual / GRADE_SD,
    }


TRUTH = generate_hidden_truth()


# =============================================================================
# 4. SYNTHETIC DRILLING
# =============================================================================

def drill_x_positions(spacing):
    """Return vertical synthetic drill-section locations."""
    start = spacing / 2.0
    return np.arange(start, LX, spacing, dtype=float)


def interpolate_hidden_grade_residual(xp, zp):
    """
    Bilinearly interpolate the hidden standardised grade residual field.
    """
    xp = np.asarray(xp, dtype=float)
    zp = np.asarray(zp, dtype=float)

    fx = (xp - X[0]) / DX
    fz = (zp - Z[0]) / DZ

    ix0 = np.floor(fx).astype(int)
    iz0 = np.floor(fz).astype(int)

    ix0 = np.clip(ix0, 0, NX - 2)
    iz0 = np.clip(iz0, 0, NZ - 2)

    ix1 = ix0 + 1
    iz1 = iz0 + 1

    tx = np.clip(fx - ix0, 0.0, 1.0)
    tz = np.clip(fz - iz0, 0.0, 1.0)

    field = TRUTH["standard_grade_residual"]

    return (
        (1.0 - tx) * (1.0 - tz) * field[iz0, ix0]
        + tx * (1.0 - tz) * field[iz0, ix1]
        + (1.0 - tx) * tz * field[iz1, ix0]
        + tx * tz * field[iz1, ix1]
    )


def hidden_truth_grade_at_points(xp, zp, contrast):
    """
    Evaluate the hidden grade truth at arbitrary synthetic assay locations.

    Grade is returned in percent Cu.
    """
    xp = np.asarray(xp, dtype=float)
    zp = np.asarray(zp, dtype=float)

    top = np.interp(xp, X, TRUTH["true_top"])
    bottom = np.interp(xp, X, TRUTH["true_bottom"])

    inside = (zp <= top) & (zp >= bottom)

    prior_mean = np.where(
        inside,
        BACKGROUND_MEAN + contrast,
        BACKGROUND_MEAN,
    )

    residual = interpolate_hidden_grade_residual(xp, zp)
    grade = prior_mean + GRADE_SD * residual
    grade = np.maximum(grade, MIN_GRADE)

    return grade, inside


def create_drill_dataset(spacing, contrast, rng):
    """
    Sample the hidden truth with explicit vertical synthetic drill sections.

    Each section contributes:
      - one noisy upper-contact observation;
      - one noisy lower-contact observation;
      - approximately 25 m vertical grade samples through and around the body.
    """
    drill_x = drill_x_positions(spacing)

    top_true = np.interp(drill_x, X, TRUTH["true_top"])
    bottom_true = np.interp(drill_x, X, TRUTH["true_bottom"])

    top_obs = (
        top_true
        + rng.normal(
            0.0,
            CONTACT_OBS_NOISE_SD,
            size=drill_x.size,
        )
    )
    bottom_obs = (
        bottom_true
        + rng.normal(
            0.0,
            CONTACT_OBS_NOISE_SD,
            size=drill_x.size,
        )
    )

    assay_x = []
    assay_z = []

    for x_i, bottom_i, top_i in zip(
        drill_x,
        bottom_true,
        top_true,
    ):
        z_start = max(Z[0], bottom_i - 50.0)
        z_stop = min(Z[-1], top_i + 50.0)

        sample_z = np.arange(
            z_start,
            z_stop + 1.0e-9,
            25.0,
        )

        assay_x.extend([x_i] * len(sample_z))
        assay_z.extend(sample_z.tolist())

    assay_x = np.asarray(assay_x, dtype=float)
    assay_z = np.asarray(assay_z, dtype=float)

    assay_grade, assay_inside_truth = hidden_truth_grade_at_points(
        assay_x,
        assay_z,
        contrast,
    )

    assay_grade = np.maximum(
        assay_grade
        + rng.normal(
            0.0,
            ASSAY_NOISE_SD,
            size=assay_grade.size,
        ),
        MIN_GRADE,
    )

    return {
        "contact_x": drill_x,
        "top_obs": top_obs,
        "bottom_obs": bottom_obs,
        "assay_x": assay_x,
        "assay_z": assay_z,
        "assay_grade": assay_grade,
        "assay_inside_truth": assay_inside_truth,
    }


# =============================================================================
# 5. CONDITIONAL GEOLOGICAL SIMULATION
# =============================================================================

def conditional_contact_draw(
    obs_x,
    obs_values,
    nominal_contact,
    boundary_sd,
    rng,
):
    """
    Draw one contact conditional on observed synthetic intersections.

    boundary_sd controls residual geological-model uncertainty. Drill spacing
    affects uncertainty through the number and geometry of observations.
    """
    obs_x = np.asarray(obs_x, dtype=float)
    obs_values = np.asarray(obs_values, dtype=float)
    nominal_contact = np.asarray(nominal_contact, dtype=float)

    nominal_obs = np.interp(
        obs_x,
        X,
        nominal_contact,
    )

    koo = exponential_covariance_1d(
        obs_x,
        obs_x,
        sill=boundary_sd ** 2,
        range_=CONTACT_RANGE,
    )
    koo.flat[:: koo.shape[0] + 1] += (
        CONTACT_OBS_NOISE_SD ** 2
    )

    kto = exponential_covariance_1d(
        X,
        obs_x,
        sill=boundary_sd ** 2,
        range_=CONTACT_RANGE,
    )

    ktt = exponential_covariance_1d(
        X,
        X,
        sill=boundary_sd ** 2,
        range_=CONTACT_RANGE,
    )

    residual = obs_values - nominal_obs

    alpha = np.linalg.solve(
        koo,
        residual,
    )
    conditional_mean = nominal_contact + kto @ alpha

    solved = np.linalg.solve(
        koo,
        kto.T,
    )
    conditional_covariance = ktt - kto @ solved
    conditional_covariance = 0.5 * (
        conditional_covariance
        + conditional_covariance.T
    )

    draw = (
        conditional_mean
        + stable_cholesky(conditional_covariance)
        @ rng.normal(size=NX)
    )

    return draw


def geological_realisation(drill, boundary_sd, rng):
    """Generate one conditioned upper/lower-contact realisation."""
    nominal_centre = nominal_centreline(X)
    nominal_top = nominal_centre + HALF_THICKNESS
    nominal_bottom = nominal_centre - HALF_THICKNESS

    top = conditional_contact_draw(
        drill["contact_x"],
        drill["top_obs"],
        nominal_top,
        boundary_sd,
        rng,
    )

    bottom = conditional_contact_draw(
        drill["contact_x"],
        drill["bottom_obs"],
        nominal_bottom,
        boundary_sd,
        rng,
    )

    # Enforce a positive minimum thickness while retaining the simulated
    # midpoint. This is a simple geological-validity constraint.
    midpoint = 0.5 * (top + bottom)
    thickness = np.maximum(top - bottom, 20.0)

    top = midpoint + 0.5 * thickness
    bottom = midpoint - 0.5 * thickness

    membership = (
        (ZZ <= top[np.newaxis, :])
        & (ZZ >= bottom[np.newaxis, :])
    )

    return membership, top, bottom


# =============================================================================
# 6. CONDITIONAL GRADE SIMULATION
# =============================================================================

def classify_assays_for_geology(drill, top, bottom):
    """
    Reclassify assay locations according to the current geological realisation.
    """
    top_at_assay = np.interp(
        drill["assay_x"],
        X,
        top,
    )
    bottom_at_assay = np.interp(
        drill["assay_x"],
        X,
        bottom,
    )

    return (
        (drill["assay_z"] <= top_at_assay)
        & (drill["assay_z"] >= bottom_at_assay)
    )


def grade_conditioning_system(
    drill,
    top,
    bottom,
    contrast,
):
    """
    Build the conditional Gaussian system for one geological realisation.

    The returned conditional mean and Cholesky factor are reused for all inner
    grade realisations within that geology.
    """
    flat_x = XX.ravel()
    flat_z = ZZ.ravel()

    block_membership = (
        (ZZ <= top[np.newaxis, :])
        & (ZZ >= bottom[np.newaxis, :])
    ).ravel()

    assay_inside = classify_assays_for_geology(
        drill,
        top,
        bottom,
    )

    target_mean = np.where(
        block_membership,
        BACKGROUND_MEAN + contrast,
        BACKGROUND_MEAN,
    )

    observation_mean = np.where(
        assay_inside,
        BACKGROUND_MEAN + contrast,
        BACKGROUND_MEAN,
    )

    obs_x = drill["assay_x"]
    obs_z = drill["assay_z"]
    obs_grade = drill["assay_grade"]

    koo = exponential_covariance_2d(
        obs_x,
        obs_z,
        obs_x,
        obs_z,
        sill=GRADE_SD ** 2,
        range_x=GRADE_RANGE_X,
        range_z=GRADE_RANGE_Z,
    )
    koo.flat[:: koo.shape[0] + 1] += ASSAY_NOISE_SD ** 2

    kto = exponential_covariance_2d(
        flat_x,
        flat_z,
        obs_x,
        obs_z,
        sill=GRADE_SD ** 2,
        range_x=GRADE_RANGE_X,
        range_z=GRADE_RANGE_Z,
    )

    ktt = exponential_covariance_2d(
        flat_x,
        flat_z,
        flat_x,
        flat_z,
        sill=GRADE_SD ** 2,
        range_x=GRADE_RANGE_X,
        range_z=GRADE_RANGE_Z,
    )

    alpha = np.linalg.solve(
        koo,
        obs_grade - observation_mean,
    )
    conditional_mean = target_mean + kto @ alpha

    solved = np.linalg.solve(
        koo,
        kto.T,
    )
    conditional_covariance = ktt - kto @ solved
    conditional_covariance = 0.5 * (
        conditional_covariance
        + conditional_covariance.T
    )

    chol = stable_cholesky(
        conditional_covariance
    )

    return (
        conditional_mean,
        chol,
        block_membership.reshape(NZ, NX),
    )


def draw_conditional_grade(
    conditional_mean,
    chol,
    rng,
):
    """Generate one conditional block-grade realisation in percent Cu."""
    draw = (
        conditional_mean
        + chol @ rng.normal(size=conditional_mean.size)
    )

    return np.maximum(
        draw.reshape(NZ, NX),
        MIN_GRADE,
    )


# =============================================================================
# 7. RESOURCE FUNCTIONALS
# =============================================================================

def resource_functionals(
    membership,
    grade_pct_cu,
    cutoff=BASE_CUTOFF,
):
    """
    Calculate resource quantities above cut-off.

    Returns
    -------
    tonnes : float
        Selected tonnes.
    average_grade_pct_cu : float
        Tonnage-weighted average grade in percent Cu. Because every block has
        equal tonnage in this experiment, this equals the arithmetic mean of
        selected block grades.
    contained_metal_t_cu : float
        Physical contained copper in tonnes Cu.

    Notes
    -----
    grade_pct_cu is stored in percent units. It is divided by 100 when
    converted to a mass fraction for contained-metal calculation.
    """
    selected = (
        np.asarray(membership, dtype=bool)
        & (np.asarray(grade_pct_cu, dtype=float) >= cutoff)
    )

    n_selected = int(selected.sum())

    if n_selected == 0:
        return 0.0, np.nan, 0.0

    block_tonnes = BLOCK_VOLUME * DENSITY
    tonnes = float(n_selected * block_tonnes)

    average_grade_pct_cu = float(
        np.mean(grade_pct_cu[selected])
    )

    contained_metal_t_cu = float(
        np.sum(
            block_tonnes
            * grade_pct_cu[selected]
            / 100.0
        )
    )

    return (
        tonnes,
        average_grade_pct_cu,
        contained_metal_t_cu,
    )


def cutoff_functionals(
    membership,
    grade_pct_cu,
    cutoffs=CUTOFFS,
):
    """Calculate resource quantities over a sequence of cut-offs."""
    output = np.zeros(
        (len(cutoffs), 3),
        dtype=float,
    )

    for i, cutoff in enumerate(cutoffs):
        output[i] = resource_functionals(
            membership,
            grade_pct_cu,
            cutoff,
        )

    return output


# =============================================================================
# 8. NESTED VARIANCE DECOMPOSITION
# =============================================================================

def nested_decomposition(values):
    """
    Balanced nested variance-component estimator.

    For values T_gr with G geological realisations and R grade realisations:

        U_grade_hat = mean_g s_g^2

        U_geo_raw_hat
            = Var_g(bar T_g) - U_grade_hat / R.

    The raw corrected geology component is retained for inference. A truncated
    non-negative component is used only for descriptive variance shares.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 2:
        raise ValueError(
            "values must be a two-dimensional array with shape (G, R)."
        )

    if np.isnan(values).any():
        raise ValueError(
            "NaN encountered in nested resource outputs."
        )

    _, n_inner = values.shape

    within_variances = np.var(
        values,
        axis=1,
        ddof=1,
    )
    geology_means = np.mean(
        values,
        axis=1,
    )

    u_grade = float(
        np.mean(within_variances)
    )

    s2_geology_means = float(
        np.var(
            geology_means,
            ddof=1,
        )
    )

    u_geology_raw = float(
        s2_geology_means
        - u_grade / n_inner
    )

    u_geology_report = max(
        0.0,
        u_geology_raw,
    )

    u_total_raw = (
        u_grade
        + u_geology_raw
    )
    u_total_report = (
        u_grade
        + u_geology_report
    )

    rg_raw = (
        u_geology_raw / u_total_raw
        if u_total_raw > 0.0
        else np.nan
    )

    rg_report = (
        u_geology_report / u_total_report
        if u_total_report > 0.0
        else np.nan
    )

    return {
        "U_grade": u_grade,
        "S2_geology_means": s2_geology_means,
        "U_geology_raw": u_geology_raw,
        "U_geology_report": u_geology_report,
        "U_total_raw": u_total_raw,
        "U_total_report": u_total_report,
        "R_G_raw": rg_raw,
        "R_G_report": rg_report,
    }


def block_decomposition(block_values):
    """
    Apply the same corrected nested decomposition independently to each block.

    block_values has shape (G, R, NZ, NX).
    """
    block_values = np.asarray(
        block_values,
        dtype=float,
    )

    if block_values.ndim != 4:
        raise ValueError(
            "block_values must have shape (G, R, NZ, NX)."
        )

    _, n_inner, _, _ = block_values.shape

    geology_means = np.mean(
        block_values,
        axis=1,
    )

    within_variances = np.var(
        block_values,
        axis=1,
        ddof=1,
    )

    u_grade = np.mean(
        within_variances,
        axis=0,
    )

    s2_geology_means = np.var(
        geology_means,
        axis=0,
        ddof=1,
    )

    u_geology_raw = (
        s2_geology_means
        - u_grade / n_inner
    )

    u_geology_report = np.maximum(
        0.0,
        u_geology_raw,
    )

    u_total_report = (
        u_grade
        + u_geology_report
    )

    rg_report = np.divide(
        u_geology_report,
        u_total_report,
        out=np.zeros_like(u_total_report),
        where=u_total_report > 0.0,
    )

    return {
        "U_grade": u_grade,
        "U_geology_raw": u_geology_raw,
        "U_geology_report": u_geology_report,
        "U_total_report": u_total_report,
        "R_G_report": rg_report,
    }


# =============================================================================
# 9. FIXED-GEOLOGY DIAGNOSTIC
# =============================================================================

def select_representative_geology(memberships):
    """
    Select the realised geology closest to the ensemble modal membership map.
    """
    memberships = np.asarray(
        memberships,
        dtype=bool,
    )

    membership_probability = np.mean(
        memberships,
        axis=0,
    )
    modal_membership = (
        membership_probability >= 0.5
    )

    disagreement = np.mean(
        memberships
        != modal_membership[np.newaxis, :, :],
        axis=(1, 2),
    )

    index = int(
        np.argmin(disagreement)
    )

    return (
        index,
        modal_membership,
    )


def fixed_to_full_variance_ratio(
    values,
    fixed_index,
    decomposition,
):
    """
    Ratio of variance conditional on one selected geology to full reported
    nested variance.
    """
    fixed_variance = float(
        np.var(
            values[fixed_index],
            ddof=1,
        )
    )

    full_variance = decomposition[
        "U_total_report"
    ]

    return (
        fixed_variance / full_variance
        if full_variance > 0.0
        else np.nan
    )


# =============================================================================
# 10. RUN ONE FACTORIAL SCENARIO
# =============================================================================

def run_scenario(
    spacing,
    boundary_sd,
    contrast,
    seed,
    retain_details=False,
    n_geo=N_GEO,
    n_grade=N_GRADE,
):
    """Run one complete nested geological/grade scenario."""
    rng = np.random.default_rng(
        int(seed)
    )

    drill = create_drill_dataset(
        spacing,
        contrast,
        rng,
    )

    memberships = np.zeros(
        (n_geo, NZ, NX),
        dtype=bool,
    )
    top_contacts = np.zeros(
        (n_geo, NX),
        dtype=np.float32,
    )
    bottom_contacts = np.zeros(
        (n_geo, NX),
        dtype=np.float32,
    )

    tonnes = np.zeros(
        (n_geo, n_grade),
        dtype=float,
    )
    average_grade = np.zeros(
        (n_geo, n_grade),
        dtype=float,
    )
    metal_t_cu = np.zeros(
        (n_geo, n_grade),
        dtype=float,
    )

    block_star = None
    cutoff_outputs = None

    if retain_details:
        block_star = np.zeros(
            (n_geo, n_grade, NZ, NX),
            dtype=np.float32,
        )
        cutoff_outputs = np.zeros(
            (
                n_geo,
                n_grade,
                len(CUTOFFS),
                3,
            ),
            dtype=np.float64,
        )

    for g in range(n_geo):
        membership, top, bottom = geological_realisation(
            drill,
            boundary_sd,
            rng,
        )

        memberships[g] = membership
        top_contacts[g] = top
        bottom_contacts[g] = bottom

        conditional_mean, chol, membership_check = (
            grade_conditioning_system(
                drill,
                top,
                bottom,
                contrast,
            )
        )

        if not np.array_equal(
            membership,
            membership_check,
        ):
            raise RuntimeError(
                "Internal geology membership mismatch."
            )

        for r in range(n_grade):
            grade = draw_conditional_grade(
                conditional_mean,
                chol,
                rng,
            )

            t, a, m = resource_functionals(
                membership,
                grade,
            )

            tonnes[g, r] = t
            average_grade[g, r] = a
            metal_t_cu[g, r] = m

            if retain_details:
                # Zero is used outside the realised resource domain as a
                # resource-membership convention for block-scale attribution.
                block_star[g, r] = np.where(
                    membership,
                    grade,
                    0.0,
                )

                cutoff_outputs[g, r] = (
                    cutoff_functionals(
                        membership,
                        grade,
                    )
                )

    if np.isnan(average_grade).any():
        raise RuntimeError(
            "At least one simulation produced zero tonnes above "
            "the baseline cut-off."
        )

    tonnes_dec = nested_decomposition(
        tonnes
    )
    grade_dec = nested_decomposition(
        average_grade
    )
    metal_dec = nested_decomposition(
        metal_t_cu
    )

    fixed_index, modal_geology = (
        select_representative_geology(
            memberships
        )
    )

    row = {
        "spacing_m": spacing,
        "boundary_sd_m": boundary_sd,
        "contrast_pct_cu": contrast,

        "Ugrade_tonnes": tonnes_dec["U_grade"],
        "Ugeology_raw_tonnes": tonnes_dec["U_geology_raw"],
        "Ugeology_report_tonnes": tonnes_dec["U_geology_report"],
        "Utotal_tonnes": tonnes_dec["U_total_report"],
        "RG_raw_tonnes": tonnes_dec["R_G_raw"],
        "RG_report_tonnes": tonnes_dec["R_G_report"],
        "fixed_full_tonnes": fixed_to_full_variance_ratio(
            tonnes,
            fixed_index,
            tonnes_dec,
        ),

        "Ugrade_avggrade": grade_dec["U_grade"],
        "Ugeology_raw_avggrade": grade_dec["U_geology_raw"],
        "Ugeology_report_avggrade": grade_dec["U_geology_report"],
        "Utotal_avggrade": grade_dec["U_total_report"],
        "RG_raw_avggrade": grade_dec["R_G_raw"],
        "RG_report_avggrade": grade_dec["R_G_report"],
        "fixed_full_avggrade": fixed_to_full_variance_ratio(
            average_grade,
            fixed_index,
            grade_dec,
        ),

        "Ugrade_metal_t_cu": metal_dec["U_grade"],
        "Ugeology_raw_metal_t_cu": metal_dec["U_geology_raw"],
        "Ugeology_report_metal_t_cu": metal_dec["U_geology_report"],
        "Utotal_metal_t_cu": metal_dec["U_total_report"],
        "RG_raw_metal": metal_dec["R_G_raw"],
        "RG_report_metal": metal_dec["R_G_report"],
        "fixed_full_metal": fixed_to_full_variance_ratio(
            metal_t_cu,
            fixed_index,
            metal_dec,
        ),

        "fixed_geology_index": fixed_index,
        "n_assays": len(drill["assay_grade"]),
        "n_drill_sections": len(drill["contact_x"]),
    }

    details = None

    if retain_details:
        details = {
            "drill": drill,
            "memberships": memberships,
            "top_contacts": top_contacts,
            "bottom_contacts": bottom_contacts,
            "tonnes": tonnes,
            "average_grade": average_grade,
            "metal_t_cu": metal_t_cu,
            "block_star": block_star,
            "cutoff_outputs": cutoff_outputs,
            "modal_geology": modal_geology,
            "fixed_index": fixed_index,
        }

    return row, details


# =============================================================================
# 11. CONVERGENCE DIAGNOSTIC
# =============================================================================

def convergence_from_full_arrays(
    values,
    g_values=(10, 20, 30, 40, 50),
    r_values=(10, 20, 30, 40, 50),
):
    """
    Recalculate the decomposition using leading G x R subsets of a completed
    nested simulation. This is a diagnostic, not an independent replication.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    g_max, r_max = values.shape
    rows = []

    for g in g_values:
        if g > g_max:
            continue

        for r in r_values:
            if r > r_max:
                continue

            decomposition = nested_decomposition(
                values[:g, :r]
            )

            rows.append(
                {
                    "G": g,
                    "R": r,
                    "U_grade": decomposition["U_grade"],
                    "U_geology_raw": decomposition["U_geology_raw"],
                    "U_geology_report": decomposition["U_geology_report"],
                    "R_G_raw": decomposition["R_G_raw"],
                    "R_G_report": decomposition["R_G_report"],
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 12. TABULAR OUTPUTS
# =============================================================================

def export_drill_dataset(
    drill,
    spacing,
):
    """Export synthetic contacts and assays for one nominal spacing."""
    contacts = pd.DataFrame(
        {
            "record_type": "contact",
            "x_m": np.repeat(
                drill["contact_x"],
                2,
            ),
            "z_m": np.ravel(
                np.column_stack(
                    [
                        drill["bottom_obs"],
                        drill["top_obs"],
                    ]
                )
            ),
            "grade_pct_cu": np.nan,
            "contact": np.tile(
                ["bottom", "top"],
                len(drill["contact_x"]),
            ),
        }
    )

    assays = pd.DataFrame(
        {
            "record_type": "assay",
            "x_m": drill["assay_x"],
            "z_m": drill["assay_z"],
            "grade_pct_cu": drill["assay_grade"],
            "contact": "",
        }
    )

    output = pd.concat(
        [contacts, assays],
        ignore_index=True,
    )

    output.to_csv(
        OUTDIR
        / f"synthetic_drill_data_{int(spacing)}m.csv",
        index=False,
    )


def baseline_quantile_table(details):
    """Create the baseline P05/P50/P95 resource table."""
    rows = []

    quantities = (
        ("Tonnes", details["tonnes"]),
        (
            "Average grade (% Cu)",
            details["average_grade"],
        ),
        (
            "Contained metal (t Cu)",
            details["metal_t_cu"],
        ),
    )

    for name, values in quantities:
        p05, p50, p95 = quantiles(
            values
        )

        rows.append(
            {
                "Quantity": name,
                "P05": p05,
                "P50": p50,
                "P95": p95,
            }
        )

    return pd.DataFrame(
        rows
    )


def baseline_cutoff_table(details):
    """Create baseline empirical cut-off curves."""
    array = details["cutoff_outputs"]
    rows = []

    names = (
        "Tonnes",
        "Average grade (% Cu)",
        "Contained metal (t Cu)",
    )

    for cutoff_index, cutoff in enumerate(
        CUTOFFS
    ):
        for quantity_index, quantity in enumerate(
            names
        ):
            values = array[
                :,
                :,
                cutoff_index,
                quantity_index,
            ].ravel()

            if np.isnan(values).all():
                p05 = np.nan
                p50 = np.nan
                p95 = np.nan
            else:
                p05, p50, p95 = np.nanquantile(
                    values,
                    [0.05, 0.50, 0.95],
                )

            rows.append(
                {
                    "cutoff_pct_cu": cutoff,
                    "quantity": quantity,
                    "P05": p05,
                    "P50": p50,
                    "P95": p95,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 13. PLOTTING
# =============================================================================

# Springer Nature electronic-artwork export. Every figure is supplied as EPS,
# PDF and TIFF. EPS is the vector master for line art. TIFF is the preferred
# submission file for raster or combination artwork such as Figs. 1--4. The
# TIFF output is RGB at 600 dpi, which meets the journal requirement for
# combination artwork. PDF is retained for reliable local pdfLaTeX compilation.
# Main-manuscript figure files use the required Fig<number> naming convention.
FIGURE_WIDTH_MM = 152.4
FIGURE_MAX_HEIGHT_MM = 234.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
TIFF_DPI = 600

def save_publication_figure(fig, file_stem):
    """Save one Springer/NRR-compliant publication or diagnostic figure.

    Each figure is written in three formats:
      * EPS: vector output with embedded TrueType fonts;
      * PDF: vector output for direct use with pdfLaTeX;
      * TIFF: 600 dpi RGB with lossless LZW compression.

    For Figs. 1--4, which contain raster heat maps plus vector annotations,
    submit the TIFF files to the journal. For predominantly vector figures,
    submit the EPS files.

    No temporary files are created, avoiding Windows/Spyder file-locking issues.
    """
    stem = str(file_stem)
    eps_path = OUTDIR / f"{stem}.eps"
    pdf_path = OUTDIR / f"{stem}.pdf"
    tiff_path = OUTDIR / f"{stem}.tiff"

    # Vector output. ps.fonttype=42 embeds TrueType fonts. Raster artists
    # embedded in EPS are rendered at the same 600 dpi used for combination art.
    fig.savefig(
        eps_path,
        format="eps",
        dpi=TIFF_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
    )

    # PDF vector output for direct use with pdfLaTeX.
    fig.savefig(
        pdf_path,
        format="pdf",
        dpi=TIFF_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
    )

    # Journal-ready raster/combination-art alternative.
    fig.savefig(
        tiff_path,
        format="tiff",
        dpi=TIFF_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        transparent=False,
        pil_kwargs={
            "compression": "tiff_lzw",
        },
    )

    # Keep the numbered manuscript figures beside the manuscript as well as
    # in the analysis output folder. pdfLaTeX will select the PDF companion,
    # while the journal-ready EPS/TIFF master remains available separately.
    if stem.startswith("Fig") and stem[3:].isdigit():
        for src in (eps_path, pdf_path, tiff_path):
            shutil.copy2(src, SCRIPT_DIR / src.name)

def save_map(
    array,
    description,
    colourbar_label,
    figure_number,
):
    """Save one block-scale uncertainty map; description is for code readability only."""
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 4.6)
    )

    image = ax.imshow(
        array,
        origin="lower",
        extent=[0, LX, 0, LZ],
        aspect="auto",
    )

    ax.plot(
        X,
        TRUTH["true_top"],
        linewidth=1.0,
        label="Hidden true contacts",
    )
    ax.plot(
        X,
        TRUTH["true_bottom"],
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Along-strike coordinate (m)"
    )
    ax.set_ylabel(
        "Vertical coordinate (m)"
    )

    colourbar = fig.colorbar(
        image,
        ax=ax,
    )
    colourbar.set_label(
        colourbar_label
    )

    fig.tight_layout()
    save_publication_figure(fig, figure_number)
    plt.close(fig)


def plot_truth():
    """Plot the hidden geological truth."""
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 4.6)
    )

    ax.imshow(
        TRUTH["membership"].astype(float),
        origin="lower",
        extent=[0, LX, 0, LZ],
        aspect="auto",
    )

    ax.plot(
        X,
        TRUTH["true_top"],
        linewidth=1.2,
    )
    ax.plot(
        X,
        TRUTH["true_bottom"],
        linewidth=1.2,
    )

    ax.set_xlabel(
        "Along-strike coordinate (m)"
    )
    ax.set_ylabel(
        "Vertical coordinate (m)"
    )

    fig.tight_layout()
    save_publication_figure(fig, "diagnostic_synthetic_truth")
    plt.close(fig)


def plot_baseline_drilling(details):
    """Plot baseline synthetic assays and contact observations."""
    drill = details["drill"]

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 4.6)
    )

    scatter = ax.scatter(
        drill["assay_x"],
        drill["assay_z"],
        c=drill["assay_grade"],
        s=16,
    )

    ax.scatter(
        drill["contact_x"],
        drill["top_obs"],
        marker="^",
        s=30,
        label="Observed top",
    )

    ax.scatter(
        drill["contact_x"],
        drill["bottom_obs"],
        marker="v",
        s=30,
        label="Observed bottom",
    )

    ax.plot(
        X,
        TRUTH["true_top"],
        linewidth=1.0,
        label="Hidden true contacts",
    )
    ax.plot(
        X,
        TRUTH["true_bottom"],
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Along-strike coordinate (m)"
    )
    ax.set_ylabel(
        "Vertical coordinate (m)"
    )
    ax.legend()

    colourbar = fig.colorbar(
        scatter,
        ax=ax,
    )
    colourbar.set_label(
        "Cu grade (%)"
    )

    fig.tight_layout()
    save_publication_figure(fig, "diagnostic_baseline_drilling")
    plt.close(fig)


PLOT_STYLES = {
    "Tonnes": {"marker": "o", "linestyle": "-"},
    "Average grade": {"marker": "s", "linestyle": "--"},
    "Contained metal": {"marker": "^", "linestyle": ":"},
}


def plot_factor_sensitivity(
    summary,
    factor,
    xlabel,
    figure_number,
):
    """Plot mean reported geological variance share by one factor."""
    grouped = (
        summary.groupby(
            factor,
            as_index=False,
        )[
            [
                "RG_report_tonnes",
                "RG_report_avggrade",
                "RG_report_metal",
            ]
        ]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 4.6)
    )

    ax.plot(
        grouped[factor],
        grouped["RG_report_tonnes"],
        label="Tonnes",
        **PLOT_STYLES["Tonnes"],
    )
    ax.plot(
        grouped[factor],
        grouped["RG_report_avggrade"],
        label="Average grade",
        **PLOT_STYLES["Average grade"],
    )
    ax.plot(
        grouped[factor],
        grouped["RG_report_metal"],
        label="Contained metal",
        **PLOT_STYLES["Contained metal"],
    )

    ax.set_xlabel(
        xlabel
    )
    ax.set_ylabel(
        "Mean reported geological variance share"
    )
    ax.legend()

    fig.tight_layout()
    save_publication_figure(fig, figure_number)
    plt.close(fig)


def plot_fixed_vs_full(summary):
    """Plot selected fixed-geology variance relative to full nested variance."""
    scenario_index = np.arange(
        len(summary)
    )

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 4.6)
    )

    ax.scatter(
        scenario_index,
        summary["fixed_full_tonnes"],
        label="Tonnes",
        marker="o",
        s=34,
    )
    ax.scatter(
        scenario_index,
        summary["fixed_full_metal"],
        label="Contained metal",
        marker="s",
        s=34,
    )

    ax.axhline(
        1.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Factorial scenario"
    )
    ax.set_ylabel(
        r"$Q_{\mathrm{fixed}}$"
    )
    ax.legend()

    fig.tight_layout()
    save_publication_figure(fig, "Fig8")
    plt.close(fig)


def plot_cutoff_curves(table):
    """Plot baseline tonnes and contained-metal uncertainty by cut-off."""
    tonnes = table[
        table["quantity"] == "Tonnes"
    ]

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 4.6)
    )

    ax.plot(
        tonnes["cutoff_pct_cu"],
        tonnes["P50"],
        label="P50 tonnes",
    )
    ax.fill_between(
        tonnes["cutoff_pct_cu"].to_numpy(),
        tonnes["P05"].to_numpy(),
        tonnes["P95"].to_numpy(),
        facecolor="none",
        hatch="///",
        linewidth=0.8,
        label="P05-P95",
    )

    ax.set_xlabel(
        "Cut-off (% Cu)"
    )
    ax.set_ylabel(
        "Tonnes"
    )
    ax.legend()

    fig.tight_layout()
    save_publication_figure(fig, "diagnostic_cutoff_tonnes")
    plt.close(fig)

    metal = table[
        table["quantity"]
        == "Contained metal (t Cu)"
    ]

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 4.6)
    )

    ax.plot(
        metal["cutoff_pct_cu"],
        metal["P50"],
        label="P50 contained metal",
    )
    ax.fill_between(
        metal["cutoff_pct_cu"].to_numpy(),
        metal["P05"].to_numpy(),
        metal["P95"].to_numpy(),
        facecolor="none",
        hatch="///",
        linewidth=0.8,
        label="P05-P95",
    )

    ax.set_xlabel(
        "Cut-off (% Cu)"
    )
    ax.set_ylabel(
        "Contained metal (t Cu)"
    )
    ax.legend()

    fig.tight_layout()
    save_publication_figure(fig, "diagnostic_cutoff_metal")
    plt.close(fig)


def plot_convergence(convergence):
    """Plot reported R_G as G increases for several values of R."""
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_IN, 4.6)
    )

    for r in sorted(
        convergence["R"].unique()
    ):
        subset = convergence[
            convergence["R"] == r
        ]

        ax.plot(
            subset["G"],
            subset["R_G_report"],
            marker="o",
            label=f"R={r}",
        )

    ax.set_xlabel(
        "Number of geological realisations, G"
    )
    ax.set_ylabel(
        r"Reported $R_G$"
    )
    ax.legend(
        ncol=2
    )

    fig.tight_layout()
    save_publication_figure(fig, "diagnostic_convergence")
    plt.close(fig)


# =============================================================================
# 14. METADATA AND AUDIT OUTPUTS
# =============================================================================

def write_metadata():
    """Write the principal experiment settings to CSV."""
    metadata = pd.DataFrame(
        {
            "parameter": [
                "seed",
                "truth_seed",
                "nx",
                "nz",
                "LX_m",
                "LZ_m",
                "block_dx_m",
                "block_dz_m",
                "block_volume",
                "density",
                "n_geological_realisations",
                "n_grade_realisations",
                "baseline_cutoff_pct_cu",
                "contact_range_m",
                "grade_range_x_m",
                "grade_range_z_m",
                "grade_sd_pct_cu",
                "contact_observation_noise_sd_m",
                "assay_noise_sd_pct_cu",
                "contained_metal_units",
            ],
            "value": [
                SEED,
                TRUTH_SEED,
                NX,
                NZ,
                LX,
                LZ,
                DX,
                DZ,
                BLOCK_VOLUME,
                DENSITY,
                N_GEO,
                N_GRADE,
                BASE_CUTOFF,
                CONTACT_RANGE,
                GRADE_RANGE_X,
                GRADE_RANGE_Z,
                GRADE_SD,
                CONTACT_OBS_NOISE_SD,
                ASSAY_NOISE_SD,
                "t Cu",
            ],
        }
    )

    metadata.to_csv(
        OUTDIR / "run_metadata.csv",
        index=False,
    )


def save_baseline_arrays(
    details,
    block_results,
):
    """Save principal baseline arrays for reproducibility and audit."""
    np.savez_compressed(
        OUTDIR / "baseline_arrays.npz",
        memberships=details["memberships"],
        top_contacts=details["top_contacts"],
        bottom_contacts=details["bottom_contacts"],
        tonnes=details["tonnes"],
        average_grade_pct_cu=details["average_grade"],
        contained_metal_t_cu=details["metal_t_cu"],
        Ugrade_block=block_results["U_grade"],
        Ugeology_raw_block=block_results["U_geology_raw"],
        Ugeology_report_block=block_results["U_geology_report"],
        Utotal_report_block=block_results["U_total_report"],
        RG_report_block=block_results["R_G_report"],
        true_top=TRUTH["true_top"],
        true_bottom=TRUTH["true_bottom"],
        true_membership=TRUTH["membership"],
    )


# =============================================================================
# 15. MAIN EXPERIMENT
# =============================================================================

def main():
    """Run the full 27-scenario conditioned synthetic experiment."""
    start_time = time.time()

    print(
        "SELF-CONTAINED CONDITIONED SYNTHETIC EXPERIMENT"
    )
    print(
        "=============================================="
    )
    print(
        f"Grid: {NX} x {NZ} = {NX * NZ:,} blocks"
    )
    print(
        "Scenarios: "
        f"{len(SPACINGS) * len(BOUNDARY_SDS) * len(CONTRASTS)}"
    )
    print(
        f"Nested design: {N_GEO} geological x "
        f"{N_GRADE} grade realisations"
    )

    total_outcomes = (
        len(SPACINGS)
        * len(BOUNDARY_SDS)
        * len(CONTRASTS)
        * N_GEO
        * N_GRADE
    )

    print(
        f"Total resource outcomes: {total_outcomes:,}"
    )
    print(
        "Drill spacing is represented by explicit synthetic drilling."
    )
    print(
        "Grade units: % Cu"
    )
    print(
        "Contained-metal units: tonnes Cu"
    )
    print()

    write_metadata()
    plot_truth()

    # Export one representative drill dataset for each nominal spacing using
    # the baseline contrast. These exports are illustrative and independent
    # of the random streams used in the factorial scenarios.
    for spacing in SPACINGS:
        export_rng = np.random.default_rng(
            SEED + int(spacing) + 1000
        )

        representative_drill = create_drill_dataset(
            spacing,
            BASELINE_CONTRAST,
            export_rng,
        )

        export_drill_dataset(
            representative_drill,
            spacing,
        )

    combinations = list(
        itertools.product(
            SPACINGS,
            BOUNDARY_SDS,
            CONTRASTS,
        )
    )

    master_rng = np.random.default_rng(
        SEED
    )
    scenario_seeds = master_rng.integers(
        1,
        np.iinfo(np.uint32).max,
        size=len(combinations),
        dtype=np.uint32,
    )

    rows = []
    baseline_details = None

    for scenario_number, (
        combination,
        scenario_seed,
    ) in enumerate(
        zip(
            combinations,
            scenario_seeds,
        ),
        start=1,
    ):
        spacing, boundary_sd, contrast = combination

        print(
            f"[{scenario_number:02d}/{len(combinations)}] "
            f"spacing={spacing:.0f} m, "
            f"boundary SD={boundary_sd:.0f} m, "
            f"contrast={contrast:.1f}% Cu"
        )

        retain_details = (
            combination == BASELINE
        )

        row, details = run_scenario(
            spacing=spacing,
            boundary_sd=boundary_sd,
            contrast=contrast,
            seed=int(scenario_seed),
            retain_details=retain_details,
        )

        rows.append(
            row
        )

        if retain_details:
            baseline_details = details

    summary = pd.DataFrame(
        rows
    )
    summary.to_csv(
        OUTDIR / "scenario_summary.csv",
        index=False,
    )

    if baseline_details is None:
        raise RuntimeError(
            "Baseline scenario was not retained."
        )

    baseline_row = summary[
        (summary["spacing_m"] == BASELINE_SPACING)
        & (
            summary["boundary_sd_m"]
            == BASELINE_BOUNDARY_SD
        )
        & (
            summary["contrast_pct_cu"]
            == BASELINE_CONTRAST
        )
    ].iloc[0]

    # -------------------------------------------------------------------------
    # Baseline quantiles
    # -------------------------------------------------------------------------
    quantile_table = baseline_quantile_table(
        baseline_details
    )
    quantile_table.to_csv(
        OUTDIR / "baseline_quantiles.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Block-scale decomposition
    # -------------------------------------------------------------------------
    block_results = block_decomposition(
        baseline_details["block_star"]
    )

    block_table = pd.DataFrame(
        {
            "x_m": XX.ravel(),
            "z_m": ZZ.ravel(),
            "Ugrade_block": block_results[
                "U_grade"
            ].ravel(),
            "Ugeology_raw_block": block_results[
                "U_geology_raw"
            ].ravel(),
            "Ugeology_report_block": block_results[
                "U_geology_report"
            ].ravel(),
            "Utotal_report_block": block_results[
                "U_total_report"
            ].ravel(),
            "RG_report_block": block_results[
                "R_G_report"
            ].ravel(),
        }
    )

    block_table.to_csv(
        OUTDIR / "baseline_block_uncertainty.csv",
        index=False,
    )

    save_map(
        block_results["U_grade"],
        "Conditional-grade uncertainty, baseline",
        r"$\widehat U_{\mathrm{grade}}(B)$",
        "Fig1",
    )

    save_map(
        block_results["U_geology_report"],
        "Geological uncertainty, baseline",
        r"$\widehat U_{\mathrm{geology}}(B)$",
        "Fig2",
    )

    save_map(
        block_results["U_total_report"],
        "Total reported uncertainty, baseline",
        r"$\widehat U_{\mathrm{total}}(B)$",
        "Fig3",
    )

    save_map(
        block_results["R_G_report"],
        "Geological uncertainty ratio, baseline",
        r"$R_G(B)$",
        "Fig4",
    )

    plot_baseline_drilling(
        baseline_details
    )

    # -------------------------------------------------------------------------
    # Cut-off curves
    # -------------------------------------------------------------------------
    cutoff_table = baseline_cutoff_table(
        baseline_details
    )
    cutoff_table.to_csv(
        OUTDIR / "baseline_cutoff_curves.csv",
        index=False,
    )
    plot_cutoff_curves(
        cutoff_table
    )

    # -------------------------------------------------------------------------
    # Factor sensitivities
    # -------------------------------------------------------------------------
    plot_factor_sensitivity(
        summary,
        "boundary_sd_m",
        "Boundary uncertainty SD (m)",
        "Fig5",
    )

    plot_factor_sensitivity(
        summary,
        "spacing_m",
        "Drill spacing (m)",
        "Fig6",
    )

    plot_factor_sensitivity(
        summary,
        "contrast_pct_cu",
        "Mineralised/background mean contrast (% Cu)",
        "Fig7",
    )

    plot_fixed_vs_full(
        summary
    )

    # -------------------------------------------------------------------------
    # Convergence diagnostic
    # -------------------------------------------------------------------------
    convergence = convergence_from_full_arrays(
        baseline_details["metal_t_cu"]
    )
    convergence.insert(
        0,
        "quantity",
        "Contained metal (t Cu)",
    )
    convergence.to_csv(
        OUTDIR / "convergence_summary.csv",
        index=False,
    )
    plot_convergence(
        convergence
    )

    # -------------------------------------------------------------------------
    # Audit arrays
    # -------------------------------------------------------------------------
    save_baseline_arrays(
        baseline_details,
        block_results,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print()
    print(
        "BASELINE CORRECTED DECOMPOSITION"
    )
    print(
        "--------------------------------"
    )

    print(
        "Tonnes: "
        f"U_grade={baseline_row['Ugrade_tonnes']:.6g}, "
        f"U_geo_raw={baseline_row['Ugeology_raw_tonnes']:.6g}, "
        f"U_geo_report={baseline_row['Ugeology_report_tonnes']:.6g}, "
        f"R_G(report)={baseline_row['RG_report_tonnes']:.4f}"
    )

    print(
        "Average grade (% Cu): "
        f"U_grade={baseline_row['Ugrade_avggrade']:.6g}, "
        f"U_geo_raw={baseline_row['Ugeology_raw_avggrade']:.6g}, "
        f"U_geo_report={baseline_row['Ugeology_report_avggrade']:.6g}, "
        f"R_G(report)={baseline_row['RG_report_avggrade']:.4f}"
    )

    print(
        "Contained metal (t Cu): "
        f"U_grade={baseline_row['Ugrade_metal_t_cu']:.6g}, "
        f"U_geo_raw={baseline_row['Ugeology_raw_metal_t_cu']:.6g}, "
        f"U_geo_report={baseline_row['Ugeology_report_metal_t_cu']:.6g}, "
        f"R_G(report)={baseline_row['RG_report_metal']:.4f}"
    )

    print()
    print(
        "BASELINE EMPIRICAL QUANTILES"
    )
    print(
        "----------------------------"
    )
    print(
        quantile_table.to_string(
            index=False
        )
    )

    print()
    print(
        "BASELINE FIXED-GEOLOGY DIAGNOSTIC"
    )
    print(
        "---------------------------------"
    )
    print(
        "Q_fixed tonnes="
        f"{baseline_row['fixed_full_tonnes']:.4f}"
    )
    print(
        "Q_fixed average grade="
        f"{baseline_row['fixed_full_avggrade']:.4f}"
    )
    print(
        "Q_fixed contained metal="
        f"{baseline_row['fixed_full_metal']:.4f}"
    )

    print()
    print(
        "FACTORIAL SANITY CHECKS"
    )
    print(
        "------------------------"
    )

    for factor in (
        "boundary_sd_m",
        "spacing_m",
        "contrast_pct_cu",
    ):
        grouped = summary.groupby(
            factor
        )[
            [
                "RG_report_tonnes",
                "RG_report_avggrade",
                "RG_report_metal",
            ]
        ].mean()

        print()
        print(
            grouped.to_string()
        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Elapsed time: {elapsed / 60.0:.1f} minutes"
    )
    print(
        f"Outputs written to: {OUTDIR.resolve()}"
    )


if __name__ == "__main__":
    main()
