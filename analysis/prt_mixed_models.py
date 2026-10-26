"""Mixed-effects analyses of pup-retrieval distances and trial-order latency trends.

See the accompanying notebook and README for scientific interpretation and inputs.
Run: python prt_mixed_models.py --data data/prt_trials.xlsx --output results
"""

# SECTION: imports
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import warnings
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrices
from scipy import stats
from statsmodels.stats.multitest import multipletests

ANALYSIS_VERSION = "1.2.0"
COLORS = {"SHAM": "#D97706", "CBD": "#15803D", "THC": "#7E22CE"}

HEADERS = {
    "Treatment": "Treatment", "Litter ID": "Litter_ID", "Sex": "Sex",
    "Trial #": "Trial", "firstEncounter sec": "tFE_s", "success sec": "tR_s",
    "Event tFE<90": "Event_FE", "Event tR<90": "Event_R",
    "distanceToFirstEncounter cm": "Distance_FE_cm",
    "distanceFEtoRet cm": "Distance_FE_R_cm", "distance cm": "Distance_total_cm",
}

DISTANCE_METRICS = {
    "dFE": ("Distance_FE_cm", "Event_FE", "Distance to first encounter (cm)"),
    "dFE_R": ("Distance_FE_R_cm", "Event_R", "Distance from encounter to retrieval (cm)"),
    "dTotal": ("Distance_total_cm", None, "Total recorded distance (cm)"),
}
LEARNING_METRICS = {
    "learn_FE": ("tFE_s", "Event_FE", "First-encounter latency (s)"),
    "learn_R": ("tR_s", "Event_R", "Retrieval latency (s)"),
}

# SECTION: input
def resolve_config(config=None):
    """Accept a dictionary or JSON configuration; unspecified options use defaults."""
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    config = {} if config is None else dict(config)
    allowed = {"columns", "distance_metrics", "learning_metrics", "treatment_levels",
               "sex_levels", "treatment_reference", "sex_reference", "distance_comparisons",
               "trial_origin", "distance_unit", "latency_unit", "deadline"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration options: {sorted(unknown)}")
    return config


def ordered_levels(values, supplied=None, reference=None, preferred=None):
    """Keep user labels; reference first. SHAM/Female are preferred only if present."""
    observed = list(pd.unique(values.dropna()))
    levels = list(supplied) if supplied is not None else observed
    if len(levels) != len(set(levels)) or set(levels) != set(observed):
        raise ValueError(f"Configured levels {levels} must match observed labels {observed}.")
    if not levels:
        return []
    if reference is None:
        reference = next((x for x in levels if preferred and x.casefold() == preferred.casefold()), levels[0])
    if reference not in levels:
        raise ValueError(f"Reference {reference!r} not found among {levels}.")
    return [reference] + [x for x in levels if x != reference]


def settings_for(result):
    return result._prt_settings


def normalize_header(value):
    """Normalize nonbreaking/duplicate spaces without changing measurement names."""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def prepare_data(raw, config=None):
    """Validate trial-level data; record exclusions instead of silently losing rows.

    One row is one pup-retrieval trial. A litter ID identifies one dam/litter and
    must be unique across treatment groups. Missing sex does not exclude a trial
    from the learning analysis, which does not use sex as a predictor.
    """
    config = resolve_config(config)
    distances = dict(config.get("distance_metrics", DISTANCE_METRICS))
    learning = dict(config.get("learning_metrics", LEARNING_METRICS))
    if not distances and not learning:
        raise ValueError("Configure at least one distance or latency outcome.")
    if set(distances) & set(learning):
        raise ValueError("Distance and latency outcome keys must be distinct.")
    for key, metric in {**distances, **learning}.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,12}", key) or len(metric) != 3:
            raise ValueError("Outcome keys must be short identifiers; each definition is [column, event-or-null, title].")
    mapping = {value: key for key, value in HEADERS.items()}
    mapping.update(config.get("columns", {}))
    required = {"Treatment", "Litter_ID"}
    if distances:
        required.add("Sex")
    if learning:
        required.add("Trial")
    for column, event, _ in [*distances.values(), *learning.values()]:
        required.add(column)
        if event is not None:
            required.add(event)
    if required - set(mapping):
        raise ValueError(f"Provide columns mapping for: {sorted(required - set(mapping))}")
    # Ignore disabled outcomes entirely; their absent/malformed columns should
    # not prevent an otherwise valid subset of analyses from running.
    mapping = {internal: source for internal, source in mapping.items()
               if internal in required | {"Sex", "Trial"}}
    df = raw.copy()
    df.columns = [normalize_header(c) for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names after whitespace normalization.")
    mapping = {internal: normalize_header(external) for internal, external in mapping.items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Column mapping must not reuse one source column for multiple variables.")
    missing = sorted(mapping[x] for x in required if mapping[x] not in df.columns)
    if missing:
        raise ValueError(f"Required spreadsheet columns are missing: {missing}")
    active_mapping = {source: internal for internal, source in mapping.items() if source in df.columns}
    df = df[list(active_mapping)].rename(columns=active_mapping)
    for optional in ("Sex", "Trial"):
        if optional not in df:
            df[optional] = np.nan
    df.insert(0, "source_row", np.arange(len(df)) + 2)
    flow = [{"stage": "input", "n_rows": len(df), "detail": "All spreadsheet rows"}]
    # Nullable strings preserve missing values. astype(str) would turn NaN into
    # a literal 'nan' litter ID and incorrectly treat it as a real group.
    for col in ("Treatment", "Sex", "Litter_ID"):
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df = df.dropna(subset=["Treatment", "Litter_ID"]).copy()
    flow.append({"stage": "valid_group_keys", "n_rows": len(df),
                 "detail": "Nonmissing treatment and litter ID"})
    if df.empty:
        raise ValueError("No trials remain after checking treatment/litter IDs.")
    for col in [c for c in df.columns if c not in ("Treatment", "Sex", "Litter_ID", "source_row")]:
        parsed = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & parsed.isna()
        if bad.any():
            raise ValueError(f"Nonnumeric entries in {col}, rows {df.loc[bad, 'source_row'].tolist()}")
        if np.isinf(parsed.dropna().to_numpy()).any():
            raise ValueError(f"Infinite values in {col}.")
        df[col] = parsed.astype(float)
    for col in {event for _, event, _ in [*distances.values(), *learning.values()] if event is not None}:
        if not df[col].dropna().isin([0, 1]).all():
            raise ValueError(f"{col} must contain 0, 1, or missing values.")
    trials = df["Trial"].dropna()
    origin = float(config.get("trial_origin", 1))
    if not np.isfinite(origin) or origin != int(origin):
        raise ValueError("trial_origin must be a finite integer.")
    if ((trials < origin) | (trials != np.floor(trials))).any():
        raise ValueError(f"Trial numbers must be integers >= trial_origin ({origin:g}).")
    measurement_cols = list({column for column, _, _ in [*distances.values(), *learning.values()]})
    if (df[measurement_cols] < 0).any().any():
        raise ValueError("Latencies and distances cannot be negative.")
    if df.groupby("Litter_ID")["Treatment"].nunique().gt(1).any():
        raise ValueError("A litter ID occurs in more than one treatment. Use globally unique litter IDs.")
    duplicates = df.loc[df["Trial"].notna()].duplicated(["Litter_ID", "Trial"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate trial numbers within a litter; confirm one row per trial.")
    treatments = ordered_levels(df["Treatment"], config.get("treatment_levels"), config.get("treatment_reference"), "SHAM")
    sexes = ordered_levels(df["Sex"], config.get("sex_levels"), config.get("sex_reference"), "Female")
    if len(treatments) < 2 or (distances and len(sexes) < 2):
        raise ValueError("Group-comparison models require >=2 treatments; distance Sex*Treatment models also require >=2 sex/factor levels.")
    df["Treatment"] = pd.Categorical(df["Treatment"], treatments)
    df["Sex"] = pd.Categorical(df["Sex"], sexes)
    df["Litter_ID"] = df["Litter_ID"].astype(str)
    # Center BOTH fixed and random trial terms. With a freely estimated random
    # intercept-slope covariance, this is the same model at a different origin.
    df["Trial_centered"] = df["Trial"] - origin
    default_comparisons = [
        (sex, treatments[0], sex, treatment, f"{sex}: {treatments[0]} - {treatment}")
        for sex in sexes for treatment in treatments[1:]
    ] + [
        (sexes[0], treatment, sex, treatment, f"{treatment}: {sexes[0]} - {sex}")
        for sex in sexes[1:] for treatment in treatments
    ] if sexes else []
    comparisons = list(config.get("distance_comparisons", default_comparisons))
    for entry in comparisons:
        if len(entry) != 5 or entry[0] not in sexes or entry[2] not in sexes or entry[1] not in treatments or entry[3] not in treatments:
            raise ValueError(f"Invalid distance comparison: {entry}")
        if tuple(entry[:2]) == tuple(entry[2:4]):
            raise ValueError("A distance comparison must compare different groups.")
    if len({entry[4] for entry in comparisons}) != len(comparisons):
        raise ValueError("Distance comparison labels must be unique.")
    if distances and not comparisons:
        raise ValueError("Configure at least one distance comparison.")
    palette = plt.get_cmap("tab10")
    df.attrs["prt_settings"] = {
        "distance_metrics": distances, "learning_metrics": learning,
        "treatments": treatments, "sexes": sexes,
        "reference_treatment": treatments[0], "reference_sex": sexes[0] if sexes else None,
        "distance_comparisons": comparisons, "trial_origin": origin,
        "distance_unit": config.get("distance_unit", "cm"),
        "latency_unit": config.get("latency_unit", "s"),
        "deadline": config.get("deadline", 90),
        "colors": {t: COLORS.get(t, palette(i % 10)) for i, t in enumerate(treatments)},
        "column_mapping": mapping,
    }
    return df, pd.DataFrame(flow)


def select_outcome(df, metric, event, learning=False):
    """Keep the original outcome-specific inclusion rules and report their scope."""
    keep = df[metric].notna()
    if event is not None:
        keep &= df[event].eq(1)
    if learning:
        keep &= df["Trial"].notna()
    else:
        keep &= df["Sex"].notna()
    selected = df.loc[keep].copy()
    selected.attrs["event_filter"] = event
    selected["Metric"] = selected[metric]
    if selected.empty:
        raise ValueError(f"No analyzable trials for {metric}.")
    counts = selected.groupby("Treatment", observed=False).agg(
        n_trials=("Metric", "size"), n_litters=("Litter_ID", "nunique")
    ).reset_index()
    counts["selection"] = (f"{event}=1; " if event else "No event filter; ") + (
        "nonmissing latency and trial" if learning else "nonmissing distance and sex")
    return selected, counts


# SECTION: model_fitting
def fit_mixed_model(data, formula, random_formula):
    """REML fit; retain optimizer warnings and never silently simplify the model."""
    model = smf.mixedlm(formula, data=data, groups="Litter_ID", re_formula=random_formula)
    # Keep explicit Patsy coding for later contrasts. Statsmodels 0.15 moved
    # internal formula metadata, so do not rely on model.data.design_info.
    _, fixed_design = dmatrices(formula, data, return_type="dataframe")
    if list(fixed_design.columns) != list(model.exog_names) or not np.allclose(
        fixed_design.to_numpy(), model.exog
    ):
        raise ValueError("Prediction design coding differs from the fitted model.")
    if len(model.endog) != len(data):
        raise ValueError("Formula processing unexpectedly dropped rows.")
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        raise ValueError("Fixed-effect design is rank deficient; check missing groups/cells.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(reml=True, method=["lbfgs", "bfgs", "cg"], maxiter=2000, disp=False)
    messages = list(dict.fromkeys(str(w.message) for w in caught))
    if not result.converged:
        raise RuntimeError("Mixed model did not converge. " + " | ".join(messages))
    result._prt_design_info = fixed_design.design_info
    result._prt_settings = data.attrs["prt_settings"]
    return result, messages


def nakagawa_r2(result):
    """Marginal/conditional R² for these Gaussian LMMs (no extra variance components).

    Var_fixed = sample variance of X beta (ddof=1).
    Var_random = mean_i z_i G z_i', using the estimated covariance G = cov_re.
    For an intercept-only random term this is G[0,0]. For random slopes it
    includes intercept variance, slope variance AND intercept-slope covariance.
    The sample variance of the predicted random effects is not substituted for G:
    these conditional predictions are shrunk toward zero.
    """
    if result.model.k_vc:
        raise NotImplementedError("This R² helper is restricted to the supplied random structures.")
    fixed_prediction = np.asarray(result.model.exog) @ np.asarray(result.fe_params)
    z = np.asarray(result.model.exog_re)
    g = np.asarray(result.cov_re)
    var_fixed = float(np.var(fixed_prediction, ddof=1))
    var_random = float(np.mean(np.einsum("ij,jk,ik->i", z, g, z)))
    var_residual = float(result.scale)
    if var_random < -1e-8 or var_residual <= 0:
        raise ValueError("Invalid fitted variance components; inspect the model fit.")
    var_random = max(0.0, var_random)
    total = var_fixed + var_random + var_residual
    return pd.DataFrame([{
        "R2_marginal": var_fixed / total,
        "R2_conditional": (var_fixed + var_random) / total,
        "variance_fixed_predictions": var_fixed,
        "mean_random_effect_variance": var_random,
        "residual_variance": var_residual,
    }])


def fixed_effects_table(result):
    """Only fixed-effect estimates; their Wald p-values and CIs are unadjusted."""
    names = list(result.fe_params.index)
    ci = result.conf_int().loc[names]
    return pd.DataFrame({
        "term": names, "estimate": result.fe_params.to_numpy(),
        "SE": np.asarray(result.bse_fe), "z": result.tvalues.loc[names].to_numpy(),
        "p_raw": result.pvalues.loc[names].to_numpy(),
        "CI95_low_unadjusted": ci.iloc[:, 0].to_numpy(),
        "CI95_high_unadjusted": ci.iloc[:, 1].to_numpy(),
    })


def random_covariance_table(result):
    """Original-scale random variances/covariances, not normalized result.params.

    Normal-Wald p-values/CIs for variance components are deliberately not copied
    from the packed parameter vector; variances have a boundary at zero.
    """
    covariance = result.cov_re
    labels = list(covariance.index)
    rows = []
    for i, left in enumerate(labels):
        for j in range(i + 1):
            rows.append({"component_1": left, "component_2": labels[j],
                         "kind": "variance" if i == j else "covariance",
                         "estimate_original_scale": float(covariance.iloc[i, j])})
    rows.append({"component_1": "Residual", "component_2": "Residual",
                 "kind": "variance", "estimate_original_scale": float(result.scale)})
    return pd.DataFrame(rows)


def omnibus_table(result, learning=False):
    """Joint Wald tests on fitted fixed coefficients AND their covariance matrix.

    No ANOVA/Kruskal-Wallis test is applied to raw values or fitted values here.
    Under reference coding and interactions, simple terms are conditional on
    reference values; the interaction test is joint over both interaction terms.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The behavior of wald_test")
        table = result.wald_test_terms(scalar=True).table
    cfg = settings_for(result); ref = cfg["reference_treatment"]; sex = cfg["reference_sex"]
    origin = cfg["trial_origin"]
    meanings = {
        "Intercept": f"{ref} mean at trial {origin:g}" if learning else f"{sex} {ref} mean",
        "Treatment": f"All treatment contrasts vs {ref} at trial {origin:g}" if learning else f"All treatment contrasts vs {ref} within {sex}",
        "Trial_centered": f"Trial slope in {ref}",
        "Sex": f"All sex/factor contrasts within {ref}",
        "Trial_centered:Treatment": f"All slope differences vs {ref} jointly equal zero",
        "Sex:Treatment": "All sex-by-treatment interactions jointly equal zero",
    }
    rows = []
    for term, row in table.iterrows():
        rows.append({"term": term,
                     "chi2": float(np.asarray(row["statistic"]).item()),
                     "df": int(np.asarray(row["df_constraint"]).item()),
                     "p_raw": float(np.asarray(row["pvalue"]).item()),
                     "hypothesis": meanings.get(term, "See formula/design coding")})
    return pd.DataFrame(rows)


# SECTION: contrasts
def design_row(result, treatment, sex=None, trial=None):
    """Build a prediction/contrast row with the SAME formula coding as the fit."""
    cfg = settings_for(result)
    trial = cfg["trial_origin"] if trial is None else trial
    sex = cfg["reference_sex"] if sex is None else sex
    grid = pd.DataFrame({
        "Treatment": pd.Categorical([treatment], cfg["treatments"]),
        "Sex": pd.Categorical([sex], cfg["sexes"]),
        "Trial_centered": [trial - cfg["trial_origin"]],
    })
    design = build_design_matrices([result._prt_design_info], grid)[0]
    return np.asarray(design)[0]


def contrast_summary(result, weights):
    """Asymptotic two-sided Wald z contrast using the fitted model covariance."""
    test = result.t_test(np.asarray(weights, dtype=float)[None, :], use_t=False)
    ci = np.asarray(test.conf_int(alpha=0.05))[0]
    return {"estimate": float(np.asarray(test.effect).item()),
            "SE": float(np.asarray(test.sd).item()),
            "z": float(np.asarray(test.tvalue).item()),
            "p_raw": float(np.asarray(test.pvalue).item()),
            "CI95_low_unadjusted": float(ci[0]),
            "CI95_high_unadjusted": float(ci[1])}


def distance_contrasts(result):
    """Configured contrasts from one factorial model; BH family is per outcome."""
    rows = []
    for sex1, trt1, sex2, trt2, label in settings_for(result)["distance_comparisons"]:
        weights = design_row(result, trt1, sex1) - design_row(result, trt2, sex2)
        rows.append({"comparison": label, **contrast_summary(result, weights)})
    table = pd.DataFrame(rows)
    table["p_BH"] = multipletests(table["p_raw"], method="fdr_bh")[1]
    table["p_Bonferroni"] = multipletests(table["p_raw"], method="bonferroni")[1]
    table["family_n"] = len(table)
    return table


def learning_contrasts(result):
    """Baseline and slope comparisons versus the configured reference treatment.

    p_raw preserves the original notebook's nominal inference. Additional BH
    columns are provided within each reference-comparison family (baseline or slope).
    They are NOT claimed to have been prespecified, and must not be described as
    the original manuscript's adjusted p-values. Decide/report the inferential
    family consistently rather than selecting whichever p-column is significant.
    """
    rows = []
    cfg = settings_for(result); ref = cfg["reference_treatment"]; origin = cfg["trial_origin"]
    sham_base = design_row(result, ref, trial=origin)
    sham_slope = design_row(result, ref, trial=origin+1) - sham_base
    for treatment in cfg["treatments"][1:]:
        base = design_row(result, treatment, trial=origin)
        slope = design_row(result, treatment, trial=origin+1) - base
        rows.append({"family": "Baseline differences", "comparison": f"{treatment} - {ref} at trial {origin:g}",
                     **contrast_summary(result, base - sham_base)})
        rows.append({"family": "Slope differences", "comparison": f"{treatment} - {ref} slope",
                     **contrast_summary(result, slope - sham_slope)})
    table = pd.DataFrame(rows)
    table["p_BH_within_family_optional"] = np.nan
    table["family_n"] = 0
    for indices in table.groupby("family", sort=False).groups.values():
        table.loc[indices, "family_n"] = len(indices)
        table.loc[indices, "p_BH_within_family_optional"] = multipletests(
            table.loc[indices, "p_raw"], method="fdr_bh"
        )[1]
    return table


def learning_slope_estimates(result):
    """Estimated slopes and unadjusted CIs; no additional significance tests."""
    rows = []
    cfg = settings_for(result); origin = cfg["trial_origin"]
    for treatment in cfg["treatments"]:
        slope = design_row(result, treatment, trial=origin+1) - design_row(result, treatment, trial=origin)
        values = contrast_summary(result, slope)
        rows.append({"Treatment": treatment,
                     **{k: v for k, v in values.items() if k not in ("p_raw", "z")}})
    return pd.DataFrame(rows)


def effect_specifications(result, learning=False):
    """Define meaningful magnitudes as fixed-effect linear contrasts.

    Pairwise directions match the existing contrast tables. Interactions are
    differences of differences; slopes retain an explicit per-trial unit.
    These definitions add effect-size summaries, not new significance tests.
    """
    specs = []
    cfg = settings_for(result); treatments = cfg["treatments"]; sexes = cfg["sexes"]
    ref = cfg["reference_treatment"]; origin = cfg["trial_origin"]
    distance_unit = cfg["distance_unit"]; latency_unit = cfg["latency_unit"]

    def add(label, kind, unit, weights):
        specs.append({"effect": label, "kind": kind, "raw_unit": unit,
                      "standardized_unit": "residual SD per trial" if kind in ("Within-group trial slope", "Treatment-by-trial slope difference") else "residual SD",
                      "weights": np.asarray(weights, dtype=float)})

    if not learning:
        for s1, t1, s2, t2, label in cfg["distance_comparisons"]:
            add(label, "Pairwise mean difference", distance_unit,
                design_row(result, t1, s1) - design_row(result, t2, s2))
        for sex in sexes[1:]:
            sham_sex_difference = design_row(result, ref, sex) - design_row(result, ref, sexes[0])
            for treatment in treatments[1:]:
                exposed_sex_difference = design_row(result, treatment, sex) - design_row(result, treatment, sexes[0])
                add(f"({sex} - {sexes[0]}): {treatment} - {ref}", "Sex-by-treatment difference of differences",
                    distance_unit, exposed_sex_difference - sham_sex_difference)
    else:
        sham_base = design_row(result, ref, trial=origin)
        sham_slope = design_row(result, ref, trial=origin+1) - sham_base
        for treatment in treatments[1:]:
            base = design_row(result, treatment, trial=origin)
            slope = design_row(result, treatment, trial=origin+1) - base
            add(f"{treatment} - {ref} at trial {origin:g}", "Baseline mean difference", latency_unit, base - sham_base)
            add(f"{treatment} - {ref} slope", "Treatment-by-trial slope difference", f"{latency_unit}/trial", slope - sham_slope)
        for treatment in treatments:
            slope = design_row(result, treatment, trial=origin+1) - design_row(result, treatment, trial=origin)
            add(f"{treatment} slope", "Within-group trial slope", f"{latency_unit}/trial", slope)
    return specs


def standardize_effects(raw_effects, residual_variance):
    """Residual-standardized model effects, not an independent-groups Cohen d.

    Mean differences and difference-of-differences are in residual SD units.
    Slopes/differences of slopes are in residual SD PER TRIAL (not dimensionless).
    Sigma uncertainty must be included when constructing standardized intervals.
    """
    if not np.isfinite(residual_variance) or residual_variance <= 0:
        raise ValueError("A positive residual variance is required to standardize effects.")
    return np.asarray(raw_effects, dtype=float) / np.sqrt(residual_variance)


def effect_size_table(result, data, formula, random_formula, learning=False,
                      bootstrap_reps=1000, seed=20261026):
    """Raw model effects + residual-standardized effects and their uncertainty.

    Raw intervals: model-based unadjusted 95% Wald intervals.
    Standardized intervals: 95% percentile parametric bootstrap. Every replicate
    simulates new litter random effects plus residuals on the SAME retained design,
    refits the full LMM and re-estimates sigma before standardizing. It therefore
    includes denominator uncertainty; it does not divide a Wald interval by a
    fixed estimated sigma and call that a full standardized-effect interval.

    This is conditional on the Gaussian LMM and the analyzed event-success rows.
    It does not model the censoring/success mechanism or add new significance tests.
    """
    if not isinstance(bootstrap_reps, (int, np.integer)) or bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be a nonnegative integer.")
    specs = effect_specifications(result, learning)
    weights = np.stack([spec["weights"] for spec in specs])
    sigma = float(np.sqrt(result.scale))
    rows = []
    for spec in specs:
        values = contrast_summary(result, spec["weights"])
        rows.append({
            **{key: value for key, value in spec.items() if key != "weights"},
            "raw_effect": values["estimate"],
            "raw_CI95_low_Wald": values["CI95_low_unadjusted"],
            "raw_CI95_high_Wald": values["CI95_high_unadjusted"],
            "residual_SD_denominator": sigma,
            "standardized_effect": float(standardize_effects(values["estimate"], result.scale)),
            "standardized_CI95_low_bootstrap": np.nan,
            "standardized_CI95_high_bootstrap": np.nan,
        })
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    draws = []
    failures, warning_count = [], 0
    if bootstrap_reps:
        print(f"Effect-size uncertainty: {bootstrap_reps} parametric bootstrap refits; seed={seed}", flush=True)
        fixed_mean = np.asarray(result.model.exog) @ np.asarray(result.fe_params)
        z = np.asarray(result.model.exog_re)
        group_index, group_labels = pd.factorize(np.asarray(result.model.groups), sort=False)
        g = np.asarray(result.cov_re)
        eigenvalues, eigenvectors = np.linalg.eigh((g + g.T) / 2)
        if eigenvalues.min() < -1e-8:
            raise ValueError("Estimated random covariance is not positive semidefinite.")
        root_g = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        for replicate in range(bootstrap_reps):
            random_effects = rng.normal(size=(len(group_labels), g.shape[0])) @ root_g.T
            simulated = fixed_mean + np.sum(z * random_effects[group_index], axis=1) + rng.normal(0, sigma, len(data))
            simulated_data = data.copy()
            simulated_data["Metric"] = simulated
            try:
                refit, fit_warnings = fit_mixed_model(simulated_data, formula, random_formula)
                raw_draw = weights @ np.asarray(refit.fe_params)
                standardized_draw = standardize_effects(raw_draw, float(refit.scale))
                if not np.isfinite(standardized_draw).all():
                    raise ValueError("Nonfinite standardized effect in bootstrap refit.")
                warning_count += bool(fit_warnings)
                for index, spec in enumerate(specs):
                    draws.append({"replicate": replicate + 1, "effect": spec["effect"],
                                  "raw_effect": float(raw_draw[index]),
                                  "residual_SD": float(np.sqrt(refit.scale)),
                                  "standardized_effect": float(standardized_draw[index])})
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failures.append(str(exc))
            if (replicate + 1) % 100 == 0:
                print(f"  Bootstrap refits attempted: {replicate + 1}/{bootstrap_reps}", flush=True)
    draw_table = pd.DataFrame(draws, columns=["replicate", "effect", "raw_effect", "residual_SD", "standardized_effect"])
    n_success = int(draw_table["replicate"].nunique())
    adequate_success = bootstrap_reps > 0 and n_success >= max(2, int(np.ceil(0.9 * bootstrap_reps)))
    if adequate_success:
        for index, row in table.iterrows():
            samples = draw_table.loc[draw_table["effect"].eq(row["effect"]), "standardized_effect"]
            low, high = np.quantile(samples, [0.025, 0.975])
            table.loc[index, "standardized_CI95_low_bootstrap"] = low
            table.loc[index, "standardized_CI95_high_bootstrap"] = high
    status = ("skipped: standardized CIs not calculated" if bootstrap_reps == 0 else
              "insufficient converged refits: standardized CIs withheld" if not adequate_success else
              "low replicate count: use >=1000 for final reporting" if bootstrap_reps < 1000 else
              "completed: inspect convergence warnings and interval stability")
    table["bootstrap_successful"] = n_success
    table["standardized_CI_status"] = status
    diagnostics = pd.DataFrame([{
        "requested": bootstrap_reps, "successful": n_success,
        "failed": len(failures), "converged_refits_with_warnings": warning_count,
        "seed": seed, "status": status,
        "failure_examples": " | ".join(list(dict.fromkeys(failures))[:5]),
    }])
    return table, diagnostics, draw_table


# SECTION: plotting
def fixed_prediction(result, grid):
    """Population fixed-effects mean and its unadjusted 95% confidence interval.

    This interval is for the mean, not for a new dam or a new observed trial.
    """
    design = np.asarray(build_design_matrices([result._prt_design_info], grid)[0])
    names = list(result.fe_params.index)
    covariance = result.cov_params().loc[names, names].to_numpy()
    estimate = design @ result.fe_params.to_numpy()
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design, covariance, design), 0))
    return estimate, estimate - stats.norm.ppf(0.975) * se, estimate + stats.norm.ppf(0.975) * se


def plot_learning(result, data, title, destination):
    cfg = settings_for(result)
    treatments = cfg["treatments"]; colors = cfg["colors"]
    fig, axes = plt.subplots(1, len(treatments), figsize=(4 * len(treatments), 4), sharex=True, sharey=True)
    for ax, treatment in zip(axes, treatments):
        subset = data.loc[data["Treatment"].eq(treatment)]
        for _, litter in subset.groupby("Litter_ID"):
            litter = litter.sort_values("Trial")
            ax.plot(litter["Trial"], litter["Metric"], color=colors[treatment],
                    alpha=0.18, linewidth=0.8, marker=".", markersize=3)
        trial = np.linspace(subset["Trial"].min(), subset["Trial"].max(), 100)
        grid = pd.DataFrame({"Trial_centered": trial - cfg["trial_origin"],
                             "Treatment": pd.Categorical([treatment] * len(trial), treatments)})
        mean, low, high = fixed_prediction(result, grid)
        ax.fill_between(trial, low, high, color=colors[treatment], alpha=0.15)
        ax.plot(trial, mean, color=colors[treatment], linewidth=2.5)
        ax.set_title(treatment)
        ax.set_xlabel("Trial number")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(title)
    scope = "Successful events only" if data.attrs.get("event_filter") else "Observed latencies"
    fig.suptitle(f"{scope}: raw dam trajectories and fitted mean (95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance(result, data, title, destination):
    cfg = settings_for(result)
    sexes = cfg["sexes"]; treatments = cfg["treatments"]; colors = cfg["colors"]
    fig, axes = plt.subplots(1, len(sexes), figsize=(4 * len(sexes), 4), sharey=True)
    rng = np.random.default_rng(1926)  # Reproducible plotting jitter only.
    for ax, sex in zip(axes, sexes):
        for x, treatment in enumerate(treatments):
            subset = data.loc[data["Sex"].eq(sex) & data["Treatment"].eq(treatment)]
            ax.scatter(x + rng.uniform(-0.13, 0.13, len(subset)), subset["Metric"],
                       s=10, alpha=0.25, color=colors[treatment])
            grid = pd.DataFrame({"Sex": pd.Categorical([sex], sexes),
                                 "Treatment": pd.Categorical([treatment], treatments)})
            mean, low, high = fixed_prediction(result, grid)
            ax.errorbar(x, mean[0], yerr=[[mean[0]-low[0]], [high[0]-mean[0]]],
                        fmt="o", color="black", capsize=4, zorder=5)
        ax.set_xticks(range(len(treatments)), treatments)
        ax.set_title(sex)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(title)
    fig.suptitle("Raw trials; fitted population means with unadjusted 95% CIs", fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(result, title, destination):
    """Exploratory model diagnostics; they do not certify that assumptions hold."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    try:
        fitted = np.asarray(result.fittedvalues)
        residual = np.asarray(result.model.endog) - fitted
        axes[0].scatter(fitted, residual, s=12, alpha=0.5)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set(xlabel="Conditional fitted value", ylabel="Residual")
        stats.probplot(residual, dist="norm", plot=axes[1])
    except (ValueError, np.linalg.LinAlgError) as exc:
        plt.close(fig)
        return f"Conditional residual diagnostics unavailable: {exc}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return None


# SECTION: running
def run_analysis(data_path, output_dir="results", sheet_name="Sheet1", make_plots=True,
                 effect_bootstrap_reps=1000, effect_seed=20261026, config=None):
    """Run all five models; export fixed effects, random covariance, Wald tests,
    contrasts, counts, R², warnings and software/input provenance.
    """
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            "Analysis dataset not found. Place the original trial-level Excel file at "
            "data/prt_trials.xlsx or set DATA_PATH/--data to its location. "
            "The sharing package does not contain the experimental data."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if data_path.suffix.lower() == ".csv":
        raw = pd.read_csv(data_path)
    elif data_path.suffix.lower() == ".tsv":
        raw = pd.read_csv(data_path, sep="\t")
    elif data_path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(data_path, sheet_name=sheet_name)
    else:
        raise ValueError("Use an .xlsx, comma-separated .csv, or tab-separated .tsv input.")
    data, flow = prepare_data(raw, config=config)
    cfg = data.attrs["prt_settings"]
    all_tables = {"input_flow": flow}
    fits, fit_records = {}, []
    configurations = [
        (key, *definition, False) for key, definition in cfg["distance_metrics"].items()
    ] + [(key, *definition, True) for key, definition in cfg["learning_metrics"].items()]
    for model_index, (key, metric, event, title, learning) in enumerate(configurations):
        selected, counts = select_outcome(data, metric, event, learning)
        formula = "Metric ~ Trial_centered * Treatment" if learning else "Metric ~ Sex * Treatment"
        random_formula = "~ Trial_centered" if learning else "1"
        title = title.replace("(s)", f"({cfg['latency_unit']})") if learning else title.replace("(cm)", f"({cfg['distance_unit']})")
        result, model_warnings = fit_mixed_model(selected, formula, random_formula)
        tables = {
            "counts": counts,
            "included_rows": selected[["source_row", "Litter_ID", "Treatment", "Sex", "Trial", "Metric"]].copy(),
            "fixed": fixed_effects_table(result),
            "random": random_covariance_table(result),
            "omnibus": omnibus_table(result, learning),
            "r2": nakagawa_r2(result),
            "contrasts": learning_contrasts(result) if learning else distance_contrasts(result),
        }
        if learning:
            tables["slopes"] = learning_slope_estimates(result)
        effect_sizes, bootstrap_diagnostics, bootstrap_draws = effect_size_table(
            result, selected, formula, random_formula, learning,
            bootstrap_reps=effect_bootstrap_reps, seed=effect_seed + model_index,
        )
        tables["effect_sizes"] = effect_sizes
        tables["effect_bootstrap"] = bootstrap_diagnostics
        # Bootstrap draws can be large; save them as a separate CSV, not an Excel sheet.
        bootstrap_draws.to_csv(output_dir / f"{key}_effect_bootstrap_draws.csv", index=False)
        for suffix, table in tables.items():
            all_tables[f"{key}_{suffix}"] = table
        fits[key] = {"result": result, "data": selected, "tables": tables}
        (output_dir / f"{key}_model_summary.txt").write_text(
            str(result.summary()) + "\n\n" + "\n".join(model_warnings), encoding="utf-8"
        )
        if make_plots:
            if learning:
                plot_learning(result, selected, title, output_dir / f"{key}_fit.png")
            else:
                plot_distance(result, selected, title, output_dir / f"{key}_fit.png")
            diagnostic_warning = plot_diagnostics(result, title, output_dir / f"{key}_diagnostics.png")
            if diagnostic_warning:
                model_warnings.append(diagnostic_warning)
        if learning and event is not None and cfg["deadline"] is not None and selected["Metric"].ge(float(cfg["deadline"])).any():
            model_warnings.append(f"Event=1 includes a latency >= deadline ({cfg['deadline']} {cfg['latency_unit']}): check endpoint/event coding.")
        fit_records.append({"analysis": key, "formula": formula, "random_formula": random_formula,
                            "REML": True, "converged": bool(result.converged),
                            "n_trials": len(selected), "n_litters": selected["Litter_ID"].nunique(),
                            "warnings": " | ".join(model_warnings)})
        print(f"\n{key}: {len(selected)} trials, {selected['Litter_ID'].nunique()} litters; converged")
        print(tables["omnibus"].to_string(index=False))
        print(tables["contrasts"].to_string(index=False))
        print(tables["effect_sizes"].to_string(index=False))
        for message in model_warnings:
            print(f"Fit note ({key}): {message}")
    all_tables["fit_diagnostics"] = pd.DataFrame(fit_records)
    for name, table in all_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    # Short, unique sheet names avoid the >31-character warnings in the source.
    with pd.ExcelWriter(output_dir / "PRT_mixed_model_results.xlsx", engine="openpyxl") as writer:
        for name, table in all_tables.items():
            if len(name) > 31:
                raise ValueError(f"Excel sheet name too long: {name}")
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                header = str(column[0].value)
                sheet.column_dimensions[column[0].column_letter].width = min(48, max(15, len(header) + 2))
                if header.startswith("p_"):
                    for cell in column[1:]:
                        cell.number_format = "0.000E+00"
    metadata = {
        "analysis_version": ANALYSIS_VERSION,
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "sheet": sheet_name, "input_rows": len(raw),
        "python": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in
                     ("numpy", "pandas", "scipy", "statsmodels", "patsy", "matplotlib", "openpyxl")},
        "distance_multiplicity": "BH across the configured comparison family separately per outcome; Bonferroni also exported",
        "learning_multiplicity": "Nominal p_raw plus optional BH within baseline/slope families of nonreference treatments vs reference",
        "learning_scope": "Each outcome uses its configured event filter; censoring is not modeled by the Gaussian LMM",
        "distance_scope": "Each outcome follows its configured event inclusion rule",
        "learning_origin": f"Trial_centered = Trial - {cfg['trial_origin']:g} for both fixed and random effects",
        "r2_fixed_variance": "Sample variance of fixed predictions (ddof=1)",
        "r2_random_variance": "Mean diagonal of Z @ cov_re @ Z.T",
        "effect_size_standardization": "Fixed-effect contrast divided by residual SD; slopes retain per-trial units",
        "effect_size_raw_CI": "Unadjusted 95% Wald interval from fitted model",
        "effect_size_standardized_CI": "Unadjusted 95% percentile parametric bootstrap; sigma re-estimated in each refit",
        "effect_bootstrap_reps": int(effect_bootstrap_reps), "effect_bootstrap_base_seed": int(effect_seed),
        "analysis_settings": cfg,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nResults written to the output folder:", output_dir.name)
    return fits, all_tables


# SECTION: cli
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prt_trials.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--sheet", default="Sheet1", help="Worksheet name; default Sheet1")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--effect-bootstrap", type=int, default=1000,
                        help="Parametric refits per outcome for standardized-effect CIs; 0 skips CIs")
    parser.add_argument("--effect-seed", type=int, default=20261026)
    parser.add_argument("--config", type=Path, help="Optional JSON for columns, outcomes, levels, references, units and comparisons")
    args = parser.parse_args()
    run_analysis(args.data, args.output, args.sheet, make_plots=not args.no_plots,
                 effect_bootstrap_reps=args.effect_bootstrap, effect_seed=args.effect_seed, config=args.config)


if __name__ == "__main__":
    main()
