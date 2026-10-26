# Mixed-effects analysis of pup-retrieval experiments

The Jupyter notebook `PRT_distance_learning_corrected.ipynb` and equivalent
`prt_mixed_models.py` fit mixed-effects models to trial-level pup-retrieval data.
They accept the user's own Excel, CSV or TSV dataset. No experimental or simulated
data, or saved statistical outputs, are included.

**Scope:** compatible repeated-trial datasets, with trials grouped by dam/litter.
The distance model is `distance ~ sex * treatment` with a litter random intercept.
The latency-trend model is `latency ~ centered_trial * treatment` with correlated
random intercept and trial slope for litter. Input configuration changes names,
outcomes, references, comparison families and units; it does not make these models
appropriate for an arbitrary experimental design.

## Run

1. Use a separate analysis environment and install `requirements.txt`. JupyterLab
   is included; an existing notebook-capable editor can also use that kernel.
2. Open the notebook. Set `DATA_PATH` to your `.xlsx`, `.csv` or `.tsv` file. For
   Excel set `SHEET_NAME` (default `Sheet1`). CSV uses commas, TSV uses tabs.
3. Edit `ANALYSIS_CONFIG` to match your columns, outcomes, labels and reference
   groups. The default `{}` accepts the original spreadsheet column names below.
   `config.example.json` illustrates a different dataset schema; it contains no data.
4. Restart the kernel and run all cells. Inspect the result tables, inclusion
   counts, convergence warnings and residual diagnostic plots.

Equivalent script commands, run from this folder:

```bash
python prt_mixed_models.py --data data/prt_trials.xlsx --output results
python prt_mixed_models.py --data my_trials.csv --config config.example.json --output my_results
```

The analysis does not modify the input file or download data. Reusing an output
folder replaces its prior results. To check the code quickly, set
`EFFECT_BOOTSTRAP_REPS = 0` or pass `--effect-bootstrap 0`. Original-unit effect
estimates/CIs and standardized point estimates remain available; standardized
CIs are explicitly missing. The default 1,000 bootstrap refits **per outcome**
can take longer; use the bootstrap run for final standardized intervals.

## Input configuration

One row represents one trial. Litter/dam IDs must be globally unique across
treatments. Trial numbers must not repeat within a litter; different litter sizes
and gaps in trial sequences are allowed. Labels are retained as supplied after
whitespace trimming, so use consistent case.

At least two treatment levels are required. Distance models also require at
least two sex/factor levels with an estimable factorial design. Learning-only
analyses do not require sex or distance columns. Distance-only analyses do not
require trial, latency or event columns unless an event filter is configured.

Default spreadsheet headers (nonbreaking/duplicate spaces are normalized):

| Header | Internal variable | Interpretation |
|---|---|---|
| `Treatment` | `Treatment` | Treatment/group label |
| `Litter ID` | `Litter_ID` | Dam/litter identifier |
| `Sex` | `Sex` | Pup sex/factor label |
| `Trial #` | `Trial` | Trial order |
| `firstEncounter sec` | `tFE_s` | First-encounter latency, seconds |
| `success sec` | `tR_s` | Retrieval latency, seconds |
| `Event tFE<90` | `Event_FE` | 1 = observed encounter, 0 = no encounter |
| `Event tR<90` | `Event_R` | 1 = observed retrieval, 0 = no retrieval |
| `distanceToFirstEncounter cm` | `Distance_FE_cm` | Distance to encounter, cm |
| `distanceFEtoRet cm` | `Distance_FE_R_cm` | Distance from encounter to retrieval, cm |
| `distance cm` | `Distance_total_cm` | Total recorded distance, cm |

Only fields required by the enabled outcomes must be present. Column mapping is
**internal variable -> input header**, e.g. `"Litter_ID": "Mother"`. New numeric
outcome names can be mapped too. All distances in one run use the configured
`distance_unit`; all latencies use `latency_unit`. These are unit labels, not
unit conversions: supply values in the units specified.

Supported `ANALYSIS_CONFIG`/JSON settings:

- `columns`: overrides/additions to the column map.
- `distance_metrics`, `learning_metrics`: dictionaries of short outcome IDs to
  `[internal_measurement, internal_event_or_null, plot_title]`. Set a dictionary
  to `{}` to disable that analysis type. An event column retains rows with value
  1; `null`/`None` means no event filtering.
- `treatment_reference`, `sex_reference`: explicit reference labels. If omitted,
  SHAM/Female are preferred only if present; otherwise the first observed label
  is the reference. Set references explicitly before interpreting an analysis.
- `treatment_levels`, `sex_levels`: optional complete category orders; observed
  labels must match, and the reference is moved to the first position.
- `trial_origin`: centering origin/first trial (default 1; can be 0).
- `distance_unit`, `latency_unit`: default `cm`, `s`.
- `deadline`: latency-event limit in the chosen latency unit, default 90; `null`
  disables the check. It only flags observed-event latencies at/above that limit.
- `distance_comparisons`: optional list of `[sex1, treatment1, sex2, treatment2,
  label]` comparisons; the estimated difference is first group minus second.

Without custom distance comparisons, the family contains reference-treatment
minus each other treatment within each sex, plus reference-sex minus each other
sex within each treatment. This gives seven contrasts for 2 sexes x 3 treatments,
four for 2 x 2, ten for 2 x 4, etc. The actual `family_n` is exported. Learning
comparisons are each nonreference treatment minus the reference, at baseline and
for trial slope; baseline and slope comparisons form separate optional BH families.

A dataset with custom columns, two treatments and only total distance/retrieval
latency can use, for example:

```python
ANALYSIS_CONFIG = {
    "columns": {"Treatment": "Condition", "Litter_ID": "Mother",
                "Sex": "SexCode", "Trial": "Sequence",
                "Track": "Track_mm", "Latency": "Latency_ms",
                "Observed": "Retrieved"},
    "distance_metrics": {"path": ["Track", None, "Total distance (mm)"]},
    "learning_metrics": {"latency": ["Latency", "Observed", "Retrieval latency (ms)"]},
    "treatment_reference": "Vehicle", "sex_reference": "F",
    "trial_origin": 0, "distance_unit": "mm", "latency_unit": "ms",
    "deadline": 90000,
}
```

Missing group IDs are removed before string conversion, so they cannot become a
literal `nan` group. Missing measurements are counted outcome by outcome. Invalid
numbers, negative latency/distance, reused litter IDs across treatments,
inconsistent configured labels and duplicate trial orders fail with a clear error.

## Models and tests

Fits use REML. Reference coding is retained in predictions and contrasts. In
latency models `Trial_centered = Trial - trial_origin` is used for both fixed and
random slopes. With freely estimated intercept-slope covariance this preserves
fitted curves and interaction tests, apart from fitting tolerance. The treatment
main-effect/omnibus test refers to the configured origin and can differ from the
test at another trial. The random intercept variance/covariance also depends on
this origin.

`wald_test_terms()` runs joint tests on fitted fixed coefficients and their
covariance matrix, accounting for the specified random structure. It does not
apply a new ANOVA to raw measurements or fitted values. For three treatments,
the treatment-by-trial interaction jointly tests two slope differences (2 df),
whereas a single treatment slope contrast is a 1-df question. With more groups,
the joint-test degrees of freedom change accordingly.

Under interactions, simple terms are conditional: the distance treatment term
compares treatments within the reference sex; the sex term compares sexes within
the reference treatment. In latency models the trial term is the reference
slope and the treatment term is evaluated at `trial_origin`. Exported hypotheses
make these meanings explicit. Tests are asymptotic Wald z/chi-square tests; no
Kenward-Roger or Satterthwaite adjustment is introduced.

Distance contrasts use the same fitted factorial model. BH adjustment is over
the configured distance family, separately per outcome. Bonferroni values are
also exported. All model/contrast `p_raw` values are nominal; learning contrasts
also have an optional BH column within each baseline/slope family. The manuscript
must specify the comparisons and correction actually used. Do not choose a
p-value column according to whether it is significant. Intervals are unadjusted
95% intervals, not simultaneous/FDR-adjusted intervals.

Default event filters exclude unobserved/censored latency events, as in the
successful-event analysis. These LMMs describe observed-event latencies and do
not model censoring or event success. Do not feed a censoring deadline as though
it were an observed latency. Total distance uses all recorded values by default,
irrespective of retrieval success; define the measurement endpoint in the study.
Trial-order trends alone do not uniquely identify learning. This notebook does
not perform the separate survival/Cox analysis.

## Variances and R²

Random variances/covariances come from `result.cov_re`, residual variance from
`result.scale`, in original outcome units and the chosen trial coding. Packed
normalized variance parameters are not exported as original-scale variances.
No normal-Wald p-values are assigned to random-variance components.

Gaussian marginal/conditional R² use sample variance of `X beta` (ddof 1), mean
`z_i G z_i'` with `G = cov_re`, and residual variance. For random intercepts this
is the estimated intercept variance; for random slopes it includes the actual
trial values and intercept-slope covariance. Shrunk predicted random effects
are not substituted for G. R² is a model-level descriptive summary, not a causal
percentage attributable to treatment/litter.

## Effect sizes

Original-unit effect estimates are primary: mean differences, baseline latency
differences, sex-by-treatment differences of differences, group slopes and
between-treatment slope differences. Their units and signs are explicit in the
`*_effect_sizes` tables; raw 95% intervals are model-based Wald intervals.

Supplementary standardized effects divide the fitted contrast by the fitted
**residual SD** (denominator exported), excluding between-litter variance. These
are residual-standardized model effects, not independent-groups Cohen d, a
dam-level effect size or a Hedges-g correction. Mean/interaction contrasts have
residual-SD units; slopes retain residual SD **per trial**. No universal size
thresholds are imposed.

Standardized 95% intervals use percentile parametric bootstrap. Each replicate
simulates new litter random effects and residual errors under the fitted
Gaussian LMM, preserving the retained design and litter membership, then refits
the model and re-estimates the SD before standardizing. This includes uncertainty
in the denominator. It is conditional on the selected rows/model assumptions;
it does not simulate the event-success/censoring mechanism or add significance
tests. Intervals are not multiplicity-adjusted.

The default 1,000 refits/outcome is configurable (`--effect-bootstrap`,
`--effect-seed`); zero skips standardized intervals. `*_effect_bootstrap` reports
fit counts/warnings and seed; draws are saved in `*_effect_bootstrap_draws.csv`.
Intervals are withheld if fewer than 90% of requested fits (or fewer than two)
converge with finite effects. Smaller than 1,000 requested refits are labelled
low-count. Inspect warnings and interval stability for final reporting.

No partial eta-squared is obtained by treating a mixed-model Wald chi-square as
an ordinary ANOVA statistic. Relevant contrast magnitudes accompany the omnibus.

## Outputs and verification

Outputs include an Excel workbook, CSV tables, counts/included spreadsheet rows,
model summaries, warnings, effect sizes, R², bootstrap draws, fit/diagnostic plots
and `run_metadata.json` with input hash, versions and resolved configuration.
Small p-values are shown in scientific notation. Population mean CIs are not
prediction intervals for a new dam or trial. Inspect residual diagnostics;
convergence alone does not establish that assumptions hold.

See `VALIDATION.md`. Implementation was checked on synthetic inputs with the
original 3-group layout and different 2/4-group layouts, labels, columns, units,
file formats and enabled outcomes. No synthetic data/results are distributed.
Applicability and assumptions must be checked for each experimental dataset.

## References

- [Statsmodels MixedLMResults](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.html)
- [Wald term tests](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.wald_test_terms.html)
- [Johnson (2014), random-slope R²](https://pmc.ncbi.nlm.nih.gov/articles/PMC4368045/)
- [Nakagawa et al. (2017), R² and ICC](https://doi.org/10.1098/rsif.2017.0213)
- [Standardized effects and SD uncertainty](https://rvlenth.github.io/emmeans/reference/eff_size.html)
