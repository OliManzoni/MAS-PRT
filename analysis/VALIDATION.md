# Computational verification

Release 1.2.0 was checked on synthetic inputs outside this distribution. No
experimental or simulated datasets or results are included in this folder.
The scientific suitability/assumptions must be assessed for each dataset.

Checks completed:

- Analytic Gaussian R² example including random intercept-slope covariance.
- Correct original-scale covariance export versus normalized packed parameters.
- Direct multivariate Wald calculation versus the omnibus wrapper.
- Distance contrasts versus the equivalent cell-means parameterization.
- Trial-origin reparameterization preserving predictions and interaction tests.
- Correct baseline contrasts and residual-standardized effect units/scaling.
- Parametric bootstrap re-estimates the SD; reproducible seeds and percentile
  intervals; failed refits are counted and inadequate intervals withheld.
- Original three-treatment Excel schema, custom two-treatment CSV schema with
  different labels/column names/units and trial origin 0, and four-treatment TSV.
- Learning-only data without sex/distance, and distance-only data without
  latency/event/trial inputs. Contrast family sizes and joint-test df adapt.
- Excel/CSV exports, sample counts and generated fit/diagnostic plots.
- Jupyter notebook schema and all code cells in order in a fresh Python process
  using a custom-column CSV configuration with bootstrap enabled.
- Notebook definitions match the script; distributed outputs/counters are clear.

Bootstrap mechanics were exercised at small verification counts (up to 20
replicates per representative model), not to estimate scientific interval
precision. The public code defaults to 1,000 refits per outcome, labels lower
counts, and records convergence diagnostics. Experimental analyses and final
bootstrap intervals must be run using the intended experimental dataset.

A native Jupyter-kernel launch was blocked by the verification environment's
local-socket restriction; full code-cell execution used a clean Python process.
No notebook magic commands are used. Tested with Python 3.13.1 and the dependency
versions in requirements.txt. These checks do not certify the assumptions of a
model fitted to a new experimental dataset.
