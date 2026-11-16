## Optimization of Machine Downtime

*A data analysis project to better understand machine failures and assess the potential of predictive alerts.* 

It combines sensor data exploration, classification model comparison, and an illustrative economic assessment. The focus is operational: which signals are worth monitoring, how maintenance actions might be prioritized, and what value a reliable alert could bring.

> **Key takeaway:** On the random train/test split used here, several models distinguish the two classes almost perfectly. This suggests potential worth investigating, but it does not yet demonstrate production performance. A chronological validation and a review of the data protocol are essential before drawing operational conclusions.

## Business problem

Unplanned downtime can cause lost production, emergency repairs, and schedule disruption. This project turns measurements of pressure, temperature, vibration, speed, voltage, torque, and cutting force into information that could help maintenance teams:

- identify sensor variables whose values differ most between failure and non-failure observations;
- compare models that produce a failure risk score;
- illustrate the trade-off between missed failures, preventive maintenance, and false alerts.

The dataset contains **2,500 observations**, **3 machines**, and **16 columns**. The target is nearly balanced: **1,265 “Machine_Failure” observations (50.6%) and 1,235 “No_Machine_Failure” observations (49.4%)**. The data cover **November 24, 2021 to July 3, 2022** (222 calendar days). These are proportions of records in this dataset; without knowing the sampling plan, they should not be interpreted as annual plant failure rates.

## Findings and operational recommendations

### Exploratory data analysis (EDA): data quality and descriptive findings

#### Missing values and data quality

The source data contain **143 missing sensor readings out of 30,000 numeric sensor cells (0.48%)**. Missingness is limited across individual sensors, ranging from **0.24% to 0.84%**. Torque has the most missing readings (21; 0.84%), followed by coolant pressure (19; 0.76%) and air pressure (17; 0.68%). The missing-value chart makes this low overall rate visible.

Two physically implausible readings are also flagged: one negative hydraulic pressure and one negative spindle vibration. The script converts these to missing values rather than treating them as valid measurements. It retains other extreme values because they may be informative failure signals; it reports IQR outlier counts but does not automatically delete observations.

For descriptive plots, missing numeric readings are imputed with the **median for the same machine**, without using the failure label. For model training, median imputation is inside the scikit-learn pipeline and is fitted on training data only. Missingness percentages by target are generally low, though not identical for every sensor; this check alone does not establish that data are missing completely at random. In an operational deployment, sensor dropouts should also be monitored as a data-quality signal.

![Missing values by sensor](outputs/downtime_analysis/figures/01_missing.png)

#### Target balance and descriptive plots

There are **1,265 failure records (50.6%)** and **1,235 non-failure records (49.4%)**. The near balance is useful for an initial model comparison, but it is much higher than one would normally assume for a stream of production-time readings; it may reflect how this dataset was sampled. Therefore, precision, alert volume, and economic value need recalibration against real operating prevalence.

The target, machine, and month count plots describe how records are distributed. The three machines have similar numbers of observations and broadly similar raw failure proportions (about 49–52%), so the current sample does not indicate a clear machine-level priority. Histograms, density curves, and boxplots compare all 12 sensor variables by class. They show that some variables have visibly different distributions between classes, while others overlap substantially. The boxplots and pair plot are useful for spotting spread, skew, and unusual values; extreme readings are not by themselves evidence of a fault mechanism.

The standardized mean-gap chart highlights cutting force, spindle speed, coolant pressure and temperature on one side, and hydraulic pressure and torque on the other. The effect-size plot is descriptive: its direction represents the difference in class averages, not an intervention effect. The distributions can be skewed, so mean differences and median comparisons need not point in exactly the same way; for example, failure records have a higher average spindle speed but a slightly lower median. This is a reason to inspect distributions rather than rely on a single summary statistic.

The correlation heatmap summarizes pairwise linear associations among sensors and the target. Correlation can reveal redundant measurements or patterns for follow-up, but it cannot establish causality. The pressure-by-cutting-force cross-tab also shows that the observed failure fraction changes across sensor bands; these are joint descriptive slices, not validated alarm thresholds. Weekly rates vary around the overall average, but week-level observation counts are not displayed, so apparent spikes require denominator checks before operational interpretation.

![Target distribution and counts by machine and month](outputs/downtime_analysis/figures/02_target_categ.png)

![Sensor distributions by target class](outputs/downtime_analysis/figures/03_hist_hue.png)

![Standardized mean differences between failure and non-failure records](outputs/downtime_analysis/figures/07_bar_effect_size.png)

![Correlation matrix for sensor variables and target](outputs/downtime_analysis/figures/10_corr_matrix.png)

![Failure percentage by hydraulic-pressure and cutting-force bands](outputs/downtime_analysis/figures/11_crosstab_heatmap.png)

#### Statistical tests: Mann–Whitney comparisons

The script applies a two-sided **Mann–Whitney U test** to compare each sensor's observed values between failure and non-failure records. The table below reports the class medians and the p-values from the current run. The test is rank-based and assesses a distributional difference; it does not prove that a sensor causes failure or, on its own, quantify operational importance.

| Sensor | Failure median | Non-failure median | Mann–Whitney p-value | Reading |
|---|---:|---:|---:|---|
| Hydraulic pressure | 80.13 | 120.02 | < 0.001 | Strong distributional difference |
| Coolant pressure | 4.79 | 5.15 | 0.578 | No evidence of a difference in this test |
| Air pressure | 6.52 | 6.50 | 0.614 | No evidence of a difference in this test |
| Coolant temperature | 23.00 | 20.70 | < 0.001 | Strong distributional difference |
| Hydraulic oil temperature | 47.60 | 47.70 | 0.651 | No evidence of a difference in this test |
| Spindle bearing temperature | 35.00 | 35.10 | 0.297 | No evidence of a difference in this test |
| Spindle vibration | 1.00 | 1.02 | 0.384 | No evidence of a difference in this test |
| Tool vibration | 25.33 | 25.56 | 0.804 | No evidence of a difference in this test |
| Spindle speed | 19,964 | 20,188 | < 0.001 | Distributional difference; mean and median directions differ |
| Voltage | 348.00 | 350.00 | 0.967 | No evidence of a difference in this test |
| Torque | 24.61 | 24.79 | < 0.001 | Distributional difference; medians are close |
| Cutting force | 2.84 | 2.71 | < 0.001 | Strong distributional difference |

Four variables—**hydraulic pressure, coolant temperature, torque, and cutting force**—have especially clear class differences in this test; spindle speed is also statistically different despite the small median gap. Coolant pressure, air pressure, hydraulic oil temperature, spindle bearing temperature, both vibration readings, and voltage do not show evidence of a difference at the usual 5% level in this univariate test. A non-significant result does not prove that a variable is useless: it may still contribute through nonlinear effects or interactions in a multivariable model.

The EDA test is run after machine-level median imputation, so imputed values can influence the resulting ranks and p-values; the test should be repeated on observed values as a sensitivity check. The analysis also tests 12 variables separately without a multiple-comparisons correction. P-values should therefore be read alongside the distribution plots and effect sizes, not as a standalone ranking. Large sample sizes can make small differences statistically significant; validation on future data is needed to establish whether they are stable and useful for decisions.

### 1. Sensor signals distinguish the classes in this dataset

The comparison of standardized means shows the largest descriptive differences for **cutting force**, **spindle speed**, **coolant pressure**, **coolant temperature**, **hydraulic pressure**, and **torque**. In this univariate comparison, failure observations have higher average cutting force and spindle speed, while hydraulic pressure and torque are lower. These are associations to investigate, not demonstrated causes.

The XGBoost feature importance ranking highlights **hydraulic pressure**, **torque**, **cutting force**, and **coolant pressure**. This ranking shows which variables the model used to separate the classes; it does not measure causality or the effect of a maintenance action.

**Operational recommendation:** prioritize these four signals in a pilot, compare their trajectories before incidents, and review potential thresholds with process and maintenance teams. A combination of signals and their evolution over time may be more useful than a single fixed threshold.

![Most important variables for the XGBoost model](outputs/downtime_analysis/figures/15_feature_importance.png)

### 2. Differences in risk between machines need further confirmation

The observation counts are similar across the three machines: 874, 808, and 818 records. Their raw failure rates are also close, at roughly **49–52%**. This sample alone does not justify treating one machine as clearly riskier than the others. The comparison should be repeated over longer periods using actual operating time and maintenance histories.

### 3. Weekly failure rates fluctuate, but denominators matter

The weekly chart shows substantial variation around the overall rate of approximately **50.6%**. Some extreme weekly values may be based on few observations; the chart does not show weekly sample counts. These variations are therefore a monitoring lead to investigate, not evidence of seasonality or a time trend.

**Operational recommendation:** add weekly observation counts and operating hours, then review unusual periods with the shop-floor team, including maintenance events, production changes, and planned stoppages.

![Weekly failure rate over time](outputs/downtime_analysis/figures/09_weekly_rate.png)

### 4. Classification results are exceptionally strong

A stratified random split reserves 25% of records for testing (625 observations). At a 0.5 decision threshold, **soft Voting** achieves **99.84% accuracy**, **99.68% recall**, and **1.0000 AUC**; its confusion matrix shows **1 missed failure and no false alerts**. AdaBoost detects all 316 failures in the test set, with 5 false alerts.

These near-perfect scores are encouraging for exploring the approach, but may overstate performance on future data. A random split can place similar observations in both training and test sets. Treat the results as **exploratory** until they are validated on a separate future period, ideally with additional checks by machine and a review of when each input variable becomes available.

![Confusion matrices for the evaluated models](outputs/downtime_analysis/figures/14_confusion_matrices.png)

### 5. The estimated cost depends on error type and assumptions

The script's economic scenario assumes a **€15,000** cost for a missed failure, **€3,000** for planned preventive maintenance, and **€800** for a false alert. On the test set, under these assumptions:

- without a model, the reference cost is **€4,740,000** (316 test failures × €15,000);
- at a 0.5 threshold, soft Voting costs **€960,000** in this scenario, or **79.7%** less than the reference;
- AdaBoost costs **€952,000**: it misses no failures but generates five false alerts.

This is a **simulation**, not realized savings. It depends entirely on the chosen cost assumptions and the test sample. The script selects its “optimal” threshold on the same test set, which makes that cost estimate optimistic; it should not be presented as a promise. Actual failure costs, intervention feasibility, alert lead time, and acceptable alert volume should be agreed with the business.

**Operational recommendation:** define a cost matrix with business stakeholders, choose the threshold on a separate validation period, and then measure costs and alerts on a future holdout period. At minimum, track missed failures, false alerts, warning lead time, and cost by machine.

![Economic scenario at the 0.5 threshold and sensitivity to the decision threshold](outputs/downtime_analysis/figures/16_economic.png)

## Project architecture

The code follows a sequence from raw data to decision-oriented outputs:

1. **Load and profile** — inspect dimensions, data types, duplicates, missing values, and descriptive statistics.
2. **Format and derive fields** — standardize variable names, parse dates, derive calendar features, split the machine identifier, and create the binary `Target`.
3. **Quality checks** — flag physically implausible values and identify outliers without automatically removing extremes that may carry failure signals.
4. **Explore** — compare sensor distributions by class, assess differences by machine, compute standardized mean gaps, weekly rates, correlations, cross-tabulations, and Mann–Whitney tests.
5. **Prepare and model** — place imputation inside the pipeline so it is fitted on training data, encode machine identity, and create a stratified train/test split.
6. **Compare models** — tune Random Forest, Gradient Boosting, XGBoost (or HistGradientBoosting when XGBoost is unavailable), and AdaBoost with randomized search; combine them with soft Voting and Stacking; evaluate accuracy, precision, recall, F1, AUC, ROC curves, and confusion matrices.
7. **Interpret and estimate costs** — report model feature importance and assess costs across decision thresholds and alternative assumptions.
8. **Export** — save figures, model comparison and economic tables, selected hyperparameters, and a cleaned dataset.

Preprocessing is encapsulated in scikit-learn pipelines so that imputation statistics and encoding are learned from training data. The code derives month and day-of-week features but uses a random split; chronological validation is the next necessary step to estimate future performance.

## Repository structure

```text
.
├── data/
│   ├── raw/                       # Local source data
│   └── processed/                 # Prepared data
├── notebooks/
│   └── 01_machine_downtime_analysis.ipynb  # Analysis notebook
├── src/
│   └── analyse_downtime.py        # Executable reference script
├── outputs/
│   └── downtime_analysis/
│       ├── figures/               # 16 analysis figures
│       ├── tableau_comparatif.csv
│       ├── analyse_economique.csv
│       ├── hyperparametres.csv
│       └── Machine_Downtime_clean.csv
├── reports/                       # Archived run results
└── requirements.txt
```

The file named `01_machine_downtime_analysis` in the repository is a **Jupyter notebook (`.ipynb`)**. The executable Python reference script is `src/analyse_downtime.py`.

## Figures at a glance

The figures below are embedded from `outputs/downtime_analysis/figures/`:

| Figure | What it shows |
|---|---|
| `02_target_categ.png` | Target distribution and failure counts by machine and month |
| `03_hist_hue.png`, `04_kde_hue.png`, `05_boxplot_target.png` | Sensor distributions by target class |
| `06_boxplot_machine_hue.png` | Distributions by machine and class |
| `07_bar_effect_size.png`, `08_pairplot_top4.png` | Variables with the largest class differences and pairwise patterns |
| `09_weekly_rate.png` | Weekly failure rate over time |
| `10_corr_matrix.png`, `11_crosstab_heatmap.png` | Correlations and failure rates by pressure/cutting-force bands |
| `12_roc_curves.png`, `13_roc_zoom.png`, `14_confusion_matrices.png` | Model discrimination and classification errors |
| `15_feature_importance.png` | Variables highlighted by the selected tree-based model |
| `16_economic.png` | Cost scenario and the effect of the decision threshold |

## Run the analysis

Python 3.10 or later is recommended. From the project root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/analyse_downtime.py
```

By default, the script reads `data/raw/Machine_Downtime.csv` and writes results to `outputs/downtime_analysis/`. To specify different paths:

```bash
python src/analyse_downtime.py --data "path/to/Machine_Downtime.csv" --out-dir "path/to/output"
```

## Limitations and next steps

- Use a **time-based split**: train on past records, select the threshold on a validation period, and reserve the most recent dates for the final test.
- Check for variables recorded after a failure and confirm that the observations represent real operating conditions.
- Evaluate by machine, time period, and false-alert rate; report uncertainty intervals.
- Replace illustrative economic assumptions with documented costs, then run the alert in observation mode before using it to drive maintenance decisions.
- The project predicts a failure class; it does not predict downtime duration, an exact failure date, or an optimal maintenance schedule.

The data are stored locally in this project. Confirm their provenance and redistribution rights before publishing or sharing them.
