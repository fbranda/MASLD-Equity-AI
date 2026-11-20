# Methods, estimands, and limitations

## Study design

Retrospective analysis of public-use NHANES examinations using two nonoverlapping survey releases: 2017–March 2020 for model development and 2021–August 2023 for temporal validation. The objective is to compare survey-weighted discrimination and **capacity-matched referral ranking**, not to prospectively evaluate VCTE allocation or patient outcomes.

## Eligibility

- Age >=20 years.
- Metabolic risk defined as BMI >=25 kg/m2, self-reported diabetes, or HbA1c >=5.7% (at least one criterion).
- VCTE technical validity: LUAXSTAT=1; at least 10 valid VCTE measures; liver stiffness IQR/median <30%; positive LSM.
- Nonmissing valid AST, ALT, platelets and an analysis weight >0.

This is an **at-risk** sample. It is not defined by histological or clinical MASLD adjudication. Other causes of liver disease cannot be definitively excluded with the available selected variables.

## Endpoints

- Main: LSM >=12 kPa (noninvasive proxy for elevated stiffness / suspected fibrosis risk).
- Secondary: LSM >=8 kPa and CAP >=248 dB/m descriptive sensitivity subset.

No individual is assigned a biopsy-confirmed F3/F4 diagnosis on these criteria alone.

## Scores and models

FIB-4 is computed as age x AST / (platelets x sqrt(ALT)). Penalized logistic regression and calibrated histogram gradient boosting use age, sex, BMI, diabetes, HbA1c, AST, ALT, platelets, and GGT. Models are developed using the earlier release, then scored without refitting in the later release. A reduced-variable analysis excludes GGT and HbA1c. Different feature sets mean improvement over FIB-4 cannot be attributed exclusively to algorithm choice.

### Primary comparisons

- Survey-weighted AUROC / AUPRC; Brier score and probability calibration for the probability-generating ML models.
- Capacity-constrained **rank-order** selection in the temporal validation cohort with a *weighted* quota of 10%, 20%, or 30% of adults.
- The share of elevated-LSM cases detected by each rule and the expected weighted yield of referred individuals.
- Approximate stratified PSU bootstrap intervals calculated with fixed predictions and reweighted respondent/PSU samples.

Weighted referral capacity uses fractional selection of the boundary individual to enforce the quota mathematically. This is a ranking estimand, not a literal operational instruction for splitting a person's diagnostic test.

### Equitable access audit

Survey-weighted referral fractions and capture rates are explored by age and sex (and, where present, demographic categories). Equal sex-specific referral quotas are simulated as an additional resource-allocation scenario. These comparisons are descriptive and should not be equated to fair treatment, equal need, or causally identified discrimination.

## Key limitations

1. Complex survey design: sampling and nonresponse, PSU/strata structure, and differences in NHANES survey weights affect interpretation; residual bias cannot be excluded.
2. Proxy outcome: liver stiffness is not histology, and no definitive fibrosis stage is inferred.
3. Restricted eligibility and missing records: selection may limit external transportability.
4. Clinical workflow: FIB-4 top-quota ranking is **not** the same policy as age-adjusted or sequential FIB-4 clinical guideline pathways.
5. Model comparability: richer predictor set of the AI models confounds the algorithm-only contrast.
6. Calibration: validation-period probability predictions are not assured to be clinically calibrated.
7. Uncertainty: cluster bootstrap estimates do not incorporate full model refitting and model-selection uncertainty.
8. Equity: observational subgroup patterns cannot establish clinical equality or reduction in structural barriers.
9. No prospective economics: simulated 20% capacity is illustrative and not a measured real-world service quota.
10. Reproducibility: reruns on different library versions and platforms can yield modestly different estimates; inspect inputs and thresholds, and independently audit the analysis before journal submission.

## Reporting and appropriate use

This software is for methodological research; do not use its scores as clinical care recommendations. The associated manuscript should be reviewed by a hepatology specialist and an independent analyst before submission. Reporting should be aligned with TRIPOD+AI; study limitations and target population must remain explicit.
