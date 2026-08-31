# Logistic Regression coefficients — diabetes classifier

Every number on this page is read straight off the fitted
`LogisticRegression` in `model.pkl` (`model.coef_`, `model.intercept_`).
Regenerate with `python interpret.py`.

## How to read these numbers

The model was trained on **standardised** features, so each coefficient is
the change in the **log-odds** of diabetes caused by a **one-standard-deviation**
increase in that feature, holding the other seven fixed. `exp(coefficient)` is
the **odds ratio**: multiply the current odds by it to get the new odds.

- positive coefficient → odds ratio above 1 → **increases** the odds of diabetes
- negative coefficient → odds ratio below 1 → **decreases** the odds of diabetes

Because the inputs share a common scale, the magnitudes are comparable, so
sorting by |coefficient| is a fair influence ranking.

**Intercept:** -0.8722 — the log-odds for a patient sitting exactly at
the training mean on all eight features. That is odds of 0.4180, i.e. a
predicted probability of diabetes of 0.2948 for an average patient,
which tracks the dataset's 34.9% positive rate.

## Full coefficient table (sorted by |coefficient|, descending)

| Rank | Feature | Coefficient (log-odds per SD) | \|Coefficient\| | Odds ratio e^coef | Direction of effect |
|---:|---|---:|---:|---:|---|
| 1 | Glucose | +1.1826 | 1.1826 | 3.2627 | increases odds of diabetes |
| 2 | BMI | +0.6887 | 0.6887 | 1.9910 | increases odds of diabetes |
| 3 | Pregnancies | +0.3774 | 0.3774 | 1.4586 | increases odds of diabetes |
| 4 | DiabetesPedigreeFunction | +0.2333 | 0.2333 | 1.2628 | increases odds of diabetes |
| 5 | Age | +0.1478 | 0.1478 | 1.1593 | increases odds of diabetes |
| 6 | Insulin | -0.0661 | 0.0661 | 0.9360 | decreases odds of diabetes |
| 7 | BloodPressure | -0.0441 | 0.0441 | 0.9568 | decreases odds of diabetes |
| 8 | SkinThickness | +0.0283 | 0.0283 | 1.0287 | increases odds of diabetes |

## Plain-English reading of the top 5 features

1. **Glucose** (coefficient +1.1826, odds ratio 3.2627) — Glucose is plasma glucose concentration at 2 hours in an oral glucose tolerance test. The coefficient is **positive**, so a one-standard-deviation rise in Glucose (one SD is about 29.98 mg/dL), with every other feature unchanged, **increases the odds** of diabetes by about 226.3% (odds multiplied by 3.2627). Higher Glucose therefore pushes the prediction towards diabetes.

2. **BMI** (coefficient +0.6887, odds ratio 1.9910) — BMI is body mass index (kg/m^2). The coefficient is **positive**, so a one-standard-deviation rise in BMI (one SD is about 6.82 kg/m^2), with every other feature unchanged, **increases the odds** of diabetes by about 99.1% (odds multiplied by 1.9910). Higher BMI therefore pushes the prediction towards diabetes.

3. **Pregnancies** (coefficient +0.3774, odds ratio 1.4586) — Pregnancies is the number of times the patient has been pregnant. The coefficient is **positive**, so a one-standard-deviation rise in Pregnancies (one SD is about 3.31 pregnancies), with every other feature unchanged, **increases the odds** of diabetes by about 45.9% (odds multiplied by 1.4586). Higher Pregnancies therefore pushes the prediction towards diabetes.

4. **DiabetesPedigreeFunction** (coefficient +0.2333, odds ratio 1.2628) — DiabetesPedigreeFunction is a score summarising diabetes history in the patient's relatives. The coefficient is **positive**, so a one-standard-deviation rise in DiabetesPedigreeFunction (one SD is about 0.33 pedigree units), with every other feature unchanged, **increases the odds** of diabetes by about 26.3% (odds multiplied by 1.2628). Higher DiabetesPedigreeFunction therefore pushes the prediction towards diabetes.

5. **Age** (coefficient +0.1478, odds ratio 1.1593) — Age is age in years. The coefficient is **positive**, so a one-standard-deviation rise in Age (one SD is about 11.82 years), with every other feature unchanged, **increases the odds** of diabetes by about 15.9% (odds multiplied by 1.1593). Higher Age therefore pushes the prediction towards diabetes.

## The rest

The remaining features carry smaller weights. The weakest is **SkinThickness** at +0.0283 (odds ratio 1.0287), close enough to zero that it barely moves a prediction once the stronger features are accounted for.

## Caveats

- These are **associations in this dataset, not causal effects**. A positive
  coefficient does not mean changing the feature would change a patient's risk.
- The five zero-coded columns were median-imputed (see `PROGRESS_LOG.md` M2).
  `Insulin` was missing for 48.7% of rows and `SkinThickness` for 29.6%, so
  their coefficients are estimated on heavily reconstructed data and should be
  trusted less than the others.
- Coefficients are correlation-sensitive: `BMI` and `SkinThickness` are strongly
  related, so the model splits the shared signal between them somewhat
  arbitrarily.
- This is a teaching model, not a diagnostic device.

## Model these coefficients came from

- Test-set accuracy: 0.7078
- Test-set ROC-AUC: 0.8130
- Test-set F1 (diabetes): 0.5455
- Evaluated on n = 154 held-out patients
