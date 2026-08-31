# Logistic Regression — Pima Indians Diabetes Classifier

Predicts **diabetes (1)** vs **no diabetes (0)** from eight clinical measurements
using `sklearn.linear_model.LogisticRegression(max_iter=1000)`, with a Streamlit
UI for the metrics, the coefficient interpretation and live predictions.

The point of this project is not just to hit a number — it is to handle the
dataset's well-known **zero-as-missing** trap correctly, keep the preprocessing
leakage-free, and read the coefficients back out in plain English.

---

## Setup

Requires Python 3.11+ (built and tested on 3.13).

```bash
cd logreg-diabetes

# create and activate a virtualenv
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Run

Run the pipeline in order. Each script prints its own verification output.

```bash
python data_utils.py     # M1-M2: download, assign columns, audit zeros
python train.py          # M3-M5: split, impute, scale, fit -> model.pkl, scaler.pkl, imputer.pkl
python evaluate.py       # M6:    metrics.json, confusion_matrix.png, roc_curve.png
python interpret.py      # M7:    coefficients.md
streamlit run app.py     # M8:    the UI
```

Verify everything at once:

```bash
python selftest.py       # all 10 milestone self-tests
python m9_launch.py      # really boots `streamlit run app.py --server.headless true`
```

`app.py` regenerates any missing artifact on first load, so `streamlit run app.py`
works from a clean checkout on its own.

---

## Data source

- **Dataset:** Pima Indians Diabetes (768 patients, 8 features + `Outcome`),
  originally from the UCI ML Repository / National Institute of Diabetes and
  Digestive and Kidney Diseases, 1988.
- **URL used:**
  `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
- Cached on first run to `data/pima-indians-diabetes.csv`. The raw file has **no
  header row**; `data_utils.COLUMNS` assigns the nine names.
- If the download fails and no cache exists, `data_utils` generates a **synthetic**
  dataset with the same schema and quirks, tags the run `source="synthetic"`, and
  surfaces a red warning in the UI plus an `is_synthetic` flag in `metrics.json`.
  The shipped metrics below came from the **real** data (`source="remote"`).

### The zero-as-missing problem

Five columns cannot legitimately be `0` in a living person, so a `0` there is a
recording gap rather than a measurement:

| Column | Zeros | % of rows |
|---|---:|---:|
| Glucose | 5 | 0.65% |
| BloodPressure | 35 | 4.56% |
| SkinThickness | 227 | 29.56% |
| Insulin | 374 | 48.70% |
| BMI | 11 | 1.43% |

376 of 768 rows (49.0%) are affected. Handling:

- Those zeros become `NaN`, then are filled by `SimpleImputer(strategy="median")`
  **fitted on the training split only**.
- **Not dropped** — that would discard half the dataset and bias it toward
  patients who happened to receive a full insulin panel.
- **Not imputed by `Outcome` class** — conditioning the features on the label is
  target leakage and is impossible at inference time, since `Outcome` is what we
  are predicting. Logged as a deliberate deviation in `PROGRESS_LOG.md` (M2).
- `Pregnancies == 0` and `Age` are left untouched; zero pregnancies is a real
  value.

---

## Sample metrics

Held-out test set, 80/20 stratified split, `random_state=42`, n_test = 154.
Positive class is `1` = diabetes. Regenerate with `python evaluate.py`.

| Metric | Value |
|---|---:|
| Accuracy | **0.7078** |
| Precision (diabetes) | 0.6000 |
| Recall (diabetes) | 0.5000 |
| F1-score (diabetes) | **0.5455** |
| ROC-AUC | **0.8130** |

Confusion matrix (rows = true, cols = predicted, order `['no diabetes', 'diabetes']`):

```
[[82, 18]      TN=82  FP=18
 [27, 27]]     FN=27  TP=27
```

Stratified 5-fold cross-validation over all 768 rows, same pipeline wrapped in a
`Pipeline` so the imputer and scaler refit per fold:

| Metric | Value |
|---|---:|
| CV accuracy | 0.7721 ± 0.0166 |
| CV ROC-AUC | 0.8366 ± 0.0203 |

### Why accuracy is 0.7078 and not in the 0.72–0.80 band

The single-split accuracy lands 0.012 below the expected floor. This was
investigated rather than waved away (full write-up in `PROGRESS_LOG.md` M6):

- ROC-AUC **is** in band (0.8130), so the model ranks patients correctly — a
  scaling or label-leakage bug would depress AUC too, and the M4 self-test proves
  the test split is transformed with the *training* scaler.
- Leaving the zeros in gives 0.7143 — also below band, so imputation is not the
  limiter.
- With n=154 the binomial standard error is ~0.037, giving a 95% CI of
  **[0.6360, 0.7796]**, which overlaps the band.
- Across `random_state` 0–29 the same pipeline averages 0.7654 accuracy, and
  seed 42 — which the spec pins — is the **worst** of those 30 draws.

No hyperparameter, threshold or seed was tuned to chase the number. The sanity
band is instead asserted against the low-variance 5-fold CV mean (0.7721).

The real weakness is recall: the model misses 27 of 54 true diabetics at the
default 0.5 threshold. The UI exposes a threshold slider so you can trade
precision for recall yourself.

---

## What the model learned

Features are standardised before fitting, so each coefficient is the change in
the log-odds of diabetes per **one standard deviation** increase in that feature.
`exp(coefficient)` is the odds ratio. Full table and prose in `coefficients.md`.

| Rank | Feature | Coefficient | Odds ratio |
|---:|---|---:|---:|
| 1 | Glucose | +1.1826 | 3.2627 |
| 2 | BMI | +0.6887 | 1.9910 |
| 3 | Pregnancies | +0.3774 | 1.4586 |
| 4 | DiabetesPedigreeFunction | +0.2333 | 1.2628 |
| 5 | Age | +0.1478 | 1.1593 |
| 6 | Insulin | −0.0661 | 0.9360 |
| 7 | BloodPressure | −0.0441 | 0.9568 |
| 8 | SkinThickness | +0.0283 | 1.0287 |

Intercept `-0.8722`. Glucose dominates: a one-SD rise of about 30 mg/dL more than
triples the odds of diabetes. The three weakest weights sit within ±0.07 of zero;
for `Insulin` and `SkinThickness` that is unsurprising given 48.7% and 29.6% of
their values were imputed, so the negative `Insulin` sign is noise around zero
rather than a protective effect.

---

## Project layout

```
logreg-diabetes/
├── data_utils.py           # load, assign columns, zero-as-missing -> NaN, audit
├── train.py                # split, impute, scale, fit; saves the three .pkl artifacts
├── evaluate.py             # all metrics + CV + plots -> metrics.json, two PNGs
├── interpret.py            # coefficient table + plain-English top 5 -> coefficients.md
├── predict.py              # the single shared inference path used by app + tests
├── app.py                  # Streamlit UI
├── selftest.py             # the 10 milestone self-tests
├── m9_launch.py            # boots the real headless Streamlit server
├── requirements.txt
├── README.md
├── PROGRESS_LOG.md         # milestone-by-milestone log + final audit
├── coefficients.md         # generated
├── model.pkl               # generated — LogisticRegression(max_iter=1000)
├── scaler.pkl              # generated — StandardScaler fitted on train
├── imputer.pkl             # generated — SimpleImputer(median) fitted on train
├── metrics.json            # generated
├── train_info.json         # generated — split sizes, medians, scaler stats, n_iter_
├── confusion_matrix.png    # generated
├── roc_curve.png           # generated
└── data/                   # cached raw CSV
```

`imputer.pkl` and `train_info.json` are additions to the file list in the original
spec: the imputer has to be persisted alongside the scaler or a live UI prediction
could not reproduce the training transform, and `train_info.json` records the
learned medians and scaler statistics for auditability.

---

## Pipeline order (and why)

```
split  ->  impute (train medians)  ->  scale (train mean/std)  ->  fit
```

The split comes first, and both the imputer and the scaler are fitted on the
training rows only, then applied unchanged to the test rows and to live UI input.
The M4 self-test enforces this: it re-scales the test set with its *own*
`StandardScaler` and asserts the result **differs** from what the pipeline
produced, then confirms `(X_test_imputed - scaler.mean_) / scaler.scale_`
reproduces it exactly.

## Disclaimer

Educational demo on a small 1988 dataset. Not a medical device, and not
diagnostic advice.
