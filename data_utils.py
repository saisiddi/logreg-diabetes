"""Data loading, column assignment and zero-as-missing handling for the
Pima Indians Diabetes dataset.

Dataset
-------
The raw CSV has **no header row**; the 9 columns are assigned here in the
canonical order used by the UCI / Kaggle distributions of the dataset.

Zero-as-missing
---------------
Five of the eight features cannot legitimately be 0 in a living person:
Glucose, BloodPressure, SkinThickness, Insulin, BMI. A 0 in those columns is a
recording gap, not a measurement. `to_nan()` converts those zeros to NaN so
they can be imputed downstream (see `train.py`) instead of being fed to the
model as if a patient had a BMI of zero.

Pregnancies may legitimately be 0 (nulliparous) and Age is never 0 in this
dataset, so neither is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "pima-indians-diabetes.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

TARGET = "Outcome"
FEATURES = [c for c in COLUMNS if c != TARGET]

# Columns where a 0 is biologically impossible => encodes missing data.
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

TARGET_NAMES = ["no diabetes", "diabetes"]

EXPECTED_SHAPE = (768, 9)

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class Dataset:
    """Loaded dataset plus provenance metadata used by the UI and the logs."""

    df: pd.DataFrame  # features with zeros already converted to NaN, plus Outcome
    source: str  # "remote", "cache" or "synthetic"
    raw: pd.DataFrame  # untouched copy, zeros intact (used by the M2 audit)

    @property
    def X(self) -> pd.DataFrame:
        return self.df[FEATURES]

    @property
    def y(self) -> pd.Series:
        return self.df[TARGET]

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURES)

    @property
    def target_names(self) -> list[str]:
        return list(TARGET_NAMES)

    @property
    def n_samples(self) -> int:
        return int(self.df.shape[0])

    @property
    def n_features(self) -> int:
        return len(FEATURES)

    def class_balance(self) -> dict[str, int]:
        counts = self.y.value_counts().sort_index()
        return {TARGET_NAMES[int(i)]: int(c) for i, c in counts.items()}

    def feature_bounds(self) -> pd.DataFrame:
        """Real min / max / median per feature, ignoring the impossible zeros.

        Used to bound the Streamlit sliders so a user cannot enter a BMI of 0.
        """
        X = self.X
        return pd.DataFrame(
            {
                "min": X.min(skipna=True),
                "max": X.max(skipna=True),
                "median": X.median(skipna=True),
            }
        )


def _download(url: str = DATA_URL, timeout: int = 30) -> str:
    """Fetch the raw CSV text. Raises on any network/HTTP problem."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    if not text.strip():
        raise ValueError("downloaded CSV was empty")
    return text


def _make_synthetic(n: int = 768, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Fallback dataset matching the Pima schema, used only if download fails.

    Deliberately reproduces the dataset's quirks (zero-coded missing values,
    ~35% positive rate) so the rest of the pipeline is exercised identically.
    """
    rng = np.random.default_rng(random_state)
    y = rng.binomial(1, 0.349, size=n)
    shift = y.astype(float)

    df = pd.DataFrame(
        {
            "Pregnancies": np.clip(rng.poisson(3.3 + 1.2 * shift, n), 0, 17),
            "Glucose": np.clip(rng.normal(110 + 32 * shift, 26, n), 44, 199),
            "BloodPressure": np.clip(rng.normal(70 + 5 * shift, 12, n), 24, 122),
            "SkinThickness": np.clip(rng.normal(27 + 4 * shift, 10, n), 7, 99),
            "Insulin": np.clip(rng.normal(130 + 40 * shift, 90, n), 14, 846),
            "BMI": np.clip(rng.normal(30.9 + 4.5 * shift, 6.8, n), 18.2, 67.1),
            "DiabetesPedigreeFunction": np.clip(
                rng.gamma(2.0, 0.22, n) + 0.08 * shift, 0.078, 2.42
            ),
            "Age": np.clip(rng.normal(31 + 6 * shift, 11, n), 21, 81).round(),
            "Outcome": y,
        }
    )

    # Re-introduce the zero-coded missingness at roughly the real rates.
    for col, rate in [
        ("Glucose", 0.007),
        ("BloodPressure", 0.045),
        ("SkinThickness", 0.295),
        ("Insulin", 0.487),
        ("BMI", 0.014),
    ]:
        mask = rng.random(n) < rate
        df.loc[mask, col] = 0

    df["Glucose"] = df["Glucose"].round()
    df["BloodPressure"] = df["BloodPressure"].round()
    df["SkinThickness"] = df["SkinThickness"].round()
    df["Insulin"] = df["Insulin"].round()
    df["BMI"] = df["BMI"].round(1)
    df["DiabetesPedigreeFunction"] = df["DiabetesPedigreeFunction"].round(3)
    return df[COLUMNS].astype(
        {c: "int64" for c in ["Pregnancies", "Glucose", "BloodPressure",
                              "SkinThickness", "Insulin", "Age", "Outcome"]}
    )


def load_raw(force_download: bool = False, allow_synthetic: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (raw DataFrame with COLUMNS assigned, source tag).

    Tries the network first, then the on-disk cache, then synthetic fallback.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if RAW_CSV.exists() and not force_download:
        df = pd.read_csv(RAW_CSV, header=None, names=COLUMNS)
        return df, "cache"

    try:
        text = _download()
        RAW_CSV.write_text(text, encoding="utf-8", newline="")
        df = pd.read_csv(RAW_CSV, header=None, names=COLUMNS)
        return df, "remote"
    except Exception as exc:  # network down, DNS blocked, 404, ...
        if RAW_CSV.exists():
            df = pd.read_csv(RAW_CSV, header=None, names=COLUMNS)
            return df, "cache"
        if not allow_synthetic:
            raise
        print(f"[data_utils] WARNING download failed ({type(exc).__name__}: {exc}).")
        print("[data_utils] WARNING falling back to SYNTHETIC data matching the Pima schema.")
        df = _make_synthetic()
        SYNTH_CSV = DATA_DIR / "pima-synthetic-fallback.csv"
        df.to_csv(SYNTH_CSV, index=False, header=False)
        return df, "synthetic"


def zero_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and % of biologically-impossible zeros (M2)."""
    n = len(raw)
    rows = []
    for col in ZERO_AS_MISSING:
        cnt = int((raw[col] == 0).sum())
        rows.append({"column": col, "zero_count": cnt, "zero_pct": round(100.0 * cnt / n, 2)})
    return pd.DataFrame(rows).set_index("column")


def nan_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-column count of true NaNs across all 9 columns (M2)."""
    na = raw.isna().sum()
    return pd.DataFrame({"nan_count": na.astype(int)})


def to_nan(raw: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zeros with NaN in the five affected columns."""
    df = raw.copy()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


def load_data(force_download: bool = False) -> Dataset:
    """Full M1 + M2 load: fetch, assign columns, convert impossible zeros to NaN."""
    raw, source = load_raw(force_download=force_download)
    return Dataset(df=to_nan(raw), source=source, raw=raw)


def split_data(dataset: Dataset, test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE):
    """80/20 stratified split (M3)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        stratify=dataset.y,
        random_state=random_state,
    )


if __name__ == "__main__":
    ds = load_data()

    print("=== M1: load & column assignment ===")
    print(f"Source            : {ds.source}"
          + ("  (SYNTHETIC FALLBACK)" if ds.is_synthetic else ""))
    print(f"df.shape          : {ds.raw.shape}")
    print(f"Columns assigned  : {list(ds.raw.columns)}")
    print(f"Expected shape    : {EXPECTED_SHAPE} -> "
          f"{'MATCH' if ds.raw.shape == EXPECTED_SHAPE else 'DEVIATION'}")
    print("\ndf.head():")
    print(ds.raw.head().to_string())
    print(f"\ndtypes:\n{ds.raw.dtypes.to_string()}")
    print(f"\nClass balance     : {ds.class_balance()}")

    print("\n=== M2: missing / zero audit ===")
    print("True NaN counts per column (raw file):")
    print(nan_audit(ds.raw).to_string())
    print("\nBiologically-impossible zeros (0 = missing reading):")
    audit = zero_audit(ds.raw)
    print(audit.to_string())
    print(f"\nTotal impossible zeros : {int(audit['zero_count'].sum())}")
    print("Rows with >=1 impossible zero : "
          f"{int((ds.raw[ZERO_AS_MISSING] == 0).any(axis=1).sum())} / {len(ds.raw)}")
    print("\nAfter to_nan() conversion, NaN counts:")
    print(nan_audit(ds.df).to_string())
    print("\nHandling strategy: median imputation fitted on the TRAINING split only "
          "(see train.py / PROGRESS_LOG.md M2).")
