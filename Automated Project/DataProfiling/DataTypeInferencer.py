import pandas as pd
import numpy as np
import warnings

class DataTypeInferencer:
    """
    Intelligent Data Type Inference & Correction
    + Faster datetime detection with common formats
    """

    COMMON_DATE_FORMATS = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y.%m.%d"
    ]

    def __init__(self, dataframe, categorical_threshold=0.05):
        self.df = dataframe
        self.categorical_threshold = categorical_threshold
        self.report = {}

    # --------------------------
    # Helper: Boolean Detection
    # --------------------------
    def _is_boolean(self, series):
        values = series.dropna().astype(str).str.lower().unique()
        return set(values).issubset({"0", "1", "true", "false"})

    # --------------------------
    # Helper: Numeric Detection
    # --------------------------
    def _to_numeric(self, series):
        cleaned = series.astype(str).str.replace(",", "")
        return pd.to_numeric(cleaned, errors="coerce")

    # --------------------------
    # Helper: Datetime Detection
    # --------------------------
    def _to_datetime(self, series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # Try common formats first
            for fmt in self.COMMON_DATE_FORMATS:
                try:
                    dt_series = pd.to_datetime(series, format=fmt, errors="coerce")
                    # if majority converts, return it
                    if dt_series.notna().mean() > 0.9:
                        return dt_series
                except:
                    continue

            # fallback to automatic inference
            return pd.to_datetime(series, errors="coerce")

    # --------------------------
    # Main Inference
    # --------------------------
    def infer(self):
        for col in self.df.columns:
            series = self.df[col]
            inferred_type = "unknown"

            # 1. Boolean
            if self._is_boolean(series):
                self.df[col] = series.astype(str).str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
                inferred_type = "boolean"

            # 2. Numeric
            elif pd.api.types.is_numeric_dtype(series):
                inferred_type = "numeric"

            else:
                numeric_try = self._to_numeric(series)
                if numeric_try.notna().mean() > 0.9:
                    self.df[col] = numeric_try
                    inferred_type = "numeric"

                # 3. Datetime
                else:
                    datetime_try = self._to_datetime(series)
                    if datetime_try.notna().mean() > 0.9:
                        self.df[col] = datetime_try
                        inferred_type = "datetime"

                    # 4. Categorical vs Text
                    else:
                        unique_ratio = series.nunique() / max(len(series), 1)

                        if unique_ratio <= self.categorical_threshold:
                            self.df[col] = series.astype("category")
                            inferred_type = "categorical"
                        else:
                            inferred_type = "text"

            self.report[col] = inferred_type

        return self.df, self.report
