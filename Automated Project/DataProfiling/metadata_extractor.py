import pandas as pd
from datetime import datetime
from collections import Counter

class MetadataExtractor:
    """
    Easy Metadata Extraction for any DataFrame
    """

    def __init__(self, df, source=None, top_n=5):
        self.df = df
        self.source = source
        self.top_n = top_n
        self.metadata = {}

    def extract_basic_info(self):
        self.metadata["rows"] = self.df.shape[0]
        self.metadata["columns"] = self.df.shape[1]

    def extract_column_info(self):
        col_info = {}
        for col in self.df.columns:
            series = self.df[col]
            info = {
                "dtype": str(series.dtype),
                "nulls": series.isna().sum(),
                "unique": series.nunique(),
                "sample": series.dropna().unique()[:self.top_n].tolist()
            }

            # Numeric columns
            if pd.api.types.is_numeric_dtype(series):
                info.update({
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max()
                })

            # Datetime columns
            elif pd.api.types.is_datetime64_any_dtype(series):
                info.update({
                    "min_date": series.min(),
                    "max_date": series.max()
                })

            # Categorical / text columns
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
                values = series.dropna().astype(str)
                info["top_values"] = Counter(values).most_common(self.top_n)
                if not pd.api.types.is_categorical_dtype(series):
                    info["avg_length"] = values.str.len().mean()

            col_info[col] = info

        self.metadata["columns"] = col_info

    def extract_source_info(self):
        self.metadata["source_file"] = self.source
        self.metadata["ingested_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run(self):
        self.extract_basic_info()
        self.extract_column_info()
        self.extract_source_info()
        return self.metadata


# --- Formatted print function ---
def print_metadata(metadata):
    print(f"\nRows: {metadata['rows']}")
    print(f"Columns: {metadata['columns']}")
    print(f"Source: {metadata['source_file']}")
    print(f"Ingested at: {metadata['ingested_at']}\n")

    print("Column Info:")
    for col, info in metadata["columns"].items():
        print(f"\n  Column: {col}")
        for key, value in info.items():
            print(f"    {key}: {value}")
