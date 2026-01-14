import pandas as pd


class SchemaValidator:
    """
    Advanced Schema Detection & Validation
    + Small Data Report
    """

    def __init__(
        self,
        dataframe,
        expected_schema=None,
        strict=False,
        min_columns=1,
        source=None
    ):
        self.df = dataframe
        self.expected_schema = expected_schema
        self.strict = strict
        self.min_columns = min_columns
        self.source = source
        self.report = {}

    # --------------------------
    # Utility
    # --------------------------
    def _fail_or_warn(self, message):
        if self.strict:
            raise ValueError(message)
        else:
            print(f"⚠ {message}")

    # --------------------------
    # 1. Shape Validation
    # --------------------------
    def validate_shape(self):
        if self.df.empty:
            self._fail_or_warn("Dataset has no rows")

        if self.df.shape[1] < self.min_columns:
            self._fail_or_warn("Dataset has too few columns")

        self.report["rows"] = self.df.shape[0]
        self.report["columns"] = self.df.shape[1]

    # --------------------------
    # 2. Column Detection (ORDER PRESERVED)
    # --------------------------
    def detect_columns(self):
        cols = list(self.df.columns)
        self.report["column_order"] = cols
        print(f"Detected columns (order preserved): {cols}")

    # --------------------------
    # 3. Duplicate Columns
    # --------------------------
    def check_duplicate_columns(self):
        duplicates = self.df.columns[self.df.columns.duplicated()].tolist()
        if duplicates:
            self._fail_or_warn(f"Duplicate columns found: {duplicates}")
        self.report["duplicate_columns"] = duplicates

    # --------------------------
    # 4. Empty Column Names
    # --------------------------
    def check_empty_columns(self):
        empty_cols = [c for c in self.df.columns if c == "" or str(c).lower() == "nan"]
        if empty_cols:
            self._fail_or_warn(f"Empty column names detected: {empty_cols}")
        self.report["empty_columns"] = empty_cols

    # --------------------------
    # 5. Schema Consistency (OPTIONAL)
    # --------------------------
    def check_schema_consistency(self):
        if not self.expected_schema:
            return

        missing = [c for c in self.expected_schema if c not in self.df.columns]
        extra = [c for c in self.df.columns if c not in self.expected_schema]

        if missing:
            self._fail_or_warn(f"Missing columns: {missing}")

        self.report["missing_columns"] = missing
        self.report["extra_columns"] = extra

    # --------------------------
    # 6. Index Validation
    # --------------------------
    def validate_index(self):
        if not self.df.index.is_unique:
            self.df.reset_index(drop=True, inplace=True)

    # --------------------------
    # 7. Small Data Report
    # --------------------------
    def generate_small_report(self):
        self.report["source"] = self.source
        self.report["data_types"] = self.df.dtypes.astype(str).to_dict()
        self.report["missing_values"] = self.df.isnull().sum().to_dict()

    # --------------------------
    # Run All
    # --------------------------
    def run(self):
        print("\n Schema Validation Started  ")

        self.validate_shape()
        self.detect_columns()
        self.check_duplicate_columns()
        self.check_empty_columns()
        self.check_schema_consistency()
        self.validate_index()
        self.generate_small_report()

        print("\n Schema Validation Completed \n")
        return self.df, self.report
