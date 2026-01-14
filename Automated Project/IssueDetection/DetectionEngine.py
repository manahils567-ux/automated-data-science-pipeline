import pandas as pd
import numpy as np
import re
from datetime import datetime
from .IssueObject import DataIssue

class IssueDetectionEngine:
    def __init__(self, df):
        self.df = df
        self.issues = []

    # 1. MISSING DATA (Strict Nulls)
    def check_missing_data(self):
        for col in self.df.columns:
            count = self.df[col].isna().sum()
            if count > 0:
                self.issues.append(DataIssue("MISSING_VAL", col, "Missing Data", "High", f"{count} null values.", []).to_dict())

    # 2. PROXY MISSINGNESS (Placeholder tokens)
    def check_proxy_missingness(self):
        tokens = ["?", "unknown", "n/a", "none", "nan", "null", "."]
        for col in self.df.select_dtypes(include=['object']).columns:
            matches = self.df[col].astype(str).str.lower().str.strip().isin(tokens).sum()
            if matches > 0:
                self.issues.append(DataIssue("PROXY_MISSING", col, "Proxy Missingness", "Medium", f"Found {matches} placeholder tokens.", []).to_dict())

    # 3. NUMERIC VALIDITY (Mathematical Logic)
    def check_numeric_validity(self):
        if "age" in "".join(self.df.columns).lower():
            for col in [c for c in self.df.columns if "age" in c.lower()]:
                inv = pd.to_numeric(self.df[col], errors='coerce')
                neg = inv[inv < 0]
                if not neg.empty:
                    self.issues.append(DataIssue("NEG_AGE", col, "Numeric Validity", "High", "Negative age detected.", neg.tolist()).to_dict())

    # 4. RANGE VIOLATIONS (Mathematical Boundaries)
    def check_range_violations(self):
        for col in self.df.columns:
            if any(x in col.lower() for x in ["pct", "percent", "probability"]):
                val = pd.to_numeric(self.df[col], errors='coerce')
                out = val[val > 100]
                if not out.empty:
                    self.issues.append(DataIssue("RANGE_EXCEEDED", col, "Range Violation", "High", "Value exceeds 100%.", out.tolist()).to_dict())

    # 5. TIME-TRAVEL ERRORS (Future Dates)
    def check_time_travel(self):
        for col in self.df.select_dtypes(include=['datetime64', 'object']).columns:
            dates = pd.to_datetime(self.df[col], errors='coerce')
            future = dates[dates > datetime.now()]
            if not future.empty:
                self.issues.append(DataIssue("FUTURE_DATE", col, "Time-Travel Error", "Medium", "Dates set in the future.", future.head(1).astype(str).tolist()).to_dict())

    # 6. LOGICAL SEQUENCE (Start vs End)
    def check_logical_sequence(self):
        # Example: if both 'start' and 'end' dates exist
        cols = self.df.columns
        if any("start" in c.lower() for c in cols) and any("end" in c.lower() for c in cols):
            s_col = [c for c in cols if "start" in c.lower()][0]
            e_col = [c for c in cols if "end" in c.lower()][0]
            invalid = self.df[pd.to_datetime(self.df[s_col], errors='coerce') > pd.to_datetime(self.df[e_col], errors='coerce')]
            if not invalid.empty:
                self.issues.append(DataIssue("SEQ_ERROR", f"{s_col}/{e_col}", "Logical Sequence", "High", "Start date is after End date.", []).to_dict())

    # 7. STRUCTURAL NOISE (Whitespaces)
    def check_structural_noise(self):
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].astype(str).str.contains(r'^\s|\s$').any():
                self.issues.append(DataIssue("WHITESPACE", col, "Structural Noise", "Low", "Leading/trailing spaces found.", []).to_dict())

    # 8. ENCODING ARTIFACTS (Junk Symbols)
    def check_encoding_artifacts(self):
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].astype(str).str.contains(r'[^\x00-\x7F]+').any():
                self.issues.append(DataIssue("ENCODING_JUNK", col, "Encoding Artifact", "Medium", "Non-ASCII / Corrupted characters detected.", []).to_dict())

    # 9. TYPE MISMATCH (Specifically catching "twenty", "15.0", etc.)
    def check_type_mismatch(self):
        for col in self.df.columns:
            # Logic: If it's an age or salary column, it should NOT have letters
            col_l = col.lower()
            if any(key in col_l for key in ["age", "salary", "price", "count", "pct"]):
                # Find rows that contain letters (like 'twenty')
                # We use regex [a-zA-Z] to find any alphabet characters
                text_in_numeric = self.df[self.df[col].astype(str).str.contains(r'[a-zA-Z]', na=False)][col]
                
                if not text_in_numeric.empty:
                    self.issues.append(DataIssue(
                        "WORD_AS_NUMBER", 
                        col, 
                        "Type Mismatch", 
                        "High", 
                        f"Found text values ('{text_in_numeric.iloc[0]}') in a numeric column.", 
                        text_in_numeric.tolist()
                    ).to_dict())
            
            # Keep the existing mixed Python type check as well
            types = self.df[col].dropna().apply(type).unique()
            if len(types) > 1:
                self.issues.append(DataIssue("MIXED_TYPE", col, "Type Mismatch", "High", 
                    f"Mixed types: {[t.__name__ for t in types]}", []).to_dict())

    # 10. FORMAT DIVERGENCE (Casing/Patterns)
    def check_format_divergence(self):
        for col in self.df.select_dtypes(include=['object']).columns:
            vals = self.df[col].dropna().astype(str)
            if vals.str.isupper().any() and vals.str.islower().any():
                self.issues.append(DataIssue("CASE_DIVERGE", col, "Format Divergence", "Low", "Inconsistent UPPER/lower casing.", []).to_dict())

    # 11. IDENTITY CLASH (ID Duplicates)
    def check_identity_clash(self):
        id_cols = [c for c in self.df.columns if "id" in c.lower()]
        for col in id_cols:
            if self.df[col].duplicated().any():
                self.issues.append(DataIssue("ID_CLASH", col, "Identity Clash", "High", "Duplicate Primary Keys detected.", []).to_dict())

    # 12. EXTREME OUTLIERS (Statistical)
    def check_extreme_outliers(self):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            z_scores = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            outliers = self.df[np.abs(z_scores) > 3][col]
            if not outliers.empty:
                self.issues.append(DataIssue("Z_OUTLIER", col, "Extreme Outlier", "Low", "Statistical anomaly (Z-Score > 3).", outliers.tolist()).to_dict())

    def run_all_checks(self):
        self.check_missing_data()
        self.check_proxy_missingness()
        self.check_numeric_validity()
        self.check_range_violations()
        self.check_time_travel()
        self.check_logical_sequence()
        self.check_structural_noise()
        self.check_encoding_artifacts()
        self.check_type_mismatch()
        self.check_format_divergence()
        self.check_identity_clash()
        self.check_extreme_outliers()
        return self.issues