import pandas as pd
import numpy as np
import re
from datetime import datetime
from .IssueObject import DataIssue


class IssueDetectionEngine:
    def __init__(self, df):
        self.df = df
        self.issues = []

        # Auto-detect column types to avoid false positives
        self.date_columns = self._detect_date_columns()
        self.numeric_columns = self._detect_numeric_columns()
        self.id_columns = self._detect_id_columns()
        self.monetary_columns = self._detect_monetary_columns()
        self.age_columns = self._detect_age_columns()

    # SMART COLUMN TYPE DETECTION
    def _detect_age_columns(self):
        """Detect age columns specifically"""
        age_cols = []
        for col in self.df.columns:
            if 'age' in col.lower():
                age_cols.append(col)
        return age_cols

    def _detect_date_columns(self):
        """Detect actual date columns based on column names and content"""
        date_cols = []
        date_keywords = ['date', 'time', 'timestamp', 'datetime', 'created', 'updated', 'joined', 'dob', 'birth',
                         'year', 'month', 'day']

        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    sample = self.df[col].dropna()
                    if len(sample) == 0:
                        continue

                    parsed = pd.to_datetime(sample, errors='coerce')

                    if parsed.notna().mean() > 0.3:
                        date_cols.append(col)
                        continue
                except:
                    pass

            try:
                if self.df[col].dtype == 'object':
                    sample = self.df[col].dropna().astype(str).head(20)
                    if len(sample) == 0:
                        continue

                    # Check if values contain date separators
                    has_date_pattern = sample.str.contains(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', na=False).mean()

                    if has_date_pattern > 0.3:  # If 30%+ look like dates
                        parsed = pd.to_datetime(self.df[col], errors='coerce', infer_datetime_format=True)
                        if parsed.notna().mean() > 0.3:
                            date_cols.append(col)
            except:
                pass

        return date_cols

    def _detect_numeric_columns(self):
        """Detect columns that should be numeric"""
        numeric_cols = []
        numeric_keywords = ['age', 'salary', 'price', 'cost', 'amount', 'count', 'pct', 'percent', 'rate', 'score',
                            'income', 'revenue', 'fee', 'payment', 'wage', 'height', 'weight', 'distance', 'quantity',
                            'total']

        categorical_keywords = ['name', 'email', 'country', 'city', 'address', 'state', 'region', 'category', 'type',
                                'status', 'gender', 'title', 'description', 'remark', 'comment', 'note']

        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in categorical_keywords):
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_cols.append(col)
            elif any(keyword in col.lower() for keyword in numeric_keywords):
                try:
                    numeric_values = pd.to_numeric(self.df[col], errors='coerce')
                    if numeric_values.notna().sum() / len(self.df) >= 0.3:
                        numeric_cols.append(col)
                except:
                    pass
        return numeric_cols

    def _detect_id_columns(self):
        """Detect ID columns (sequential integers)"""
        id_cols = []
        for col in self.df.columns:
            if 'id' in col.lower():
                try:
                    vals = pd.to_numeric(self.df[col], errors='coerce')
                    if vals.notna().all():
                        sorted_vals = sorted(vals.dropna().unique())
                        if len(sorted_vals) > 1:
                            gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
                            avg_gap = np.mean(gaps)
                            if avg_gap <= 2:
                                id_cols.append(col)
                except:
                    pass
        return id_cols

    def _detect_monetary_columns(self):
        """Detect monetary columns (salary, price, cost, etc.)"""
        monetary_cols = []
        monetary_keywords = ['salary', 'price', 'cost', 'amount', 'payment', 'wage', 'income', 'revenue', 'fee']

        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in monetary_keywords):
                monetary_cols.append(col)
        return monetary_cols

    # 1. MISSING DATA (Strict Nulls)
    def check_missing_data(self):
        for col in self.df.columns:
            count = self.df[col].isna().sum()
            if count > 0:
                self.issues.append(
                    DataIssue("MISSING_VAL", col, "Missing Data", "High", f"{count} null values.", []).to_dict())

    # 2. PROXY MISSINGNESS (Placeholder tokens)
    def check_proxy_missingness(self):
        tokens = ["?", "unknown", "n/a", "none", "null", "."]
        for col in self.df.select_dtypes(include=['object']).columns:
            if col in self.date_columns:
                continue
            matches = self.df[col].astype(str).str.lower().str.strip().isin(tokens).sum()
            if matches > 0:
                examples = self.df[col][self.df[col].astype(str).str.lower().str.strip().isin(tokens)].head(3).tolist()
                self.issues.append(DataIssue("PROXY_MISSING", col, "Proxy Missingness", "Medium",
                                             f"Found {matches} placeholder tokens.", examples).to_dict())

    # 3. CHECK FOR EMPTY STRINGS VARIANTS
    def check_empty_string_variants(self):
        """Detects empty strings and text representations of NaN/None."""
        for col in self.df.select_dtypes(include=['object']).columns:
            if col in self.date_columns:
                continue
            empty_variants = self.df[col].astype(str).str.strip().isin(['', 'nan', 'NaN', 'None', 'NONE'])

            if empty_variants.any():
                count = empty_variants.sum()
                self.issues.append(DataIssue(
                    "EMPTY_TEXT",
                    col,
                    "Proxy Missingness",
                    "Medium",
                    f"Found {count} empty or NaN-text values.",
                    []
                ).to_dict())

    # 4. NUMERIC VALIDITY (Mathematical Logic)
    def check_numeric_validity(self):
        for col in self.numeric_columns:
            if 'age' in col.lower():
                inv = pd.to_numeric(self.df[col], errors='coerce')
                neg = inv[inv < 0]
                if not neg.empty:
                    self.issues.append(DataIssue("NEG_AGE", col, "Numeric Validity", "High",
                                                 f"Found {len(neg)} negative age values.",
                                                 neg.tolist()).to_dict())

    # 5. RANGE VIOLATIONS (Mathematical Boundaries)
    def check_range_violations(self):
        for col in self.df.columns:
            if any(x in col.lower() for x in ["pct", "percent", "probability"]):
                val = pd.to_numeric(self.df[col], errors='coerce')
                out = val[val > 100]
                if not out.empty:
                    self.issues.append(
                        DataIssue("RANGE_EXCEEDED", col, "Range Violation", "High", "Value exceeds 100%.",
                                  out.tolist()).to_dict())

    # 6. DOMAIN-SPECIFIC CONSTRAINTS
    def check_domain_constraints(self):
        """
        Check domain-specific impossible values:
        - Age: 0-120 (biologically realistic)
        - Monetary values: > 0 (salary/price cannot be zero)
        """
        # Age constraints (0-120) - STRICT
        for col in self.age_columns:
            numeric_col = pd.to_numeric(self.df[col], errors='coerce')

            # Check for impossible ages (outside 0-120 range)
            impossible = numeric_col[(numeric_col > 120) | (numeric_col < 0)]
            if not impossible.empty:
                self.issues.append(DataIssue(
                    "IMPOSSIBLE_AGE",
                    col,
                    "Domain Constraint Violation",
                    "High",
                    f"Found {len(impossible)} biologically impossible age values (must be 0-120).",
                    impossible.tolist()
                ).to_dict())

        # Monetary constraints (must be > 0)
        for col in self.monetary_columns:
            numeric_col = pd.to_numeric(self.df[col], errors='coerce')
            zeros_or_neg = numeric_col[numeric_col <= 0]

            if not zeros_or_neg.empty:
                self.issues.append(DataIssue(
                    "INVALID_MONETARY",
                    col,
                    "Domain Constraint Violation",
                    "High",
                    f"Found {len(zeros_or_neg)} zero or negative monetary values.",
                    zeros_or_neg.tolist()
                ).to_dict())

    # 7. TIME-TRAVEL ERRORS (Future Dates)
    def check_time_travel(self):
        for col in self.date_columns:
            dates = pd.to_datetime(self.df[col], errors='coerce')
            future = dates[dates > datetime.now()]
            if not future.empty:
                self.issues.append(
                    DataIssue("FUTURE_DATE", col, "Time-Travel Error", "Medium", "Dates set in the future.",
                              future.head(1).astype(str).tolist()).to_dict())

    # 8. CHECK FOR INVALID DATE FORMAT - ONLY for actual date columns
    def check_invalid_date_format(self):
        """Detects TRULY unparseable date strings (not just different formats)."""
        for col in self.date_columns:
            parsed_dates = pd.to_datetime(self.df[col], errors='coerce')

            still_invalid = parsed_dates.isna() & self.df[col].notna()

            if still_invalid.any():
                for date_format in ['%m/%d/%Y', '%d/%m/%Y', '%Y.%m.%d', '%d.%m.%Y', '%Y-%m-%d', '%d-%m-%Y']:
                    try:
                        temp_parsed = pd.to_datetime(
                            self.df.loc[still_invalid, col],
                            format=date_format,
                            errors='coerce'
                        )
                        parsed_dates.loc[still_invalid] = parsed_dates.loc[still_invalid].fillna(temp_parsed)
                        still_invalid = parsed_dates.isna() & self.df[col].notna()
                    except:
                        continue

            truly_invalid_mask = parsed_dates.isna() & self.df[col].notna()
            invalid_count = truly_invalid_mask.sum()

            if invalid_count > 0:
                invalid_examples = self.df[col][truly_invalid_mask].head(3).tolist()
                self.issues.append(DataIssue(
                    "INVALID_DATE_FORMAT",
                    col,
                    "Invalid Date Format",
                    "High",
                    f"Found {invalid_count} truly unparseable date values (not just different formats).",
                    invalid_examples
                ).to_dict())

    # 9. DATE FORMAT INCONSISTENCY
    def check_date_format_inconsistency(self):
        """
        Detects mixed date formats in the same column.
        e.g., some dates as "2023-01-01" and others as "01/05/2023"
        """
        for col in self.date_columns:
            valid_dates = self.df[col].dropna().astype(str)

            if len(valid_dates) < 2:
                continue

            has_slash = valid_dates.str.contains('/', na=False).sum()
            has_dash = valid_dates.str.contains('-', na=False).sum()
            has_dot = valid_dates.str.contains(r'\.', na=False, regex=True).sum()

            separators_used = []
            if has_slash > 0:
                separators_used.append('/')
            if has_dash > 0:
                separators_used.append('-')
            if has_dot > 0:
                separators_used.append('.')

            if len(separators_used) > 1:
                examples = valid_dates.head(6).tolist()
                self.issues.append(DataIssue(
                    "DATE_FORMAT_MIXED",
                    col,
                    "Format Divergence",
                    "High",
                    f"Mixed date formats detected ({', '.join(separators_used)}) in same column.",
                    examples
                ).to_dict())

    # 10. LOGICAL SEQUENCE (Start vs End)
    def check_logical_sequence(self):
        cols = self.df.columns
        if any("start" in c.lower() for c in cols) and any("end" in c.lower() for c in cols):
            s_col = [c for c in cols if "start" in c.lower()][0]
            e_col = [c for c in cols if "end" in c.lower()][0]
            invalid = self.df[
                pd.to_datetime(self.df[s_col], errors='coerce') > pd.to_datetime(self.df[e_col], errors='coerce')]
            if not invalid.empty:
                self.issues.append(DataIssue("SEQ_ERROR", f"{s_col}/{e_col}", "Logical Sequence", "High",
                                             "Start date is after End date.", []).to_dict())

    # 11. STRUCTURAL NOISE (Whitespaces)
    def check_structural_noise(self):
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].astype(str).str.contains(r'^\s|\s$').any():
                count = self.df[col].astype(str).str.contains(r'^\s|\s$').sum()
                self.issues.append(
                    DataIssue("WHITESPACE", col, "Structural Noise", "Low",
                              f"Found {count} values with leading/trailing spaces.",
                              []).to_dict())

    # 12. CHECK FOR STRUCTURAL NOISE (Special Characters)
    def check_special_characters(self):
        """Detects special characters (?, !, @, etc.) within text values."""
        skip_special_char_check = ['email', 'url', 'website', 'link', 'phone', 'contact']

        for col in self.df.select_dtypes(include=['object']).columns:
            if col in self.date_columns:
                continue

            if any(keyword in col.lower() for keyword in skip_special_char_check):
                continue

            has_special = self.df[col].astype(str).str.contains(r'[?!#$%^&*]', na=False)

            if has_special.any():
                count = has_special.sum()
                examples = self.df[col][has_special].head(3).tolist()
                self.issues.append(DataIssue(
                    "SPECIAL_CHARS",
                    col,
                    "Structural Noise",
                    "Medium",
                    f"Found {count} values with special characters (?, !, #, etc.).",
                    examples
                ).to_dict())

    # 13. ENCODING ARTIFACTS (Junk Symbols)
    def check_encoding_artifacts(self):
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].astype(str).str.contains(r'[^\x00-\x7F]+').any():
                count = self.df[col].astype(str).str.contains(r'[^\x00-\x7F]+').sum()
                self.issues.append(DataIssue("ENCODING_JUNK", col, "Encoding Artifact", "Medium",
                                             f"Found {count} values with non-ASCII/corrupted characters.",
                                             []).to_dict())

    # 14. TYPE MISMATCH
    def check_type_mismatch(self):
        for col in self.numeric_columns:
            text_in_numeric = self.df[self.df[col].astype(str).str.contains(r'[a-zA-Z]', na=False)][col]

            if not text_in_numeric.empty:
                self.issues.append(DataIssue(
                    "WORD_AS_NUMBER",
                    col,
                    "Type Mismatch",
                    "High",
                    f"Found {len(text_in_numeric)} text values in numeric column.",
                    text_in_numeric.head(3).tolist()
                ).to_dict())

    # 15. FORMAT DIVERGENCE (Casing/Patterns)
    def check_format_divergence(self):
        for col in self.df.select_dtypes(include=['object']).columns:
            if col in self.date_columns or col in self.id_columns:
                continue
            vals = self.df[col].dropna().astype(str)
            if vals.str.isupper().any() and vals.str.islower().any():
                self.issues.append(
                    DataIssue("CASE_DIVERGE", col, "Format Divergence", "Low", "Inconsistent UPPER/lower casing.",
                              []).to_dict())

    # 16. IDENTITY CLASH (ID Duplicates)
    def check_identity_clash(self):
        for col in self.id_columns:
            if self.df[col].duplicated().any():
                dup_count = self.df[col].duplicated().sum()
                self.issues.append(
                    DataIssue("ID_CLASH", col, "Identity Clash", "High",
                              f"Found {dup_count} duplicate Primary Keys.",
                              []).to_dict())

    # 17. EXTREME OUTLIERS (Statistical)
    def check_extreme_outliers(self):
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col in self.id_columns or col in self.age_columns:
                continue

            col_data = self.df[col].dropna()
            if len(col_data) < 3:
                continue

            mean_val = col_data.mean()
            std_val = col_data.std()

            if std_val == 0:
                continue

            z_scores = np.abs((col_data - mean_val) / std_val)
            outliers = col_data[z_scores > 3]

            if not outliers.empty:
                self.issues.append(
                    DataIssue("Z_OUTLIER", col, "Extreme Outlier", "Medium",
                              f"Found {len(outliers)} statistical outliers (Z-Score > 3).",
                              outliers.head(3).tolist()).to_dict())

    def run_all_checks(self):
        self.check_missing_data()
        self.check_proxy_missingness()
        self.check_empty_string_variants()
        self.check_numeric_validity()
        self.check_range_violations()
        self.check_domain_constraints()
        self.check_time_travel()
        self.check_invalid_date_format()
        self.check_date_format_inconsistency()
        self.check_logical_sequence()
        self.check_structural_noise()
        self.check_special_characters()
        self.check_encoding_artifacts()
        self.check_type_mismatch()
        self.check_format_divergence()
        self.check_identity_clash()
        self.check_extreme_outliers()
        return self.issues