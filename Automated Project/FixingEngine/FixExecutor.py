import pandas as pd
import numpy as np


class FixExecutor:
    """
    Applies recommended fixes to the DataFrame.
    Executes cleaning operations and tracks changes.
    """

    def __init__(self, df):
        self.df = df.copy()  # Work on a copy
        self.execution_log = []
        self.id_columns = self._detect_id_columns()

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

    def _reset_id_columns(self):
        """Reset ID columns to sequential 1,2,3,... after row deletions"""
        for col in self.id_columns:
            self.df[col] = range(1, len(self.df) + 1)
            self.execution_log.append({
                "column": col,
                "fix_applied": "Reset ID to sequential (1 to n)",
                "values_changed": "All IDs"
            })

    # --------------------------
    # Missing Value Fixes
    # --------------------------
    def _apply_median_impute(self, fix):
        column = fix.column
        median_val = fix.metadata.get("median_value")
        round_to_int = fix.metadata.get("round_to_int", False)
        before_count = self.df[column].isna().sum()

        # Round for integer contexts (age, count, etc.)
        if round_to_int:
            median_val = int(round(median_val))

        self.df[column].fillna(median_val, inplace=True)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_mean_impute(self, fix):
        column = fix.column
        mean_val = fix.metadata.get("mean_value")
        round_to_int = fix.metadata.get("round_to_int", False)
        before_count = self.df[column].isna().sum()

        # Round for integer contexts (age, count, etc.)
        if round_to_int:
            mean_val = int(round(mean_val))

        self.df[column].fillna(mean_val, inplace=True)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_mode_impute(self, fix):
        column = fix.column

        if pd.api.types.is_numeric_dtype(self.df[column]):
            return

        mode_val = fix.metadata.get("mode_value")
        before_count = self.df[column].isna().sum()

        self.df[column].fillna(mode_val, inplace=True)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_drop_column(self, fix):
        column = fix.column
        self.df.drop(columns=[column], inplace=True)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": "Column dropped"
        })

    def _apply_drop_rows(self, fix):
        column = fix.column
        before_rows = len(self.df)

        if self._should_skip_row_drop(before_rows):
            print(f"⚠️  Skipping row drop for '{column}' - too much data already lost")
            return

        rows_to_drop = self.df[column].isna().sum()
        drop_pct = (rows_to_drop / before_rows) * 100 if before_rows > 0 else 0

        if drop_pct > 30:
            print(f"⚠️  Skipping row drop for '{column}' - would remove {drop_pct:.1f}% of data")
            return

        self.df.dropna(subset=[column], inplace=True)
        after_rows = len(self.df)

        if before_rows != after_rows:
            self._reset_id_columns()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": f"{before_rows - after_rows} rows removed"
        })

    def _apply_forward_fill(self, fix):
        column = fix.column
        before_count = self.df[column].isna().sum()

        self.df[column].fillna(method='ffill', inplace=True)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    # --------------------------
    # Numeric Validity Fixes
    # --------------------------
    def _apply_negative_to_abs(self, fix):
        column = fix.column
        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[numeric_col < 0])

        self.df[column] = numeric_col.abs()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_negative_to_median(self, fix):
        column = fix.column
        median_val = fix.metadata.get("median_value")

        if pd.isna(median_val):
            median_val = 0  # Default fallback
        else:
            median_val = int(round(median_val))

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[numeric_col < 0])

        self.df[column] = numeric_col
        self.df.loc[self.df[column] < 0, column] = median_val

        if self.df[column].notna().all():
            self.df[column] = self.df[column].astype(int)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_negative_to_nan(self, fix):
        column = fix.column

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[numeric_col < 0])

        self.df[column] = numeric_col
        self.df.loc[self.df[column] < 0, column] = np.nan

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_cap_at_100(self, fix):
        column = fix.column

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[numeric_col > 100])

        self.df[column] = numeric_col
        self.df.loc[self.df[column] > 100, column] = 100

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_range_to_nan(self, fix):
        column = fix.column

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[numeric_col > 100])

        self.df[column] = numeric_col
        self.df.loc[self.df[column] > 100, column] = np.nan

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    # --------------------------
    # Type Mismatch Fixes
    # --------------------------
    def _apply_word_to_number(self, fix):
        from .fix_strategies.TypeMismatchStrategy import TypeMismatchStrategy

        column = fix.column
        word_map = TypeMismatchStrategy.WORD_TO_NUM

        def convert_word(val):
            if pd.isna(val):
                return val
            val_str = str(val).lower().strip()
            return word_map.get(val_str, val)

        before_count = self.df[column].astype(str).str.contains(r'[a-zA-Z]', na=False).sum()
        self.df[column] = self.df[column].apply(convert_word)
        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_text_to_nan_impute(self, fix):
        column = fix.column
        median_val = fix.metadata.get("median_value")

        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
        before_count = self.df[column].isna().sum()
        self.df[column].fillna(median_val, inplace=True)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    # --------------------------
    # Outlier Fixes
    # --------------------------
    def _apply_cap_percentile(self, fix):
        column = fix.column
        p1 = fix.metadata.get("p1")
        p99 = fix.metadata.get("p99")

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[(numeric_col < p1) | (numeric_col > p99)])

        self.df[column] = numeric_col
        self.df.loc[self.df[column] < p1, column] = p1
        self.df.loc[self.df[column] > p99, column] = p99

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_cap_iqr(self, fix):
        column = fix.column
        lower = fix.metadata.get("lower_bound")
        upper = fix.metadata.get("upper_bound")

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[(numeric_col < lower) | (numeric_col > upper)])

        self.df[column] = numeric_col
        self.df.loc[self.df[column] < lower, column] = lower
        self.df.loc[self.df[column] > upper, column] = upper

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_remove_outliers(self, fix):
        column = fix.column

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        mean_val = numeric_col.mean()
        std_val = numeric_col.std()

        before_rows = len(self.df)

        if std_val > 0:
            z_scores = np.abs((numeric_col - mean_val) / std_val)
            self.df = self.df[z_scores <= 3]

        after_rows = len(self.df)

        # Reset ID columns after dropping rows
        if before_rows != after_rows:
            self._reset_id_columns()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": f"{before_rows - after_rows} rows removed"
        })

    # --------------------------
    # Duplicate Fixes
    # --------------------------
    def _apply_keep_first_id(self, fix):
        column = fix.column
        before_rows = len(self.df)

        self.df.drop_duplicates(subset=[column], keep='first', inplace=True)
        after_rows = len(self.df)

        # Reset ID columns after dropping rows
        if before_rows != after_rows:
            self._reset_id_columns()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": f"{before_rows - after_rows} rows removed"
        })

    def _apply_drop_exact_duplicates(self, fix):
        before_rows = len(self.df)

        self.df.drop_duplicates(inplace=True)
        after_rows = len(self.df)

        # Reset ID columns after dropping rows
        if before_rows != after_rows:
            self._reset_id_columns()

        self.execution_log.append({
            "column": "All columns",
            "fix_applied": fix.fix_label,
            "values_changed": f"{before_rows - after_rows} rows removed"
        })

    # --------------------------
    # Text Cleaning Fixes
    # --------------------------
    def _apply_strip_whitespace(self, fix):
        column = fix.column
        before_count = self.df[column].astype(str).str.contains(r'^\s|\s$', na=False).sum()

        self.df[column] = self.df[column].astype(str).str.strip()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_remove_non_ascii(self, fix):
        column = fix.column
        before_count = self.df[column].astype(str).str.contains(r'[^\x00-\x7F]+', na=False).sum()

        self.df[column] = self.df[column].astype(str).str.replace(r'[^\x00-\x7F]+', '', regex=True)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_standardize_case_lower(self, fix):
        column = fix.column
        self.df[column] = self.df[column].astype(str).str.lower()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": "All values"
        })

    def _apply_proxy_to_nan(self, fix):
        column = fix.column
        tokens = ["?", "unknown", "n/a", "none", "null", "."]

        before_count = self.df[column].astype(str).str.lower().str.strip().isin(tokens).sum()
        self.df.loc[self.df[column].astype(str).str.lower().str.strip().isin(tokens), column] = np.nan

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_remove_special_chars(self, fix):
        column = fix.column
        before_count = self.df[column].astype(str).str.contains(r'[?!@#$%^&*]', na=False).sum()

        self.df[column] = self.df[column].astype(str).str.replace(r'[?!@#$%^&*]', '', regex=True)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_replace_special_with_space(self, fix):
        column = fix.column
        before_count = self.df[column].astype(str).str.contains(r'[?!@#$%^&*]', na=False).sum()

        self.df[column] = self.df[column].astype(str).str.replace(r'[?!@#$%^&*]', ' ', regex=True)
        self.df[column] = self.df[column].str.replace(r'\s+', ' ', regex=True).str.strip()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_empty_text_to_mode(self, fix):
        column = fix.column

        if pd.api.types.is_numeric_dtype(self.df[column]):
            return

        mode_val = fix.metadata.get("mode_value")

        empty_mask = self.df[column].astype(str).str.strip().isin(
            ['', 'nan', 'NaN', 'None', 'NONE']
        )
        before_count = empty_mask.sum()

        self.df.loc[empty_mask, column] = mode_val

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_empty_text_to_nan(self, fix):
        column = fix.column

        empty_mask = self.df[column].astype(str).str.strip().isin(['', 'nan', 'NaN', 'None', 'NONE'])
        before_count = empty_mask.sum()

        self.df.loc[empty_mask, column] = np.nan

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_invalid_date_to_nan(self, fix):
        column = fix.column

        parsed_dates = pd.to_datetime(self.df[column], errors='coerce')
        invalid_mask = parsed_dates.isna() & self.df[column].notna()
        before_count = invalid_mask.sum()

        self.df.loc[invalid_mask, column] = np.nan

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_drop_invalid_date_rows(self, fix):
        column = fix.column

        parsed_dates = pd.to_datetime(self.df[column], errors='coerce')
        invalid_mask = parsed_dates.isna() & self.df[column].notna()

        before_rows = len(self.df)
        self.df = self.df[~invalid_mask]
        after_rows = len(self.df)

        # Reset ID columns after dropping rows
        if before_rows != after_rows:
            self._reset_id_columns()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": f"{before_rows - after_rows} rows removed"
        })

    def _apply_invalid_date_default(self, fix):
        column = fix.column
        default_date = fix.metadata.get("default_date", "1900-01-01")

        parsed_dates = pd.to_datetime(self.df[column], errors='coerce')
        invalid_mask = parsed_dates.isna() & self.df[column].notna()
        before_count = invalid_mask.sum()

        self.df.loc[invalid_mask, column] = default_date

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_invalid_date_impute_median(self, fix):
        column = fix.column
        median_date_str = fix.metadata.get("median_date")

        parsed_dates = pd.to_datetime(self.df[column], errors='coerce')
        invalid_mask = parsed_dates.isna() & self.df[column].notna()
        before_count = invalid_mask.sum()

        self.df.loc[invalid_mask, column] = median_date_str

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": f"{before_count} dates imputed with median"
        })

    # --------------------------
    # Domain Validation Fixes
    # --------------------------
    def _apply_impossible_age_to_median(self, fix):
        column = fix.column
        median_val = fix.metadata.get("median_value")

        # Handle NaN median
        if pd.isna(median_val):
            median_val = 30
        else:
            median_val = int(round(median_val))

        # Convert to numeric first
        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[(numeric_col > 120) | (numeric_col < 0)])

        self.df[column] = numeric_col
        self.df.loc[(self.df[column] > 120) | (self.df[column] < 0), column] = median_val

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

        if self.df[column].notna().all():
            self.df[column] = self.df[column].astype(int)

    def _apply_impossible_age_to_nan(self, fix):
        column = fix.column

        # Convert to numeric first
        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[(numeric_col > 120) | (numeric_col < 0)])

        self.df[column] = numeric_col
        self.df.loc[(self.df[column] > 120) | (self.df[column] < 0), column] = np.nan

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_drop_impossible_age_rows(self, fix):
        column = fix.column

        # Convert to numeric first
        numeric_col = pd.to_numeric(self.df[column], errors='coerce')

        before_rows = len(self.df)
        self.df = self.df[~((numeric_col > 120) | (numeric_col < 0))]
        after_rows = len(self.df)

        if before_rows != after_rows:
            self._reset_id_columns()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": f"{before_rows - after_rows} rows removed"
        })

    def _apply_zero_monetary_to_median(self, fix):
        column = fix.column

        numeric_col = (
            self.df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
        )

        numeric_col = pd.to_numeric(numeric_col, errors="coerce")

        valid_median = numeric_col[numeric_col > 0].median()

        before_count = len(numeric_col[numeric_col <= 0])

        self.df[column] = numeric_col
        self.df.loc[self.df[column] <= 0, column] = valid_median

        self.df[column] = self.df[column].astype(float)

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_zero_monetary_to_nan(self, fix):
        column = fix.column

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')
        before_count = len(numeric_col[numeric_col <= 0])

        self.df[column] = numeric_col
        self.df.loc[self.df[column] <= 0, column] = np.nan

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": before_count
        })

    def _apply_drop_zero_monetary_rows(self, fix):
        column = fix.column

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')

        before_rows = len(self.df)
        self.df = self.df[~(numeric_col <= 0)]
        after_rows = len(self.df)

        if before_rows != after_rows:
            self._reset_id_columns()

        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": f"{before_rows - after_rows} rows removed"
        })

    def _apply_standardize_date_format(self, fix):
        column = fix.column

        before_values = self.df[column].copy()
        result_dates = pd.Series([pd.NaT] * len(self.df), index=self.df.index)

        date_formats = [
            None,
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y.%m.%d',
            '%d.%m.%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%m-%d-%Y',
        ]

        unparsed_mask = pd.Series([True] * len(self.df), index=self.df.index)
        unparsed_mask[self.df[column].isna()] = False  # Don't try to parse actual NaNs

        for fmt in date_formats:
            if unparsed_mask.sum() == 0:
                break

            try:
                if fmt is None:
                    temp = pd.to_datetime(self.df.loc[unparsed_mask, column], errors='coerce')
                else:
                    temp = pd.to_datetime(self.df.loc[unparsed_mask, column], format=fmt, errors='coerce')

                successfully_parsed = temp.notna()
                result_dates.loc[unparsed_mask & successfully_parsed] = temp[successfully_parsed]

                unparsed_mask = result_dates.isna() & self.df[column].notna()

            except Exception as e:
                continue

        successfully_parsed_mask = result_dates.notna()
        parsed_count = successfully_parsed_mask.sum()

        if parsed_count > 0:
            self.df.loc[successfully_parsed_mask, column] = result_dates[successfully_parsed_mask].dt.strftime(
                '%Y-%m-%d')


        self.execution_log.append({
            "column": column,
            "fix_applied": fix.fix_label,
            "values_changed": f"{parsed_count} dates standardized to YYYY-MM-DD"
        })

    def _should_skip_row_drop(self, current_rows):
        """
        Determine if we should skip row dropping to prevent data catastrophe.
        Returns True if we've already lost too much data.
        """
        original_rows = len(self.df) if not hasattr(self, '_original_row_count') else self._original_row_count

        data_loss_pct = ((original_rows - current_rows) / original_rows) * 100

        return data_loss_pct > 50

    # --------------------------
    # Main Application Method
    # --------------------------
    def apply_recommended_fixes(self, all_fixes):
        """
        Apply all recommended fixes to the DataFrame in SMART ORDER.
        """
        self._original_row_count = len(self.df)

        recommended_fixes = [fix for fix in all_fixes if fix.is_recommended]

        def get_fix_priority(fix):
            priority_map = {
                "FIX_STRIP_WHITESPACE": 1,
                "FIX_REMOVE_SPECIAL_CHARS": 1,
                "FIX_REPLACE_SPECIAL_WITH_SPACE": 1,
                "FIX_REMOVE_NON_ASCII": 1,
                "FIX_STANDARDIZE_DATE_FORMAT": 2,
                "FIX_WORD_TO_NUMBER": 3,
                "FIX_PROXY_TO_NAN": 4,
                "FIX_EMPTY_TEXT_TO_NAN": 4,
                "FIX_INVALID_DATE_TO_NAN": 5,
                "FIX_INVALID_DATE_IMPUTE_MEDIAN": 5,
                "FIX_NEGATIVE_TO_MEDIAN": 6,
                "FIX_NEGATIVE_TO_ABS": 6,
                "FIX_CAP_AT_100": 6,
                "FIX_IMPOSSIBLE_AGE_TO_MEDIAN": 6,
                "FIX_ZERO_MONETARY_TO_MEDIAN": 6,
                "FIX_MEDIAN_IMPUTE": 7,
                "FIX_MEAN_IMPUTE": 7,
                "FIX_MODE_IMPUTE": 7,
                "FIX_EMPTY_TEXT_TO_MODE": 7,
                "FIX_TEXT_TO_NAN_IMPUTE": 7,
                "FIX_KEEP_FIRST_ID": 8,
                "FIX_DROP_EXACT_DUPLICATES": 8,
            }
            return priority_map.get(fix.fix_id, 99)

        # Sort by priority
        recommended_fixes.sort(key=get_fix_priority)

        fix_method_map = {
            "FIX_MEDIAN_IMPUTE": self._apply_median_impute,
            "FIX_MEAN_IMPUTE": self._apply_mean_impute,
            "FIX_MODE_IMPUTE": self._apply_mode_impute,
            "FIX_DROP_COLUMN": self._apply_drop_column,
            "FIX_DROP_ROWS": self._apply_drop_rows,
            "FIX_FORWARD_FILL": self._apply_forward_fill,
            "FIX_NEGATIVE_TO_ABS": self._apply_negative_to_abs,
            "FIX_NEGATIVE_TO_MEDIAN": self._apply_negative_to_median,
            "FIX_NEGATIVE_TO_NAN": self._apply_negative_to_nan,
            "FIX_CAP_AT_100": self._apply_cap_at_100,
            "FIX_RANGE_TO_NAN": self._apply_range_to_nan,
            "FIX_WORD_TO_NUMBER": self._apply_word_to_number,
            "FIX_TEXT_TO_NAN_IMPUTE": self._apply_text_to_nan_impute,
            "FIX_CAP_PERCENTILE": self._apply_cap_percentile,
            "FIX_CAP_IQR": self._apply_cap_iqr,
            "FIX_REMOVE_OUTLIERS": self._apply_remove_outliers,
            "FIX_KEEP_FIRST_ID": self._apply_keep_first_id,
            "FIX_DROP_EXACT_DUPLICATES": self._apply_drop_exact_duplicates,
            "FIX_STRIP_WHITESPACE": self._apply_strip_whitespace,
            "FIX_REMOVE_NON_ASCII": self._apply_remove_non_ascii,
            "FIX_STANDARDIZE_CASE_LOWER": self._apply_standardize_case_lower,
            "FIX_PROXY_TO_NAN": self._apply_proxy_to_nan,
            "FIX_REMOVE_SPECIAL_CHARS": self._apply_remove_special_chars,
            "FIX_REPLACE_SPECIAL_WITH_SPACE": self._apply_replace_special_with_space,
            "FIX_EMPTY_TEXT_TO_MODE": self._apply_empty_text_to_mode,
            "FIX_EMPTY_TEXT_TO_NAN": self._apply_empty_text_to_nan,
            "FIX_INVALID_DATE_TO_NAN": self._apply_invalid_date_to_nan,
            "FIX_DROP_INVALID_DATE_ROWS": self._apply_drop_invalid_date_rows,
            "FIX_INVALID_DATE_DEFAULT": self._apply_invalid_date_default,
            "FIX_INVALID_DATE_IMPUTE_MEDIAN": self._apply_invalid_date_impute_median,
            "FIX_IMPOSSIBLE_AGE_TO_MEDIAN": self._apply_impossible_age_to_median,
            "FIX_IMPOSSIBLE_AGE_TO_NAN": self._apply_impossible_age_to_nan,
            "FIX_DROP_IMPOSSIBLE_AGE_ROWS": self._apply_drop_impossible_age_rows,
            "FIX_ZERO_MONETARY_TO_MEDIAN": self._apply_zero_monetary_to_median,
            "FIX_ZERO_MONETARY_TO_NAN": self._apply_zero_monetary_to_nan,
            "FIX_DROP_ZERO_MONETARY_ROWS": self._apply_drop_zero_monetary_rows,
            "FIX_STANDARDIZE_DATE_FORMAT": self._apply_standardize_date_format
        }

        for fix in recommended_fixes:
            fix_method = fix_method_map.get(fix.fix_id)
            if fix_method:
                try:
                    fix_method(fix)
                except Exception as e:
                    print(f"⚠ Error applying {fix.fix_label}: {e}")
            else:
                print(f"⚠ No implementation found for fix: {fix.fix_id}")

        return self.df, self.execution_log