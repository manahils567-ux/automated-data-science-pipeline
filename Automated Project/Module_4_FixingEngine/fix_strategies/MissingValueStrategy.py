import pandas as pd
import numpy as np
from ..FixObject import DataFix


class MissingValueStrategy:
    """
    Generates fix recommendations for missing data issues.
    ENHANCED: Smarter handling for high-missingness columns with pattern extraction.
    """

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def _extract_numeric_from_text(self, series):
        """
        Extract numeric values from text like '1[4]' -> 1
        """
        return series.astype(str).str.extract(r'(\d+)', expand=False).astype(float)

    def generate_fixes(self, issue):
        """
        Generate multiple fix options for missing value issues.
        Recommends based on column type and distribution.
        """
        fixes = []
        column = issue["column"]
        issue_id = issue["issue_id"]
        col_metadata = self.metadata["columns"].get(column, {})

        missing_count = self.df[column].isna().sum()
        missing_pct = (missing_count / len(self.df)) * 100

        dtype = col_metadata.get("dtype", "unknown")

        if missing_pct > 40:
            non_null_values = self.df[column].dropna()

            if len(non_null_values) > 0 and dtype == "object":
                sample_values = non_null_values.astype(str).head(10).tolist()

                has_numeric_pattern = non_null_values.astype(str).str.contains(r'\d+', na=False).any()

                if has_numeric_pattern:
                    extracted = self._extract_numeric_from_text(non_null_values)
                    if extracted.notna().sum() > 0:
                        median_extracted = int(extracted.median())

                        fixes.append(DataFix(
                            fix_id="FIX_EXTRACT_NUMERIC_IMPUTE",
                            issue_id=issue_id,
                            column=column,
                            fix_label=f"Extract Numbers & Impute Missing with Median ({median_extracted})",
                            fix_description=f"Extract numeric values from existing data (e.g., '1[4]' → 1) and fill missing with median",
                            impact=f"Preserves column, extracts {len(non_null_values)} values, imputes {missing_count} missing",
                            risk="May lose non-numeric context",
                            is_recommended=True,
                            metadata={
                                "median_value": median_extracted,
                                "extract_pattern": r'(\d+)',
                                "sample_values": sample_values
                            }
                        ))

                        fixes.append(DataFix(
                            fix_id="FIX_DROP_COLUMN",
                            issue_id=issue_id,
                            column=column,
                            fix_label=f"Drop Column (>{missing_pct:.1f}% missing)",
                            fix_description="Remove this column entirely due to high missingness",
                            impact=f"Column will be removed from dataset",
                            risk="Loss of potentially valuable information",
                            is_recommended=False,
                            metadata={"missing_pct": missing_pct}
                        ))

                        return fixes

            fixes.append(DataFix(
                fix_id="FIX_DROP_COLUMN",
                issue_id=issue_id,
                column=column,
                fix_label=f"Drop Column (>{missing_pct:.1f}% missing)",
                fix_description="Remove this column entirely due to high missingness",
                impact=f"Column will be removed from dataset",
                risk="Loss of potentially valuable information",
                is_recommended=True,
                metadata={"missing_pct": missing_pct}
            ))

        if "int" in dtype or "float" in dtype:
            mean_val = col_metadata.get("mean")
            std_val = col_metadata.get("std")

            skewness = self.df[column].skew() if pd.api.types.is_numeric_dtype(self.df[column]) else 0

            is_age_column = 'age' in column.lower()
            is_integer_context = is_age_column or any(kw in column.lower() for kw in ['count', 'quantity', 'number'])

            if abs(skewness) > 1 or is_age_column:
                median_val = self.df[column].median()

                if is_integer_context:
                    median_val = int(round(median_val))

                fixes.append(DataFix(
                    fix_id="FIX_MEDIAN_IMPUTE",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Replace with Median ({median_val})",
                    fix_description="Impute missing values using median (robust to outliers)",
                    impact=f"Preserves all {missing_count} missing rows",
                    risk="May not capture true distribution",
                    is_recommended=True,
                    metadata={"median_value": median_val, "skewness": skewness, "round_to_int": is_integer_context}
                ))
            else:
                mean_display = int(round(mean_val)) if is_integer_context else mean_val

                fixes.append(DataFix(
                    fix_id="FIX_MEAN_IMPUTE",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Replace with Mean ({mean_display})",
                    fix_description="Impute missing values using mean",
                    impact=f"Preserves all {missing_count} missing rows",
                    risk="Sensitive to outliers",
                    is_recommended=True,
                    metadata={"mean_value": mean_val, "std": std_val, "round_to_int": is_integer_context}
                ))

                median_val = self.df[column].median()
                if is_integer_context:
                    median_val = int(round(median_val))

                fixes.append(DataFix(
                    fix_id="FIX_MEDIAN_IMPUTE",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Replace with Median ({median_val})",
                    fix_description="Impute missing values using median",
                    impact=f"Preserves all {missing_count} missing rows",
                    risk="May not capture true distribution",
                    is_recommended=False,
                    metadata={"median_value": median_val, "round_to_int": is_integer_context}
                ))

        elif "category" in dtype or "object" in dtype:
            clean_values = self.df[column].dropna()
            clean_values = clean_values[~clean_values.astype(str).str.contains(r'[?!@#$%^&*]', na=False)]
            clean_values = clean_values[
                ~clean_values.astype(str).str.lower().str.strip().isin(['unknown', 'none', 'null', 'n/a'])]

            unique_count = clean_values.nunique()
            total_count = len(clean_values)

            diversity_ratio = unique_count / total_count if total_count > 0 else 0

            mode_val = clean_values.mode() if not clean_values.empty else None

            if mode_val is not None and not mode_val.empty:
                mode_val = mode_val[0]
                mode_count = (clean_values == mode_val).sum()
                mode_frequency = mode_count / total_count if total_count > 0 else 0

                recommend_mode = (mode_frequency > 0.2) and (diversity_ratio < 0.7)

                fixes.append(DataFix(
                    fix_id="FIX_MODE_IMPUTE",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Replace with Mode ('{mode_val}', appears {mode_frequency * 100:.1f}%)",
                    fix_description="Impute missing values using most frequent CLEAN category",
                    impact=f"Preserves all {missing_count} missing rows",
                    risk="May increase class imbalance",
                    is_recommended=recommend_mode,
                    metadata={"mode_value": mode_val, "mode_frequency": mode_frequency}
                ))

            if diversity_ratio > 0.5 or mode_val is None or mode_val.empty:
                fixes.append(DataFix(
                    fix_id="FIX_DROP_ROWS",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Drop Rows with Missing Values ({missing_count} rows)",
                    fix_description="Remove rows with missing values",
                    impact=f"Removes {missing_count} rows ({missing_pct:.1f}% of dataset)",
                    risk="Data loss, but preserves diversity",
                    is_recommended=True if (mode_val is None or not recommend_mode) else False,
                    metadata={"rows_to_drop": missing_count, "diversity_ratio": diversity_ratio}
                ))

        elif "datetime" in dtype:
            fixes.append(DataFix(
                fix_id="FIX_FORWARD_FILL",
                issue_id=issue_id,
                column=column,
                fix_label="Forward Fill (Use Previous Date)",
                fix_description="Fill missing dates with the previous valid date",
                impact=f"Preserves all {missing_count} missing rows",
                risk="Assumes temporal continuity",
                is_recommended=True,
                metadata={}
            ))

        if not any(f.fix_id == "FIX_DROP_ROWS" for f in fixes):
            total_rows = len(self.df)
            rows_remaining_pct = ((total_rows - missing_count) / total_rows) * 100 if total_rows > 0 else 0

            if rows_remaining_pct >= 60:
                fixes.append(DataFix(
                    fix_id="FIX_DROP_ROWS",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Drop Rows with Missing Values ({missing_count} rows)",
                    fix_description="Remove all rows containing missing values in this column",
                    impact=f"Removes {missing_count} rows ({missing_pct:.1f}% of dataset)",
                    risk="Data loss, reduced sample size",
                    is_recommended=False,
                    metadata={"rows_to_drop": missing_count}
                ))

        return fixes