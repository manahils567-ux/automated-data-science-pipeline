import pandas as pd
import numpy as np
from ..FixObject import DataFix


class NumericValidityStrategy:
    """
    Generates fix recommendations for numeric validity issues
    (negative ages, invalid ranges, etc.)
    """

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def generate_fixes(self, issue):
        fixes = []
        column = issue["column"]
        issue_id = issue["issue_id"]

        # Convert column to numeric for analysis
        numeric_col = pd.to_numeric(self.df[column], errors='coerce')

        percentage_keywords = ['discount', 'tax', 'rate', 'percentage', 'markup']

        if any(key in column.lower() for key in percentage_keywords):
            # Count values that violate the 0-100 business rule
            invalid_count = len(numeric_col[(numeric_col > 100) | (numeric_col < 0)])
            
            if invalid_count > 0:
                fixes.append(DataFix(
                    fix_id="FIX_CLIP_PERCENTAGE",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Clip {column} to Range (0-100)",
                    fix_description="Enforce business rule: Force values < 0 to 0 and > 100 to 100.",
                    impact=f"Corrects {invalid_count} business rule violations",
                    risk="Low - preserves data while ensuring logical bounds",
                    is_recommended=True
                ))
        
        # Negative age fix
        if "NEG_AGE" in issue_id or "age" in column.lower():
            invalid_count = len(numeric_col[numeric_col < 0])

            # Only proceed if there are actual negative values
            if invalid_count > 0:
                median_val = numeric_col[numeric_col >= 0].median()

                fixes.append(DataFix(
                    fix_id="FIX_NEGATIVE_TO_ABS",
                    issue_id=issue_id,
                    column=column,
                    fix_label="Convert to Absolute Value",
                    fix_description="Convert negative ages to positive (assume data entry error)",
                    impact=f"Corrects {invalid_count} negative values",
                    risk="May not reflect true values if negatives are intentional placeholders",
                    is_recommended=False,
                    metadata={"invalid_count": invalid_count}
                ))

                fixes.append(DataFix(
                    fix_id="FIX_NEGATIVE_TO_MEDIAN",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Replace with Median ({median_val:.1f})",
                    fix_description="Replace negative ages with median of valid ages",
                    impact=f"Corrects {invalid_count} negative values",
                    risk="Loss of original data pattern",
                    is_recommended=True,
                    metadata={"median_value": median_val, "invalid_count": invalid_count}
                ))

                fixes.append(DataFix(
                    fix_id="FIX_NEGATIVE_TO_NAN",
                    issue_id=issue_id,
                    column=column,
                    fix_label="Replace with NaN (Mark as Missing)",
                    fix_description="Treat negative ages as missing values",
                    impact=f"Marks {invalid_count} values as missing",
                    risk="Increases missingness, requires subsequent imputation",
                    is_recommended=False,
                    metadata={"invalid_count": invalid_count}
                ))

        # Range violations (percentages > 100)
        elif "RANGE_EXCEEDED" in issue_id:
            invalid_count = len(numeric_col[numeric_col > 100])

            if invalid_count > 0:
                fixes.append(DataFix(
                    fix_id="FIX_CAP_AT_100",
                    issue_id=issue_id,
                    column=column,
                    fix_label="Cap Values at 100",
                    fix_description="Set all values exceeding 100 to exactly 100",
                    impact=f"Caps {invalid_count} values at maximum",
                    risk="May distort distribution",
                    is_recommended=True,
                    metadata={"cap_value": 100, "invalid_count": invalid_count}
                ))

                fixes.append(DataFix(
                    fix_id="FIX_RANGE_TO_NAN",
                    issue_id=issue_id,
                    column=column,
                    fix_label="Replace with NaN",
                    fix_description="Treat out-of-range values as invalid/missing",
                    impact=f"Marks {invalid_count} values as missing",
                    risk="Increases missingness",
                    is_recommended=False,
                    metadata={"invalid_count": invalid_count}
                ))

                fixes.append(DataFix(
                    fix_id="FIX_DROP_INVALID_ROWS",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Drop Rows ({invalid_count} rows)",
                    fix_description="Remove all rows with out-of-range values",
                    impact=f"Removes {invalid_count} rows from dataset",
                    risk="Data loss",
                    is_recommended=False,
                    metadata={"rows_to_drop": invalid_count}
                ))
        elif "PERCENT_VIOLATION" in issue_id:
            invalid_count = len(numeric_col[(numeric_col > 100) | (numeric_col < 0)])
            
            # Option 1: Clipping
            fixes.append(DataFix(
                fix_id="FIX_CLIP_PERCENTAGE",
                issue_id=issue_id,
                column=column,
                fix_label="Clip Values to Range (0-100)",
                fix_description="Force negative values to 0 and values over 100 to 100.",
                impact=f"Corrects {invalid_count} business rule violations",
                risk="Low - preserves data intent while fixing logical errors",
                is_recommended=True
            ))

            # Option 2: Scale Normalization (0.1 -> 10)
            decimals = numeric_col[(numeric_col > 0) & (numeric_col < 1)]
            if not decimals.empty:
                fixes.append(DataFix(
                    fix_id="FIX_SCALE_PERCENTAGE",
                    issue_id=issue_id,
                    column=column,
                    fix_label="Normalize Decimals (Scale 0.1 to 10%)",
                    fix_description="Detects decimal values and converts them to percentages.",
                    impact=f"Scales {len(decimals)} decimal entries to match percentage format",
                    risk="Medium - assumes 0.1 was intended as 10%",
                    is_recommended=False
                ))

        return fixes