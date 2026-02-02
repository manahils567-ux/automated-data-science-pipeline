import pandas as pd
import numpy as np
from ..FixObject import DataFix


class DomainValidationStrategy:
    """
    Handles domain-specific constraint violations.
    ENHANCED: Strict age validation (0-120, integers only)
    """

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def generate_fixes(self, issue):
        """
        Generate fixes for domain constraint violations.
        """
        fixes = []
        column = issue["column"]
        issue_id = issue["issue_id"]

        numeric_col = pd.to_numeric(self.df[column], errors='coerce')

        # IMPOSSIBLE AGE (>120 or <0)
        if "IMPOSSIBLE_AGE" in issue_id:
            invalid_count = len(numeric_col[(numeric_col > 120) | (numeric_col < 0)])

            # Calculate median from VALID ages only (0-120)
            valid_ages = numeric_col[(numeric_col >= 0) & (numeric_col <= 120)]
            median_age = int(round(valid_ages.median())) if not valid_ages.empty else 30

            fixes.append(DataFix(
                fix_id="FIX_IMPOSSIBLE_AGE_TO_MEDIAN",
                issue_id=issue_id,
                column=column,
                fix_label=f"Replace with Median of Valid Ages ({median_age})",
                fix_description="Replace impossible ages with median calculated from realistic ages (0-120), rounded to integer",
                impact=f"Corrects {invalid_count} impossible values",
                risk="Assumes unrealistic values are data errors",
                is_recommended=True,
                metadata={"median_value": median_age, "invalid_count": invalid_count, "round_to_int": True}
            ))

            fixes.append(DataFix(
                fix_id="FIX_IMPOSSIBLE_AGE_TO_NAN",
                issue_id=issue_id,
                column=column,
                fix_label="Mark as Missing (NaN)",
                fix_description="Treat impossible ages as missing values",
                impact=f"Marks {invalid_count} values as missing",
                risk="Increases missingness, requires subsequent imputation",
                is_recommended=False,
                metadata={"invalid_count": invalid_count}
            ))

            fixes.append(DataFix(
                fix_id="FIX_DROP_IMPOSSIBLE_AGE_ROWS",
                issue_id=issue_id,
                column=column,
                fix_label=f"Drop Rows with Impossible Ages ({invalid_count} rows)",
                fix_description="Remove all rows with biologically impossible ages",
                impact=f"Removes {invalid_count} rows from dataset",
                risk="Data loss",
                is_recommended=False,
                metadata={"rows_to_drop": invalid_count}
            ))

        # INVALID MONETARY VALUES (salary/price <= 0)
        elif "INVALID_MONETARY" in issue_id:
            invalid_count = len(numeric_col[numeric_col <= 0])

            # Calculate median from VALID monetary values only (> 0)
            valid_values = numeric_col[numeric_col > 0]
            median_value = valid_values.median() if not valid_values.empty else 50000

            fixes.append(DataFix(
                fix_id="FIX_ZERO_MONETARY_TO_MEDIAN",
                issue_id=issue_id,
                column=column,
                fix_label=f"Replace with Median ({median_value:.0f})",
                fix_description="Replace zero/negative monetary values with median of valid amounts (> 0)",
                impact=f"Corrects {invalid_count} invalid values",
                risk="Assumes zeros are data errors, not intentional",
                is_recommended=True,
                metadata={"median_value": median_value, "invalid_count": invalid_count}
            ))

            fixes.append(DataFix(
                fix_id="FIX_ZERO_MONETARY_TO_NAN",
                issue_id=issue_id,
                column=column,
                fix_label="Mark as Missing (NaN)",
                fix_description="Treat zero/negative values as missing",
                impact=f"Marks {invalid_count} values as missing",
                risk="Increases missingness",
                is_recommended=False,
                metadata={"invalid_count": invalid_count}
            ))

            fixes.append(DataFix(
                fix_id="FIX_DROP_ZERO_MONETARY_ROWS",
                issue_id=issue_id,
                column=column,
                fix_label=f"Drop Rows ({invalid_count} rows)",
                fix_description="Remove all rows with zero/negative monetary values",
                impact=f"Removes {invalid_count} rows from dataset",
                risk="Data loss",
                is_recommended=False,
                metadata={"rows_to_drop": invalid_count}
            ))

        return fixes