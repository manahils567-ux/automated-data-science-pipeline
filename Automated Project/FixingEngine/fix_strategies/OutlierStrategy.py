import pandas as pd
import numpy as np
from ..FixObject import DataFix


class OutlierStrategy:
    """
    Generates fix recommendations for statistical outliers (Z-score based).
    """

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def generate_fixes(self, issue):
        """
        Generate fixes for extreme outlier issues.
        """
        fixes = []
        column = issue["column"]
        issue_id = issue["issue_id"]

        # Convert to numeric and drop NaN for calculations
        col_data = pd.to_numeric(self.df[column], errors='coerce').dropna()

        # Check if we have enough data
        if len(col_data) < 3:
            return fixes

        # Calculate outlier statistics
        mean_val = col_data.mean()
        std_val = col_data.std()

        # Avoid division by zero
        if std_val == 0:
            return fixes

        z_scores = np.abs((col_data - mean_val) / std_val)
        outlier_count = len(col_data[z_scores > 3])

        # If no outliers, return empty
        if outlier_count == 0:
            return fixes

        # Calculate percentiles for capping
        p1 = col_data.quantile(0.01)
        p99 = col_data.quantile(0.99)

        # IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap at percentiles (most recommended)
        fixes.append(DataFix(
            fix_id="FIX_CAP_PERCENTILE",
            issue_id=issue_id,
            column=column,
            fix_label=f"Cap at 1st/99th Percentile ({p1:.2f} - {p99:.2f})",
            fix_description="Cap extreme values at 1st and 99th percentiles",
            impact=f"Caps approximately {outlier_count} extreme values",
            risk="Preserves distribution shape while removing extremes",
            is_recommended=True,
            metadata={
                "p1": p1,
                "p99": p99,
                "outlier_count": outlier_count
            }
        ))

        # IQR-based capping
        fixes.append(DataFix(
            fix_id="FIX_CAP_IQR",
            issue_id=issue_id,
            column=column,
            fix_label=f"Cap at IQR Boundaries ({lower_bound:.2f} - {upper_bound:.2f})",
            fix_description="Cap values beyond 1.5×IQR from quartiles",
            impact=f"Caps values outside IQR range",
            risk="More aggressive than percentile method",
            is_recommended=False,
            metadata={
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR
            }
        ))

        # Remove outliers
        fixes.append(DataFix(
            fix_id="FIX_REMOVE_OUTLIERS",
            issue_id=issue_id,
            column=column,
            fix_label=f"Remove Outlier Rows (Z-score > 3)",
            fix_description="Drop all rows with extreme outlier values",
            impact=f"Removes approximately {outlier_count} rows",
            risk="Data loss, reduced sample size",
            is_recommended=False,
            metadata={"rows_to_drop": outlier_count}
        ))

        # Winsorization (transform to bounds)
        p5 = col_data.quantile(0.05)
        p95 = col_data.quantile(0.95)

        fixes.append(DataFix(
            fix_id="FIX_WINSORIZE",
            issue_id=issue_id,
            column=column,
            fix_label=f"Winsorize at 5th/95th Percentile ({p5:.2f} - {p95:.2f})",
            fix_description="Replace outliers with nearest non-outlier value",
            impact="Preserves all rows while reducing extreme values",
            risk="May still leave some outliers",
            is_recommended=False,
            metadata={
                "p5": p5,
                "p95": p95
            }
        ))

        return fixes