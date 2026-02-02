import pandas as pd
import numpy as np


class ImpactAnalyzer:
    """
    Analyzes the impact of data cleaning operations.
    Compares before and after states of the dataset.
    """

    def __init__(self, original_df, cleaned_df, execution_log):
        self.original_df = original_df
        self.cleaned_df = cleaned_df
        self.execution_log = execution_log

    def calculate_completeness(self, df):
        """
        Calculate overall completeness percentage.
        """
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        return (non_null_cells / total_cells) * 100 if total_cells > 0 else 0

    def calculate_column_completeness(self, df):
        """
        Calculate completeness for each column.
        """
        completeness = {}
        for col in df.columns:
            total = len(df)
            non_null = df[col].count()
            completeness[col] = (non_null / total) * 100 if total > 0 else 0
        return completeness

    def count_issues(self, df):
        """
        Count various data quality issues in a DataFrame.
        """
        issues = {
            "missing_values": df.isna().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }
        return issues

    def calculate_quality_score(self, df):
        """
        Calculate an overall data quality score (0-100).
        Based on completeness, duplicates, and other factors.
        """
        completeness = self.calculate_completeness(df)
        duplicate_penalty = (df.duplicated().sum() / len(df)) * 10 if len(df) > 0 else 0

        quality_score = completeness - duplicate_penalty
        return max(0, min(100, quality_score))  # Clamp between 0-100

    def generate_report(self):
        """
        Generate a comprehensive impact analysis report.
        """
        # Before stats
        before_issues = self.count_issues(self.original_df)
        before_completeness = self.calculate_completeness(self.original_df)
        before_col_completeness = self.calculate_column_completeness(self.original_df)
        before_quality = self.calculate_quality_score(self.original_df)

        # After stats
        after_issues = self.count_issues(self.cleaned_df)
        after_completeness = self.calculate_completeness(self.cleaned_df)
        after_col_completeness = self.calculate_column_completeness(self.cleaned_df)
        after_quality = self.calculate_quality_score(self.cleaned_df)

        # Improvements
        completeness_improvement = after_completeness - before_completeness
        quality_improvement = after_quality - before_quality

        report = {
            "before": {
                "total_rows": before_issues["total_rows"],
                "total_columns": before_issues["total_columns"],
                "missing_values": before_issues["missing_values"],
                "duplicate_rows": before_issues["duplicate_rows"],
                "completeness": before_completeness,
                "quality_score": before_quality,
                "column_completeness": before_col_completeness
            },
            "after": {
                "total_rows": after_issues["total_rows"],
                "total_columns": after_issues["total_columns"],
                "missing_values": after_issues["missing_values"],
                "duplicate_rows": after_issues["duplicate_rows"],
                "completeness": after_completeness,
                "quality_score": after_quality,
                "column_completeness": after_col_completeness
            },
            "improvements": {
                "rows_removed": before_issues["total_rows"] - after_issues["total_rows"],
                "columns_removed": before_issues["total_columns"] - after_issues["total_columns"],
                "missing_values_fixed": before_issues["missing_values"] - after_issues["missing_values"],
                "duplicates_removed": before_issues["duplicate_rows"] - after_issues["duplicate_rows"],
                "completeness_gain": completeness_improvement,
                "quality_score_gain": quality_improvement
            },
            "execution_log": self.execution_log
        }

        return report

    def display_report(self, report):
        """
        Display the impact analysis report in a formatted manner.
        """
        print(f"\n{'=' * 60}")
        print("BEFORE vs AFTER COMPARISON")
        print(f"{'=' * 60}\n")

        # Basic stats
        print(f"Total Rows       : {report['before']['total_rows']} → {report['after']['total_rows']}", end="")
        if report['improvements']['rows_removed'] > 0:
            print(f" ({report['improvements']['rows_removed']} rows removed)")
        else:
            print()

        print(f"Total Columns    : {report['before']['total_columns']} → {report['after']['total_columns']}", end="")
        if report['improvements']['columns_removed'] > 0:
            print(f" ({report['improvements']['columns_removed']} columns removed)")
        else:
            print()

        # Completeness
        print(f"\n{'-' * 60}")
        print("COMPLETENESS IMPROVEMENT")
        print(f"{'-' * 60}")

        # Column-level completeness changes
        before_cols = report['before']['column_completeness']
        after_cols = report['after']['column_completeness']

        for col in before_cols:
            if col in after_cols:  # Column still exists
                before_pct = before_cols[col]
                after_pct = after_cols[col]
                improvement = after_pct - before_pct

                if improvement > 0:
                    print(f"  {col:<20} : {before_pct:>5.1f}% → {after_pct:>5.1f}% (+{round(improvement, 1)}%)")

        # Overall completeness
        before_comp = report['before']['completeness']
        after_comp = report['after']['completeness']
        comp_gain = after_comp - before_comp

        print(f"\n  Overall          : {before_comp:.1f}% → {after_comp:.1f}% ", end="")
        if comp_gain > 0.05:
            print(f"(+{comp_gain:.1f}%)")
        elif comp_gain < -0.05:
            print(f"({comp_gain:.1f}%)")
        else:
            print()

        # Quality improvements
        print(f"\n{'-' * 60}")
        print("QUALITY IMPROVEMENTS")
        print(f"{'-' * 60}")

        if report['improvements']['missing_values_fixed'] > 0:
            reduction_pct = (report['improvements']['missing_values_fixed'] / report['before']['missing_values']) * 100
            print(
                f"  ✓ Missing Values    : {report['before']['missing_values']} → {report['after']['missing_values']} (-{reduction_pct:.0f}%)")

        if report['improvements']['duplicates_removed'] > 0:
            print(
                f"  ✓ Duplicate Rows    : {report['before']['duplicate_rows']} → {report['after']['duplicate_rows']} (-100%)")

        # Execution summary
        print(f"\n{'-' * 60}")
        print("FIXES APPLIED")
        print(f"{'-' * 60}")

        for log_entry in report['execution_log']:
            print(f"  ✓ {log_entry['column']:<20} : {log_entry['fix_applied']}")

        # Data quality score
        before_quality = report['before']['quality_score']
        after_quality = report['after']['quality_score']
        quality_gain = after_quality - before_quality

        print(f"\n{'-' * 60}")
        print(f"DATA QUALITY SCORE : {before_quality:.0f}% → {after_quality:.0f}% ", end="")
        if quality_gain > 0.5:
            print(f"(+{quality_gain:.0f}%)")
        elif quality_gain < -0.5:
            print(f"({quality_gain:.0f}%)")
        else:
            print()
        print(f"{'=' * 60}\n")


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    # Sample before/after dataframes
    original_df = pd.DataFrame({
        'age': [25, None, 30, None, 45],
        'salary': [50000, 60000, None, 70000, 80000]
    })

    cleaned_df = pd.DataFrame({
        'age': [25, 30, 30, 30, 45],
        'salary': [50000, 60000, 65000, 70000, 80000]
    })

    execution_log = [
        {"column": "age", "fix_applied": "Median imputation", "values_changed": 2},
        {"column": "salary", "fix_applied": "Mean imputation", "values_changed": 1}
    ]

    analyzer = ImpactAnalyzer(original_df, cleaned_df, execution_log)
    report = analyzer.generate_report()
    analyzer.display_report(report)