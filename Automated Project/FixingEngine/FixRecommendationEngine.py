from .fix_strategies.MissingValueStrategy import MissingValueStrategy
from .fix_strategies.NumericValidityStrategy import NumericValidityStrategy
from .fix_strategies.TypeMismatchStrategy import TypeMismatchStrategy
from .fix_strategies.OutlierStrategy import OutlierStrategy
from .fix_strategies.DuplicateStrategy import DuplicateStrategy
from .fix_strategies.TextCleaningStrategy import TextCleaningStrategy
from .fix_strategies.DateFormatStrategy import DateFormatStrategy
from .fix_strategies.DomainValidationStrategy import DomainValidationStrategy


class FixRecommendationEngine:
    """
    Main orchestrator for generating fix recommendations.
    Routes each detected issue to appropriate strategy.
    """

    def __init__(self, df, detected_issues, metadata):
        self.df = df
        self.detected_issues = detected_issues
        self.metadata = metadata

        # Initialize all strategy handlers
        self.strategies = {
            "Missing Data": MissingValueStrategy(df, metadata),
            "Proxy Missingness": TextCleaningStrategy(df, metadata),
            "Numeric Validity": NumericValidityStrategy(df, metadata),
            "Range Violation": NumericValidityStrategy(df, metadata),
            "Type Mismatch": TypeMismatchStrategy(df, metadata),
            "Extreme Outlier": OutlierStrategy(df, metadata),
            "Identity Clash": DuplicateStrategy(df, metadata),
            "Structural Noise": TextCleaningStrategy(df, metadata),
            "Encoding Artifact": TextCleaningStrategy(df, metadata),
            "Format Divergence": DateFormatStrategy(df, metadata),
            "Invalid Date Format": DateFormatStrategy(df, metadata),
            "Domain Constraint Violation": DomainValidationStrategy(df, metadata)
        }

    def generate_recommendations(self):
        """
        Generate fix recommendations for all detected issues.
        Returns a list of DataFix objects grouped by issue.
        """
        all_fixes = []

        for issue in self.detected_issues:
            issue_type = issue.get("issue_type", "")

            strategy = self.strategies.get(issue_type)

            if strategy:
                fixes = strategy.generate_fixes(issue)
                all_fixes.extend(fixes)
            else:
                print(f"⚠ No strategy found for issue type: {issue_type}")

        return all_fixes

    def display_recommendations(self, all_fixes):
        """
        Display fix recommendations in a formatted manner.
        """
        if not all_fixes:
            print("✓ No fixes needed - dataset is clean!")
            return

        print(f"\n{'=' * 60}")
        print(f"AVAILABLE FIX OPTIONS")
        print(f"{'=' * 60}\n")

        fixes_by_issue = {}
        for fix in all_fixes:
            issue_key = f"{fix.issue_id}_{fix.column}"
            if issue_key not in fixes_by_issue:
                fixes_by_issue[issue_key] = []
            fixes_by_issue[issue_key].append(fix)

        for issue_key, fixes in fixes_by_issue.items():
            first_fix = fixes[0]
            print(f"\n[{first_fix.issue_id} - {first_fix.column}]")
            print(f"{'-' * 60}")

            for idx, fix in enumerate(fixes, 1):
                rec_tag = " [RECOMMENDED]" if fix.is_recommended else ""
                print(f"\n  {idx}. {fix.fix_label}{rec_tag}")
                print(f"     Impact : {fix.impact}")
                print(f"     Risk   : {fix.risk}")

        print(f"\n{'=' * 60}\n")

    def get_recommended_fixes(self, all_fixes):
        """
        Extract only the recommended fixes from all available fixes.
        """
        return [fix for fix in all_fixes if fix.is_recommended]


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    import pandas as pd

    # Sample data with issues
    df = pd.DataFrame({
        'age': [25, -5, 30, None, 45],
        'salary': [50000, 60000, 'twenty', 70000, 80000],
        'email': ['a@b.com', 'a@b.com', 'c@d.com', 'd@e.com', 'f@g.com']
    })

    # Sample detected issues
    issues = [
        {
            "issue_id": "MISSING_VAL",
            "column": "age",
            "issue_type": "Missing Data",
            "severity": "High",
            "description": "1 null values.",
            "examples": [None]
        },
        {
            "issue_id": "WORD_AS_NUMBER",
            "column": "salary",
            "issue_type": "Type Mismatch",
            "severity": "High",
            "description": "Found text values in numeric column.",
            "examples": ["twenty"]
        }
    ]

    # Sample metadata
    metadata = {
        "columns": {
            "age": {"dtype": "float64", "mean": 33.33, "std": 10.0},
            "salary": {"dtype": "object"}
        }
    }

    engine = FixRecommendationEngine(df, issues, metadata)
    all_fixes = engine.generate_recommendations()
    engine.display_recommendations(all_fixes)