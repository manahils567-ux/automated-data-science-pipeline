from .FixExecutor import FixExecutor
from .FixRecommendationEngine import FixRecommendationEngine
class InteractiveFixController:

    def __init__(self, df, issues, metadata):
        self.executor = FixExecutor(df)
        self.issues = issues
        self.metadata = metadata
        self.execution_log = []

        self.recommender = FixRecommendationEngine(
            self.executor.df,
            self.issues,
            self.metadata
        )

    def run(self):

        for idx, issue in enumerate(self.issues, start=1):

            print(f"\nISSUE {idx} / {len(self.issues)}")
            print("-" * 50)
            print(f"Type     : {issue['issue_type']}")
            print(f"Column   : {issue['column']}")
            print(f"Severity : {issue['severity']}")
            print(f"Details  : {issue['description']}")

            choice = input("Do you want to fix this issue? (y/n): ").strip().lower()

            if choice != 'y':
                print("⏭ Skipped")
                continue

            fixes = self.recommender.generate_recommendations_for_issue(issue)

            if not fixes:
                print("No fix available.")
                continue

            print("\nAvailable Fixes:")
            for i, fix in enumerate(fixes, start=1):
                tag = " (Recommended)" if fix.is_recommended else ""
                print(f"[{i}] {fix.fix_label}{tag}")

            try:
                selected = int(input("Select fix number: ")) - 1
                selected_fix = fixes[selected]
            except:
                print("Invalid selection. Skipping issue.")
                continue

            # APPLY FIX (stateful)
            self.executor.apply_fix(selected_fix)

            print(f"✓ Applied fix: {selected_fix.fix_label}")

        return self.executor.df, self.executor.execution_log
