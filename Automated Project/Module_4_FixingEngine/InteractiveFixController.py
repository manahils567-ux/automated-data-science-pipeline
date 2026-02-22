import sys
import time
from .FixExecutor import FixExecutor
from .FixRecommendationEngine import FixRecommendationEngine


def _flush_stdin():
    """
    Discard any pending characters that were buffered into stdin by background
    events (e.g. closing a matplotlib window, pandas stderr output) before we
    call input().  A short sleep lets any in-flight writes settle first; then
    we drain whatever is already sitting in the buffer without blocking.
    Only the Windows path uses msvcrt; on other platforms the except branch
    is a no-op so the function is always safe to call.
    """
    time.sleep(0.15)
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    except Exception:
        pass


def _prompt(message):
    """
    Flush stdout so the prompt is visible, drain stale stdin bytes, then
    call input().  Returns the stripped, lowercased response string.
    """
    sys.stdout.flush()
    _flush_stdin()
    return input(message).strip().lower()


def _prompt_raw(message):
    """
    Same as _prompt but returns the original strip (no lower) for fix number input.
    """
    sys.stdout.flush()
    _flush_stdin()
    return input(message).strip()


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

            column = issue.get('column')

            if column not in self.executor.df.columns:
                print(f"\nISSUE {idx} / {len(self.issues)}")
                print("-" * 50)
                print(f"Type     : {issue['issue_type']}")
                print(f"Column   : {column}")
                print(f"⚠ Skipped - Column no longer exists (was dropped earlier)")
                continue

            print(f"\nISSUE {idx} / {len(self.issues)}")
            print("-" * 50)
            print(f"Type     : {issue['issue_type']}")
            print(f"Column   : {column}")
            print(f"Severity : {issue['severity']}")
            print(f"Details  : {issue['description']}")

            choice = _prompt("Do you want to fix this issue? (y/n): ")

            if choice != 'y':
                print("⭕ Skipped")
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
                selected = int(_prompt_raw("Select fix number: ")) - 1
                selected_fix = fixes[selected]
            except:
                print("Invalid selection. Skipping issue.")
                continue

            self.executor.apply_fix(selected_fix)

            print(f"✓ Applied fix: {selected_fix.fix_label}")

        return self.executor.df, self.executor.execution_log