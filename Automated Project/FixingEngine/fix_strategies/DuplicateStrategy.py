import pandas as pd
from ..FixObject import DataFix


class DuplicateStrategy:
    """
    Generates fix recommendations for duplicate records and ID clashes.
    """

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def generate_fixes(self, issue):
        """
        Generate fixes for duplicate issues.
        """
        fixes = []
        column = issue["column"]
        issue_id = issue["issue_id"]

        # ID Clash (duplicate IDs)
        if "ID_CLASH" in issue_id:
            duplicate_count = self.df[column].duplicated().sum()

            fixes.append(DataFix(
                fix_id="FIX_KEEP_FIRST_ID",
                issue_id=issue_id,
                column=column,
                fix_label=f"Keep First Occurrence ({duplicate_count} duplicates removed)",
                fix_description="Keep the first occurrence of each ID, remove subsequent duplicates",
                impact=f"Removes {duplicate_count} duplicate ID entries",
                risk="May lose more recent/updated records",
                is_recommended=True,
                metadata={"duplicates_removed": duplicate_count}
            ))

            fixes.append(DataFix(
                fix_id="FIX_KEEP_LAST_ID",
                issue_id=issue_id,
                column=column,
                fix_label=f"Keep Last Occurrence",
                fix_description="Keep the last occurrence of each ID (assumes latest is correct)",
                impact=f"Removes {duplicate_count} duplicate ID entries",
                risk="May lose historical records",
                is_recommended=False,
                metadata={"duplicates_removed": duplicate_count}
            ))

            # Most complete record (fewest nulls)
            fixes.append(DataFix(
                fix_id="FIX_KEEP_COMPLETE",
                issue_id=issue_id,
                column=column,
                fix_label="Keep Most Complete Record",
                fix_description="For each duplicate ID, keep the row with fewest missing values",
                impact=f"Intelligently removes {duplicate_count} duplicates",
                risk="More complex logic, may take longer to process",
                is_recommended=False,
                metadata={"duplicates_removed": duplicate_count}
            ))

        # Exact duplicate rows (all columns identical)
        else:
            duplicate_rows = self.df.duplicated().sum()

            fixes.append(DataFix(
                fix_id="FIX_DROP_EXACT_DUPLICATES",
                issue_id=issue_id,
                column="All Columns",
                fix_label=f"Remove Exact Duplicate Rows ({duplicate_rows} rows)",
                fix_description="Remove rows that are completely identical across all columns",
                impact=f"Removes {duplicate_rows} duplicate rows",
                risk="Minimal - exact duplicates have no unique information",
                is_recommended=True,
                metadata={"duplicates_removed": duplicate_rows}
            ))

        return fixes