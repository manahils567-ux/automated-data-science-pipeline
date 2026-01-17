import pandas as pd
import numpy as np
from ..FixObject import DataFix


class TypeMismatchStrategy:
    """
    Handles type mismatch issues like "twenty" in numeric columns.
    """

    # Word to number mapping
    WORD_TO_NUM = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
    }

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def generate_fixes(self, issue):
        """
        Generate fixes for type mismatch issues (text in numeric columns).
        """
        fixes = []
        column = issue["column"]
        issue_id = issue["issue_id"]

        # Find text values in numeric column
        text_values = self.df[self.df[column].astype(str).str.contains(r'[a-zA-Z]', na=False)][column]
        invalid_count = len(text_values)

        # Check if values can be converted using word mapping
        can_convert = False
        sample_words = text_values.astype(str).str.lower().str.strip().unique()[:5]

        for word in sample_words:
            if word in self.WORD_TO_NUM:
                can_convert = True
                break

        if can_convert:
            fixes.append(DataFix(
                fix_id="FIX_WORD_TO_NUMBER",
                issue_id=issue_id,
                column=column,
                fix_label="Convert Text to Numeric (e.g., 'twenty' → 20)",
                fix_description="Convert word representations to numeric values",
                impact=f"Converts {invalid_count} text values to numbers",
                risk="Limited to common English number words",
                is_recommended=True,
                metadata={
                    "invalid_count": invalid_count,
                    "sample_conversions": {w: self.WORD_TO_NUM.get(w, '?') for w in sample_words if
                                           w in self.WORD_TO_NUM}
                }
            ))

        # Replace with NaN then impute
        col_metadata = self.metadata["columns"].get(column, {})
        numeric_values = pd.to_numeric(self.df[column], errors='coerce')
        median_val = numeric_values.median()

        fixes.append(DataFix(
            fix_id="FIX_TEXT_TO_NAN_IMPUTE",
            issue_id=issue_id,
            column=column,
            fix_label=f"Replace with NaN, then Impute Median ({median_val:.2f})",
            fix_description="Treat text values as missing, then impute with median",
            impact=f"Marks {invalid_count} values as missing, then imputes",
            risk="Loss of original information",
            is_recommended=not can_convert,  # Recommend if word conversion not possible
            metadata={"median_value": median_val, "invalid_count": invalid_count}
        ))

        # Drop rows with invalid values
        fixes.append(DataFix(
            fix_id="FIX_DROP_TEXT_ROWS",
            issue_id=issue_id,
            column=column,
            fix_label=f"Drop Rows with Text Values ({invalid_count} rows)",
            fix_description="Remove all rows containing text in numeric column",
            impact=f"Removes {invalid_count} rows from dataset",
            risk="Data loss",
            is_recommended=False,
            metadata={"rows_to_drop": invalid_count}
        ))

        return fixes