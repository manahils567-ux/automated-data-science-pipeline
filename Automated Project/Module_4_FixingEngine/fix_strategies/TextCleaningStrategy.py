import pandas as pd
from ..FixObject import DataFix


class TextCleaningStrategy:
    """
    Generates fix recommendations for text-related issues.
    ENHANCED: Better handling of special characters - removes them BEFORE mode calculation.
    """

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def generate_fixes(self, issue):
        """
        Generate fixes for text cleaning issues.
        """
        fixes = []
        column = issue["column"]
        issue_id = issue["issue_id"]

        # Whitespace issues
        if "WHITESPACE" in issue_id:
            affected_count = self.df[column].astype(str).str.contains(r'^\s|\s$', na=False).sum()

            fixes.append(DataFix(
                fix_id="FIX_STRIP_WHITESPACE",
                issue_id=issue_id,
                column=column,
                fix_label=f"Strip Leading/Trailing Spaces ({affected_count} values)",
                fix_description="Remove whitespace from beginning and end of text",
                impact=f"Cleans {affected_count} values",
                risk="None - standard text cleaning",
                is_recommended=True,
                metadata={"affected_count": affected_count}
            ))

        # Encoding artifacts
        elif "ENCODING_JUNK" in issue_id:
            affected_count = self.df[column].astype(str).str.contains(r'[^\x00-\x7F]+', na=False).sum()

            fixes.append(DataFix(
                fix_id="FIX_REMOVE_NON_ASCII",
                issue_id=issue_id,
                column=column,
                fix_label=f"Remove Non-ASCII Characters ({affected_count} values)",
                fix_description="Remove or replace corrupted/non-ASCII characters",
                impact=f"Cleans {affected_count} values",
                risk="May remove legitimate foreign characters",
                is_recommended=True,
                metadata={"affected_count": affected_count}
            ))

        # Special characters - HIGHEST PRIORITY
        elif "SPECIAL_CHARS" in issue_id:
            affected_count = self.df[column].astype(str).str.contains(r'[?!@#$%^&*]', na=False).sum()

            fixes.append(DataFix(
                fix_id="FIX_REMOVE_SPECIAL_CHARS",
                issue_id=issue_id,
                column=column,
                fix_label=f"Remove Special Characters ({affected_count} values)",
                fix_description="Remove special characters (?, !, @, #, etc.) from text - APPLIES FIRST",
                impact=f"Cleans {affected_count} values",
                risk="May remove intentional punctuation",
                is_recommended=True,
                metadata={"affected_count": affected_count, "priority": 1}
            ))

            fixes.append(DataFix(
                fix_id="FIX_REPLACE_SPECIAL_WITH_SPACE",
                issue_id=issue_id,
                column=column,
                fix_label="Replace Special Characters with Space",
                fix_description="Replace special characters with spaces",
                impact=f"Modifies {affected_count} values",
                risk="May create extra spaces",
                is_recommended=False,
                metadata={"affected_count": affected_count}
            ))

        # Case inconsistencies
        elif "CASE_DIVERGE" in issue_id:
            upper_count = self.df[column].astype(str).str.isupper().sum()
            lower_count = self.df[column].astype(str).str.islower().sum()

            fixes.append(DataFix(
                fix_id="FIX_STANDARDIZE_CASE_LOWER",
                issue_id=issue_id,
                column=column,
                fix_label="Convert All to Lowercase",
                fix_description="Standardize all text to lowercase",
                impact="Ensures consistent casing across column",
                risk="May lose semantic meaning (e.g., proper nouns)",
                is_recommended=True,
                metadata={"upper_count": upper_count, "lower_count": lower_count}
            ))

        # Proxy missingness (placeholder tokens)
        elif "PROXY_MISSING" in issue_id:
            affected_values = self.df[column].astype(str).str.lower().str.strip()
            tokens = ["?", "unknown", "n/a", "none", "null", "."]
            affected_count = affected_values.isin(tokens).sum()

            fixes.append(DataFix(
                fix_id="FIX_PROXY_TO_NAN",
                issue_id=issue_id,
                column=column,
                fix_label=f"Convert Placeholders to NaN ({affected_count} values)",
                fix_description="Replace placeholder tokens (?, NA, unknown, none, null) with proper NaN",
                impact=f"Marks {affected_count} values as missing",
                risk="Increases missingness, requires subsequent imputation",
                is_recommended=True,
                metadata={"affected_count": affected_count, "priority": 3}
            ))

        # Empty text variants
        elif "EMPTY_TEXT" in issue_id:
            affected_count = self.df[column].astype(str).str.strip().isin(['', 'nan', 'NaN', 'None', 'NONE']).sum()

            valid_values = self.df[column].dropna()

            if 'email' not in column.lower():
                valid_values = valid_values[~valid_values.astype(str).str.contains(r'[?!#$%^&*]', na=False)]

            valid_values = valid_values[
                ~valid_values.astype(str).str.lower().str.strip().isin(['unknown', 'none', 'null', 'n/a', '', 'nan'])]

            valid_values = valid_values[valid_values.astype(str).str.strip() != '']

            if not valid_values.empty:
                mode_val = valid_values.mode()
                if not mode_val.empty:
                    mode_val = mode_val[0]

                    fixes.append(DataFix(
                        fix_id="FIX_EMPTY_TEXT_TO_MODE",
                        issue_id=issue_id,
                        column=column,
                        fix_label=f"Replace with Mode ('{mode_val}')",
                        fix_description="Replace empty/nan text with most frequent CLEAN value",
                        impact=f"Fills {affected_count} empty values",
                        risk="May increase class imbalance",
                        is_recommended=True,
                        metadata={"mode_value": mode_val, "affected_count": affected_count}
                    ))

            fixes.append(DataFix(
                fix_id="FIX_EMPTY_TEXT_TO_NAN",
                issue_id=issue_id,
                column=column,
                fix_label="Convert to Proper NaN",
                fix_description="Convert text representations of missing to actual NaN",
                impact=f"Standardizes {affected_count} missing values",
                risk="None - improves data consistency",
                is_recommended=not valid_values.empty,
                metadata={"affected_count": affected_count}
            ))

        return fixes