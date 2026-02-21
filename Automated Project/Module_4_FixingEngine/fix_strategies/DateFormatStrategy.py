import pandas as pd
from ..FixObject import DataFix


class DateFormatStrategy:
    """
    Generates fix recommendations for date format issues.
    Handles invalid dates, mixed formats, and standardization.
    """

    def __init__(self, df, metadata):
        self.df = df
        self.metadata = metadata

    def generate_fixes(self, issue):
        """
        Generate fixes for date format issues.
        PRIORITY: Standardize formats FIRST, then handle truly invalid dates.
        """
        fixes = []
        column = issue["column"]
        issue_id = issue["issue_id"]

        # MIXED DATE FORMATS - This should be PRIMARY fix (applies to all dates)
        if "DATE_FORMAT_MIXED" in issue_id or "FORMAT_DIVERGENCE" in issue_id:
            valid_dates = self.df[column].dropna()
            sample_formats = valid_dates.head(6).tolist()

            fixes.append(DataFix(
                fix_id="FIX_STANDARDIZE_DATE_FORMAT",
                issue_id=issue_id,
                column=column,
                fix_label="Standardize to YYYY-MM-DD Format (Parses Multiple Formats)",
                fix_description="Intelligently parse and convert all date formats (YYYY-MM-DD, MM/DD/YYYY, DD.MM.YYYY, etc.) to ISO format",
                impact=f"Standardizes all parseable dates to YYYY-MM-DD, leaves truly invalid as-is",
                risk="None - improves consistency while preserving valid data",
                is_recommended=True,
                metadata={"target_format": "YYYY-MM-DD", "sample_before": sample_formats}
            ))

        # INVALID DATE FORMAT - Handle TRULY unparseable dates (like "not_a_date")
        elif "INVALID_DATE_FORMAT" in issue_id:
            parsed_dates = pd.to_datetime(self.df[column], errors='coerce')

            still_invalid_mask = parsed_dates.isna() & self.df[column].notna()

            for fmt in ['%m/%d/%Y', '%d/%m/%Y', '%Y.%m.%d', '%d.%m.%Y']:
                if still_invalid_mask.sum() == 0:
                    break
                try:
                    temp = pd.to_datetime(self.df.loc[still_invalid_mask, column], format=fmt, errors='coerce')
                    parsed_dates.loc[still_invalid_mask] = parsed_dates.loc[still_invalid_mask].fillna(temp)
                    still_invalid_mask = parsed_dates.isna() & self.df[column].notna()
                except:
                    continue

            truly_invalid_count = still_invalid_mask.sum()

            if truly_invalid_count > 0:
                valid_dates = parsed_dates.dropna()
                if len(valid_dates) > 0:
                    median_date = valid_dates.median()
                    median_date_str = pd.Timestamp(median_date).strftime('%Y-%m-%d')

                    # Option 1: Impute with median date
                    fixes.append(DataFix(
                        fix_id="FIX_INVALID_DATE_IMPUTE_MEDIAN",
                        issue_id=issue_id,
                        column=column,
                        fix_label=f"Replace with Median Date ({median_date_str})",
                        fix_description="Replace unparseable dates with the median of valid dates",
                        impact=f"Fills {truly_invalid_count} invalid dates with median date",
                        risk="Assumes invalid dates should follow the central tendency of valid dates",
                        is_recommended=True,
                        metadata={"median_date": median_date_str, "invalid_count": truly_invalid_count}
                    ))

                # Option 2: Convert to NaN
                fixes.append(DataFix(
                    fix_id="FIX_INVALID_DATE_TO_NAN",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Convert Truly Invalid Dates to NaN ({truly_invalid_count} values)",
                    fix_description="Replace completely unparseable strings (like 'not_a_date', 'TBD') with NaN",
                    impact=f"Marks {truly_invalid_count} truly invalid dates as missing",
                    risk="Increases missingness, requires subsequent imputation",
                    is_recommended=False if len(valid_dates) > 0 else True,
                    metadata={"invalid_count": truly_invalid_count}
                ))

                # Option 3: Drop rows
                fixes.append(DataFix(
                    fix_id="FIX_DROP_INVALID_DATE_ROWS",
                    issue_id=issue_id,
                    column=column,
                    fix_label=f"Drop Rows with Invalid Dates ({truly_invalid_count} rows)",
                    fix_description="Remove all rows with unparseable date values",
                    impact=f"Removes {truly_invalid_count} rows from dataset",
                    risk="Data loss",
                    is_recommended=False,
                    metadata={"rows_to_drop": truly_invalid_count}
                ))

                # Option 4: Default date
                fixes.append(DataFix(
                    fix_id="FIX_INVALID_DATE_DEFAULT",
                    issue_id=issue_id,
                    column=column,
                    fix_label="Replace with Default Date (1900-01-01)",
                    fix_description="Replace invalid dates with a sentinel default date",
                    impact=f"Fills {truly_invalid_count} invalid dates with placeholder",
                    risk="May be misleading if not documented",
                    is_recommended=False,
                    metadata={"default_date": "1900-01-01", "invalid_count": truly_invalid_count}
                ))
                
        elif "DATE_SEQUENCE_VIOLATION" in issue_id:
            fixes.append(DataFix(
                fix_id="FIX_SWAP_LOGICAL_DATES",
                issue_id=issue_id,
                column=column, 
                fix_label="Swap Inverted Dates",
                fix_description="Assumes dates were entered in the wrong columns and swaps them.",
                impact="Restores logical timeline for the record",
                risk="Medium - assumes the error was a column swap",
                is_recommended=True
            ))

        return fixes