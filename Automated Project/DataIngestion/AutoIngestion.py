
import os
from .csv_Ingestion import CSVIngestion
from .excel_Ingestion import ExcelIngestion
from .json_Ingestion import JSONIngestion

class AutoIngestion:
    """
    Automatically detects the file type (CSV / Excel / JSON)
    and uses the corresponding ingestion class.
    """

    def __init__(self, file_path, excel_sheet=0):

        self.file_path = file_path
        self.excel_sheet = excel_sheet
        self.dataframe = None

    def detect_file_type(self):
        _, ext = os.path.splitext(self.file_path)
        return ext.lower()

    def run(self):
        ext = self.detect_file_type()

        try:
            if ext == ".csv":
                self.dataframe = CSVIngestion(self.file_path).run()
                print(f" Auto-detected CSV file: {self.file_path}")

            elif ext in [".xls", ".xlsx"]:
                self.dataframe = ExcelIngestion(self.file_path, sheet_name=self.excel_sheet).run()
                print(f" Auto-detected Excel file: {self.file_path}")

            elif ext == ".json":
                self.dataframe = JSONIngestion(self.file_path).run()
                print(f" Auto-detected JSON file: {self.file_path}")

            else:
                raise ValueError(f"Unsupported file type: {ext}")

        except FileNotFoundError as fnf:
            print(f" File not found: {fnf}")
        except ValueError as ve:
            print(f" File loading error: {ve}")
        except Exception as e:
            print(f" Unexpected error during ingestion: {e}")

        return self.dataframe


# ==============================
# Example Usage (Remove in production)
# ==============================

if __name__ == "__main__":
    test_files = [
        "C:\\Users\\lenovo\\Desktop\\SEM 3\\DSA\\project\\customers.csv",
        "data/data.xlsx",
        "data/data.json"
    ]

    for f in test_files:
        print(f"\n=== Loading: {f} ===")
        df = AutoIngestion(f).run()
        if df is not None:
            print(df.head())
