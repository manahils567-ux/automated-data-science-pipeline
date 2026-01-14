import pandas as pd
import os


class ExcelIngestion:
    """
    Handles ingestion of Excel files
    """

    def __init__(self, file_path, sheet_name=0):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.dataframe = None

    def check_file_exists(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("Excel file not found.")

    def load_excel(self):
        try:
            self.dataframe = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name
            )
            print("Excel file loaded successfully")
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {e}")

    def run(self):
        self.check_file_exists()
        self.load_excel()
        return self.dataframe


# ==============================
# EXECUTION
# ==============================

if __name__ == "__main__":
    file_path = "data.xlsx"

    excel_ingestion = ExcelIngestion(file_path)
    df = excel_ingestion.run()

    print("\nPreview:")
    print(df.head())
