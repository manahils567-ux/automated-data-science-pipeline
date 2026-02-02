import pandas as pd
import os


class CSVIngestion:
    """
    Handles ingestion of CSV files
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = None

    def check_file_exists(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("CSV file not found.")

    def load_csv(self):
        try:
            self.dataframe = pd.read_csv(self.file_path)
            print("CSV file loaded successfully")
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    def run(self):
        self.check_file_exists()
        self.load_csv()
        return self.dataframe


# ==============================
# EXECUTION
# ==============================

if __name__ == "__main__":
    file_path = "data.csv"

    csv_ingestion = CSVIngestion(file_path)
    df = csv_ingestion.run()

    print("\nPreview:")
    print(df.head())
