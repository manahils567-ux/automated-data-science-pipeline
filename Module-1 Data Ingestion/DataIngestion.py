# ==============================
# STEP 1: DATA INGESTION MODULE
# ==============================

import pandas as pd
import os


class DataIngestion:
    """
    This class handles:
    - Loading dataset
    - Basic validation
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = None

    def check_file_exists(self):
        """
        Checks whether dataset file exists
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("Dataset file not found. Check the file path.")
        print("File found")

    def load_data(self):
        """
        Loads the dataset into a pandas DataFrame
        """
        self.dataframe = pd.read_csv(self.file_path)
        print("Dataset loaded successfully")

    def dataset_summary(self):
        """
        Prints dataset structure and health info
        """
        print("\nDATASET SUMMARY")
        print("\n")

        print("Shape (Rows, Columns):", self.dataframe.shape)

        print("\nColumn Names:")
        for col in self.dataframe.columns:
            print("-", col)

        print("\nData Types:")
        print(self.dataframe.dtypes)

        print("\nMissing Values per Column:")
        print(self.dataframe.isnull().sum())

    def run(self):
        """
        Executes the complete ingestion pipeline
        """
        self.check_file_exists()
        self.load_data()
        self.dataset_summary()
        return self.dataframe


# PIPELINE EXECUTION

if __name__ == "__main__":
    file_path = "data.csv"   # change path if needed

    ingestion = DataIngestion(file_path)
    df = ingestion.run()

    print("\n🚀 Step 1 Completed. Dataset is ready for cleaning.")

  
