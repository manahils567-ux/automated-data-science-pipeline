import pandas as pd
import os


class JSONIngestion:
    """
    Handles ingestion of JSON files
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataframe = None

    def check_file_exists(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("JSON file not found.")

    def load_json(self):
        try:
            raw_data = pd.read_json(self.file_path)

            # Handle nested JSON
            if isinstance(raw_data, pd.DataFrame):
                self.dataframe = raw_data
            else:
                self.dataframe = pd.json_normalize(raw_data)

            print("JSON file loaded successfully")

        except ValueError:
            # Fallback for complex nested JSON
            try:
                raw_data = pd.read_json(self.file_path, lines=True)
                self.dataframe = raw_data
                print("JSON Lines file loaded successfully")
            except Exception as e:
                raise ValueError(f"Error loading JSON file: {e}")

    def run(self):
        self.check_file_exists()
        self.load_json()
        return self.dataframe


# ==============================
# EXECUTION
# ==============================

if __name__ == "__main__":
    file_path = "data.json"

    json_ingestion = JSONIngestion(file_path)
    df = json_ingestion.run()

    print("\nPreview:")
    print(df.head())
