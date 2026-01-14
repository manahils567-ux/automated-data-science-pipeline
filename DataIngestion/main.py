from AutoIngestion import AutoIngestion
from schema_validator import SchemaValidator

files = [
    "C:\\Users\\lenovo\\Desktop\\SEM 3\\DSA\\project\\customers.csv",
    "data/data.xlsx",
    "data/data.json"
]

if __name__ == "__main__":
    for file_path in files:
        print(f"\n Loading file: {file_path}")

        try:
            df = AutoIngestion(file_path).run()

            if df is None:
                continue

            validator = SchemaValidator(
                df,
                expected_schema=None,   # auto schema
                strict=False,
                min_columns=1,
                source=file_path
            )

            df, schema_report = validator.run()

            print("Small Data Report:")
            for k, v in schema_report.items():
                print(f"{k}: {v}")

            print("\nPreview:")
            print(df.head())

        except Exception as e:
            print(f" Pipeline failed for {file_path}: {e}")
