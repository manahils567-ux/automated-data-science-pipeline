from AutoIngestion import AutoIngestion

files = [
    "C:\\Users\\lenovo\\Desktop\\SEM 3\\DSA\\project\\customers.csv",  # CSV
    "data/data.xlsx",  # Excel
    "data/data.json"   # JSON
]

if __name__ == "__main__":
    for file_path in files:
        print(f"\n Loading file: {file_path} ")
        df = AutoIngestion(file_path).run()
        if df is not None:
            print("\nPreview:")
            print(df.head())
