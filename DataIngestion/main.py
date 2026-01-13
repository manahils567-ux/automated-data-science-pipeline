from csv_Ingestion import CSVIngestion
from excel_Ingestion import ExcelIngestion
from json_Ingestion import JSONIngestion


def run_csv():
    csv_path = "C:\\Users\\lenovo\\Desktop\\SEM 3\\DSA\\project\\customers.csv"
    try:
        df = CSVIngestion(csv_path).run()
        print("\nCSV Preview:")
        print(df.head())
    except FileNotFoundError as fnf:
        print(f"CSV file not found: {fnf}")
    except ValueError as ve:
        print(f"CSV loading error: {ve}")
    except Exception as e:
        print(f"Unexpected error in CSV ingestion: {e}")


def run_excel():
    excel_path = "data/data.xlsx"
    try:
        df = ExcelIngestion(excel_path).run()
        print("\nExcel Preview:")
        print(df.head())
    except FileNotFoundError as fnf:
        print(f"Excel file not found: {fnf}")
    except ValueError as ve:
        print(f"Excel loading error: {ve}")
    except Exception as e:
        print(f"Unexpected error in Excel ingestion: {e}")


def run_json():
    json_path = "data/data.json"
    try:
        df = JSONIngestion(json_path).run()
        print("\nJSON Preview:")
        print(df.head())
    except FileNotFoundError as fnf:
        print(f"JSON file not found: {fnf}")
    except ValueError as ve:
        print(f"JSON loading error: {ve}")
    except Exception as e:
        print(f"Unexpected error in JSON ingestion: {e}")


if __name__ == "__main__":
    print("=== Running CSV Ingestion ===")
    run_csv()
    print("\n=== Running Excel Ingestion ===")
    run_excel()
    print("\n=== Running JSON Ingestion ===")
    run_json()

