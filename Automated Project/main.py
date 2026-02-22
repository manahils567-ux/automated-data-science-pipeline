import pandas as pd
from Module_1_DataIngestion.AutoIngestion import AutoIngestion
from Module_2_DataProfiling.schema_validator import SchemaValidator
from Module_2_DataProfiling.DataTypeInferencer import DataTypeInferencer
from Module_2_DataProfiling.metadata_extractor import MetadataExtractor
from Module_3_IssueDetection.DetectionEngine import IssueDetectionEngine
from Module_4_FixingEngine.ImpactAnalyzer import ImpactAnalyzer
from Module_4_FixingEngine.InteractiveFixController import InteractiveFixController

from Module_Auto_Detect.AdaptiveOutlierDetectionModule import AdaptiveOutlierModule
from datetime import datetime
from colorama import init
init(autoreset=True)

# ---------- ANSI STYLES ----------
RESET = "\033[0m"
BOLD = "\033[1m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
GRAY = "\033[90m"

files = ["dirty_test_data.csv"]

if __name__ == "__main__":

    print(f"\n{BOLD}{CYAN}AUTOMATED DATA SCIENCE PIPELINE{RESET}")
    print(f"{GRAY}{'=' * 60}{RESET}\n")

    for file_path in files:
        print(f"{BOLD}{BLUE}FILE:{RESET} {file_path}")
        print(f"{GRAY}{'-' * 60}{RESET}")

        try:
            # ================= STEP 2.1 & 2.2 : Auto Ingestion =================
            print(f"\n{YELLOW}STEP 2.1–2.2 : AUTO INGESTION{RESET}")
            df = AutoIngestion(file_path).run()

            if df is None or df.empty:
                print(f"{MAGENTA}STATUS : No data returned (empty or failed){RESET}\n")
                continue

            print(f"{GREEN}STATUS : Ingestion successful{RESET}")
            print(f"Rows    : {df.shape[0]}")
            print(f"Columns : {df.shape[1]}")

            # ================= STEP 2.3 : Schema Validation =================
            print(f"\n{YELLOW}STEP 2.3 : SCHEMA VALIDATION{RESET}")
            validator = SchemaValidator(df, expected_schema=None, strict=False, min_columns=1, source=file_path)
            df, schema_report = validator.run()
            for key, value in schema_report.items():
                print(f"{CYAN}{key:<22}{RESET} : {value}")

            # ================= STEP 2.4 : Data Type Inference =================
            print(f"\n{YELLOW}STEP 2.4 : DATA TYPE INFERENCE{RESET}")
            inferencer = DataTypeInferencer(df)
            df, type_report = inferencer.infer()
            for col, dtype in type_report.items():
                print(f"{MAGENTA}{col:<22}{RESET}  : {dtype}")

            # ================= STEP 2.5 : Metadata Extraction =================
            print(f"\n{YELLOW}STEP 2.5 : METADATA EXTRACTION{RESET}")
            metadata = MetadataExtractor(df, source=file_path).run()
            print(f"\n{BOLD}{CYAN}DATASET OVERVIEW{RESET}")
            print(f"Rows        : {metadata['rows']}")
            print(f"Columns     : {len(metadata['columns'])}")
            print(f"Source File : {metadata['source_file']}")
            print(f"Ingested At : {metadata['ingested_at']}")

            print(f"\n{BOLD}{CYAN}COLUMN DETAILS{RESET}")
            print(f"{GRAY}{'-' * 60}{RESET}")
            for col_name, info in metadata["columns"].items():
                print(f"\n{BOLD}{BLUE}Column : {col_name}{RESET}")
                print(f"Type        : {info.get('dtype')}")
                print(f"Missing     : {info.get('nulls')}")
                print(f"Unique      : {info.get('unique')}")
                if "sample" in info: print(f"Sample      : {info['sample']}")
                if "mean" in info:
                    print(f"{GREEN}Numeric Statistics{RESET}")
                    print(f"  Mean      : {round(info['mean'], 3)}")
                    print(f"  Std Dev   : {round(info['std'], 3)}")
                    print(f"  Min       : {info['min']}")
                    print(f"  Max       : {info['max']}")
                if "min_date" in info:
                    print(f"{CYAN}Date Range{RESET}")
                    print(f"  Earliest  : {info['min_date']}")
                    print(f"  Latest    : {info['max_date']}")
                if "top_values" in info:
                    print(f"{MAGENTA}Top Values{RESET}")
                    for val, count in info["top_values"]:
                        print(f"  {val} ({count})")
                if "avg_length" in info:
                    print(f"Avg Length  : {round(info['avg_length'], 2)}")

            # ================= STEP 3.x : Adaptive Outlier Detection =================
            print(f"\n{YELLOW}STEP 3.x : ADAPTIVE OUTLIER DETECTION{RESET}")
            try:
                outlier_detector = AdaptiveOutlierModule(contamination=0.05)
                outlier_detector.fit(df)  # handles NaNs internally
                df['Outlier'] = outlier_detector.detect(df)
                outlier_detector.summary(df)
                outlier_detector.visualize(df)
                clean_df = outlier_detector.get_clean_data(df)
                clean_df = clean_df.drop(columns=['Outlier'], errors='ignore')
                print(f"{GREEN}STATUS : Cleaned dataset without outliers has {clean_df.shape[0]} rows{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠ Outlier detection skipped due to error: {e}{RESET}")
                clean_df = df.copy()

            # ================= STEP 4.0 : Issue Detection Engine =================
            print(f"\n{YELLOW}STEP 4.0 : ISSUE DETECTION ENGINE{RESET}")
            engine = IssueDetectionEngine(clean_df)
            detected_issues = engine.run_all_checks()
            if not detected_issues:
                print(f"{GREEN}STATUS : No critical issues detected!{RESET}")
            else:
                print(f"{MAGENTA}STATUS : {len(detected_issues)} Issues Identified{RESET}")
                print(f"\n{BOLD}{CYAN}DETECTION REPORT{RESET}")
                print(f"{GRAY}{'-' * 60}{RESET}")
                for issue in detected_issues:
                    color = RED if issue['severity'] == "High" else YELLOW
                    print(f"\n{BOLD}{color}[{issue['issue_id']}]{RESET}")
                    print(f"  {BOLD}Category{RESET}    : {issue['issue_type']}")
                    print(f"  {BOLD}Column{RESET}      : {issue['column']}")
                    print(f"  {BOLD}Severity{RESET}    : {issue['severity']}")
                    print(f"  {BOLD}Description{RESET} : {issue['description']}")
                    if issue['examples'] and issue['examples'] != [None]:
                        print(f"  {BOLD}Examples{RESET}    : {issue['examples']}")

            # ================= STEP 5.0 : Interactive Data Cleaning =================
            print(f"\n{YELLOW}STEP 5.0 : INTERACTIVE DATA CLEANING{RESET}")
            if not detected_issues:
                print(f"{GREEN}STATUS : No fixes needed - dataset is clean!{RESET}")
                cleaned_df = clean_df.copy()
                execution_log = []
            else:
                controller = InteractiveFixController(
                    df=clean_df,
                    issues=detected_issues,
                    metadata=metadata
                )
                cleaned_df, execution_log = controller.run()
                print(f"\n{GREEN}✓ Interactive cleaning completed{RESET}")
                print(f"{CYAN}✓ Total actions logged: {len(execution_log)}{RESET}")

                # ================= STEP 5.1 : Impact Analysis =================
                print(f"\n{YELLOW}STEP 5.1 : DATA CLEANING IMPACT ANALYSIS{RESET}")
                analyzer = ImpactAnalyzer(clean_df, cleaned_df, execution_log)
                impact_report = analyzer.generate_report()
                analyzer.display_report(impact_report)

                # Save cleaned dataset
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = file_path.replace(".csv", f"_cleaned_{timestamp}.csv")
                cleaned_df.drop(columns=['Outlier'], errors='ignore').to_csv(output_path, index=False)
                print(f"{GREEN}✓ Cleaned dataset saved: {output_path}{RESET}")

                # Export execution log
                if execution_log:
                    save_log = input(f"\n{YELLOW}Do you want to save the detailed execution log? (y/n): {RESET}").strip().lower()
                    if save_log == 'y':
                        log_path = file_path.replace(".csv", f"_log_{timestamp}.csv")
                        pd.DataFrame(execution_log).to_csv(log_path, index=False)
                        print(f"{GREEN}✓ Execution log saved: {log_path}{RESET}")

            # ================= FINAL PREVIEW =================
            print(f"\n{BOLD}{CYAN}FINAL DATA PREVIEW{RESET}")
            print(cleaned_df.drop(columns=['Outlier'], errors='ignore').head())
            print(f"\n{GRAY}{'=' * 60}{RESET}\n")

        except FileNotFoundError:
            print(f"{RED}ERROR : File not found{RESET}\n")
        except Exception as e:
            print(f"{RED}ERROR : Pipeline failed{RESET}")
            print(f"Reason : {e}\n")

    print(f"{BOLD}{GREEN}PIPELINE EXECUTION COMPLETED{RESET}\n")