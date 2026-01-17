from DataIngestion.AutoIngestion import AutoIngestion
from DataProfiling.schema_validator import SchemaValidator
from DataProfiling.DataTypeInferencer import DataTypeInferencer
from DataProfiling.metadata_extractor import MetadataExtractor
from IssueDetection.DetectionEngine import IssueDetectionEngine
from FixingEngine.FixRecommendationEngine import FixRecommendationEngine
from FixingEngine.FixExecutor import FixExecutor
from FixingEngine.ImpactAnalyzer import ImpactAnalyzer

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

files = [
    "dirty_test_data.csv"
]

if __name__ == "__main__":

    print(f"\n{BOLD}{CYAN}AUTOMATED DATA SCIENCE PIPELINE{RESET}")
    print(f"{GRAY}{'=' * 60}{RESET}\n")

    for file_path in files:
        print(f"{BOLD}{BLUE}FILE:{RESET} {file_path}")
        print(f"{GRAY}{'-' * 60}{RESET}")

        try:
            # ================= STEP 2.1 & 2.2 =================
            print(f"\n{YELLOW}STEP 2.1–2.2 : AUTO INGESTION{RESET}")

            df = AutoIngestion(file_path).run()

            if df is None or df.empty:
                print(f"{RED}STATUS : No data returned (empty or failed){RESET}\n")
                continue

            print(f"{GREEN}STATUS : Ingestion successful{RESET}")
            print(f"Rows    : {df.shape[0]}")
            print(f"Columns : {df.shape[1]}")

            # ================= STEP 2.3 =================
            print(f"\n{YELLOW}STEP 2.3 : SCHEMA VALIDATION{RESET}")

            validator = SchemaValidator(
                df,
                expected_schema=None,
                strict=False,
                min_columns=1,
                source=file_path
            )

            df, schema_report = validator.run()

            for key, value in schema_report.items():
                print(f"{CYAN}{key:<22}{RESET} : {value}")

            # ================= STEP 2.4 =================
            print(f"\n{YELLOW}STEP 2.4 : DATA TYPE INFERENCE{RESET}")

            inferencer = DataTypeInferencer(df)
            df, type_report = inferencer.infer()

            for col, dtype in type_report.items():
                print(f"{MAGENTA}{col:<22}{RESET}  : {dtype}")

            # ================= STEP 2.5 =================
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

                if "sample" in info:
                    print(f"Sample      : {info['sample']}")

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

            # ================= STEP 3.0 : ISSUE DETECTION (YOUR MODULE) =================
            print(f"\n{YELLOW}STEP 3.0 : ISSUE DETECTION ENGINE{RESET}")

            engine = IssueDetectionEngine(df)
            detected_issues = engine.run_all_checks()

            if not detected_issues:
                print(f"{GREEN}STATUS : No critical issues detected!{RESET}")
            else:
                print(f"{RED}STATUS : {len(detected_issues)} Issues Identified{RESET}")
                
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

            # ================= STEP 4.0 : FIX RECOMMENDATION ENGINE =================
            print(f"\n{YELLOW}STEP 4.0 : FIXING ENGINE{RESET}")

            if not detected_issues:
                print(f"{GREEN}STATUS : No fixes needed - dataset is clean!{RESET}")
            else:
                print(f"{CYAN}STATUS : Analyzing {len(detected_issues)} detected issues...{RESET}")

                recommender = FixRecommendationEngine(df, detected_issues, metadata)
                all_fixes = recommender.generate_recommendations()

                # Display all fix options
                recommender.display_recommendations(all_fixes)

                # ================= STEP 4.1 : APPLY RECOMMENDED FIXES =================
                print(f"\n{YELLOW}STEP 4.1 : APPLYING RECOMMENDED FIXES{RESET}")
                print(f"{CYAN}STATUS : Executing best-practice cleaning operations...{RESET}\n")

                executor = FixExecutor(df)
                cleaned_df, execution_log = executor.apply_recommended_fixes(all_fixes)

                print(f"{GREEN}✓ Applied {len(execution_log)} cleaning operations{RESET}")

                # ================= STEP 4.2 : IMPACT ANALYSIS =================
                print(f"\n{YELLOW}STEP 4.2 : DATA CLEANING IMPACT ANALYSIS{RESET}")

                analyzer = ImpactAnalyzer(df, cleaned_df, execution_log)
                impact_report = analyzer.generate_report()
                analyzer.display_report(impact_report)

                # Save cleaned dataset with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = file_path.replace(".csv", f"_cleaned_{timestamp}.csv")
                cleaned_df.to_csv(output_path, index=False)

                print(f"{GREEN}✓ Cleaned dataset saved: {output_path}{RESET}")

                # Use cleaned_df for preview
                df = cleaned_df

            # ================= PREVIEW =================
            print(f"\n{BOLD}{CYAN}DATA PREVIEW{RESET}")
            print(df.head())

            print(f"\n{GRAY}{'=' * 60}{RESET}\n")

        except FileNotFoundError:
            print(f"{RED}ERROR : File not found{RESET}\n")

        except Exception as e:
            print(f"{RED}ERROR : Pipeline failed{RESET}")
            print(f"Reason : {e}\n")

    print(f"{BOLD}{GREEN}PIPELINE EXECUTION COMPLETED{RESET}\n")
