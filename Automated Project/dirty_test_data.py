import pandas as pd
import numpy as np

data = {
    "customer_id": [1, 2, 2, 4, 5, 6],  # Duplicate ID (2)
    "name": ["John Doe", " Jane Smith", "Bob  ", "Alice?", "unknown", "None"], # Whitespace, ?, unknown
    "age": [25, -5, 30, "twenty", 45, 150], # Negative age, Mixed type (string), Outlier (150)
    "salary": [50000, 0, 60000, "70,000", np.nan, 55000], # Salary=0, Comma string, Null
    "joined_date": ["2023-01-01", "01/05/2023", "2023.07.12", "not_a_date", "2023-12-12", "2023-05-01"], # Mixed formats, Invalid date
    "completion_pct": [85, 90, 120, 45, 0, 75] # Percentage > 100
}

df_dirty = pd.DataFrame(data)
df_dirty.to_csv("dirty_test_data.csv", index=False)
print("dirty_test_data.csv has been created.")