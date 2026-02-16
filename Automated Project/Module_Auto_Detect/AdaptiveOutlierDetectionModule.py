# Module_Auto_Detect/AdaptiveOutlierDetectionModule.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

class AdaptiveOutlierModule:
    def __init__(self, method='isolation_forest', contamination=0.05, random_state=42):
        """
        Adaptive outlier detection that works on numeric features and handles NaNs automatically.

        Parameters:
        -----------
        method : str
            'isolation_forest' or 'lof' (Local Outlier Factor)
        contamination : float
            Expected proportion of outliers
        random_state : int
            Random state for reproducibility
        """
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.numeric_cols = None
        self.fitted = False

    def _prepare_data(self, df):
        """
        Select numeric columns and impute missing values with median.
        Returns imputed DataFrame.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = numeric_cols

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns available for outlier detection.")

        # Median imputation for NaNs
        df_numeric = df[numeric_cols].copy()
        df_numeric = df_numeric.astype('float64')
        df_numeric = df_numeric.fillna(df_numeric.median())

        return df_numeric

    def fit(self, df):
        """
        Fit the outlier detection model on numeric features
        """
        df_numeric = self._prepare_data(df)

        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            )
            self.model.fit(df_numeric)
        elif self.method == 'lof':
            # LOF does not have a fit method; fit_predict is used
            self.model = LocalOutlierFactor(
                n_neighbors=min(20, len(df_numeric)-1),
                contamination=self.contamination,
                novelty=True  # Allow separate detect
            )
            self.model.fit(df_numeric)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.fitted = True
        return self

    def detect(self, df):
        """
        Detect outliers in a DataFrame.
        Returns 1 for inliers, -1 for outliers
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Only use columns that exist in df
        available_cols = [c for c in self.numeric_cols if c in df.columns]
        if len(available_cols) < 2:
            print(" Warning: Not enough columns available for detection. Skipping outlier detection.")
            return pd.Series(np.ones(len(df)), index=df.index)  # Treat all as inliers

        df_numeric = df[available_cols].astype('float64')
        df_numeric = df_numeric.fillna(df_numeric.median())

        if self.method == 'isolation_forest':
            preds = self.model.predict(df_numeric)
        elif self.method == 'lof':
            preds = self.model.predict(df_numeric)
        return pd.Series(preds, index=df.index)

    def get_clean_data(self, df):
        """
        Returns DataFrame without detected outliers
        """
        if 'Outlier' not in df.columns:
            df['Outlier'] = self.detect(df)
        return df[df['Outlier'] == 1].copy()

    def summary(self, df):
        """
        Print summary of outliers
        """
        if 'Outlier' not in df.columns:
            df['Outlier'] = self.detect(df)
        total = len(df)
        outliers = (df['Outlier'] == -1).sum()
        percent = 100 * outliers / total
        print("📊 Adaptive Outlier Detection Summary")
        print("-" * 36)
        print(f"Total Samples: {total}")
        print(f"Detected Outliers: {outliers}")
        print(f"Outlier Percentage: {percent:.2f}%")

    def visualize(self, df, cols=None):
        """
        Visualize outliers safely
        """
        if 'Outlier' not in df.columns:
            df['Outlier'] = self.detect(df)

        # Choose columns for plotting
        available_cols = [c for c in (cols or self.numeric_cols[:2]) if c in df.columns]
        if len(available_cols) < 2:
            print("⚠ Warning: Not enough numeric columns to visualize outliers.")
            return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x=available_cols[0],
            y=available_cols[1],
            hue='Outlier',
            palette={1: 'green', -1: 'red'},
            alpha=0.6
        )
        plt.title("Adaptive Outlier Detection Scatter Plot")
        plt.show()

