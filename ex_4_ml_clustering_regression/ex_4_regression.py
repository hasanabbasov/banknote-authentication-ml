"""
Hotel_lead_score verisi üzerinde regresyon deneyleri.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from .data_utils import load_hotel_lead_scores
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ex_4_ml_clustering_regression.data_utils import load_hotel_lead_scores

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class Ex4Regression:
    """
    hotel_lead_score verisi üzerinde iki regresyon modeli çalıştırır.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self._split_cache = None

    def _prepare_dataset(self):
        df = load_hotel_lead_scores()
        df["average_score"] = pd.to_numeric(df["average_score"], errors="coerce")
        df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
        df["min_score"] = pd.to_numeric(df["min_score"], errors="coerce")
        df["total_campaigns"] = pd.to_numeric(df["total_campaigns"], errors="coerce")
        df["total_score_sum"] = pd.to_numeric(df["total_score_sum"], errors="coerce")

        df = df.dropna(subset=["average_score", "max_score", "min_score", "total_campaigns", "total_score_sum"])

        features = df[["max_score", "min_score", "total_campaigns", "total_score_sum"]]
        target = df["average_score"]
        return features, target

    def _get_train_test(self):
        if self._split_cache is None:
            X, y = self._prepare_dataset()
            self._split_cache = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        return self._split_cache

    @staticmethod
    def _evaluate(predictions, y_test):
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return {"MAE": mae, "MSE": mse, "R2": r2}

    @staticmethod
    def _plot_predictions(model_name, y_test, predictions):
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.7)
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Mükemmel tahmin")
        ax.set_xlabel("Gerçek average_score")
        ax.set_ylabel("Tahmin edilen average_score")
        ax.set_title(f"{model_name} - Gerçek vs Tahmin")
        ax.legend()
        fig.tight_layout()
        output_path = PLOTS_DIR / f"{model_name.lower().replace(' ', '_')}_pred_vs_actual.png"
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def run_linear_regression(self):
        X_train, X_test, y_train, y_test = self._get_train_test()
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = self._evaluate(predictions, y_test)
        plot_path = self._plot_predictions("Linear Regression", y_test, predictions)
        return {"model": "Linear Regression", "metrics": metrics, "plot_path": plot_path}

    def run_random_forest(self):
        X_train, X_test, y_train, y_test = self._get_train_test()
        model = RandomForestRegressor(random_state=self.random_state, n_estimators=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = self._evaluate(predictions, y_test)
        plot_path = self._plot_predictions("Random Forest Regressor", y_test, predictions)
        return {"model": "Random Forest Regressor", "metrics": metrics, "plot_path": plot_path}

    def run_all(self):
        return [self.run_linear_regression(), self.run_random_forest()]


def print_regression_results(results):
    for result in results:
        print(f"=== {result['model']} ===")
        for metric, value in result["metrics"].items():
            print(f"{metric}: {value:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    reg = Ex4Regression()
    reg_results = reg.run_all()
    print_regression_results(reg_results)
