"""
Banknote Authentication veri seti üzerinde 3.a deneylerini yürütür.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
COLUMN_NAMES = ["variance", "skewness", "curtosis", "entropy", "class"]
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_banknote_data(test_size=0.3, random_state=42):
    """
    Veriyi indirir ve eğitim/test olarak böler.
    """
    df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)
    X = df[COLUMN_NAMES[:-1]]
    y = df["class"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def compute_classification_metrics(y_true, y_pred):
    """
    Confusion matrix, accuracy, sensitivity ve specificity değerlerini döndürür.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)  # recall
    specificity = tn / (tn + fp)
    return {
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }



class SVMExperiment:
    """
    Banknote Authentication verisini SVM ile değerlendiren deney sınıfı.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def run(self):
        X_train, X_test, y_train, y_test = load_banknote_data()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred)
        return metrics


class RandomForestExperiment:
    """
    Banknote Authentication verisini Random Forest ile değerlendiren deney sınıfı.
    """

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def run(self):
        X_train, X_test, y_train, y_test = load_banknote_data()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred)
        return metrics


def plot_confusion_matrix(name, matrix):
    fig, ax = plt.subplots()
    ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Tahmin edilen")
    ax.set_ylabel("Gerçek")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sahte", "Gerçek"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sahte", "Gerçek"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")

    fig.tight_layout()
    output_path = PLOTS_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_metric_bars(name, metrics):
    labels = ["Accuracy", "Sensitivity", "Specificity"]
    values = [metrics["accuracy"], metrics["sensitivity"], metrics["specificity"]]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=["#4caf50", "#2196f3", "#ff9800"])
    ax.set_ylim(0, 1)
    ax.set_title(f"{name} - Performans Metrikleri")
    ax.set_ylabel("Skor")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    output_path = PLOTS_DIR / f"{name.lower().replace(' ', '_')}_metrics.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def print_metrics(name, metrics):
    print(f"=== {name} sonuçları ===")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    svm_experiment = SVMExperiment()
    rf_experiment = RandomForestExperiment()

    svm_metrics = svm_experiment.run()
    rf_metrics = rf_experiment.run()

    print_metrics("SVM", svm_metrics)
    print_metrics("Random Forest", rf_metrics)

    svm_conf_path = plot_confusion_matrix("SVM", svm_metrics["confusion_matrix"])
    svm_metrics_path = plot_metric_bars("SVM", svm_metrics)
    rf_conf_path = plot_confusion_matrix("Random Forest", rf_metrics["confusion_matrix"])
    rf_metrics_path = plot_metric_bars("Random Forest", rf_metrics)

    print("Grafikler kaydedildi:")
    print(f"SVM Confusion Matrix: {svm_conf_path}")
    print(f"SVM Metrics: {svm_metrics_path}")
    print(f"Random Forest Confusion Matrix: {rf_conf_path}")
    print(f"Random Forest Metrics: {rf_metrics_path}")
