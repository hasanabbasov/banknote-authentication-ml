"""
Hotel_data verisini StandardScaler + PCA'den geçirerek KMeans ve DBSCAN ile kümeleyen sınıf.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from .data_utils import load_hotel_data
except ImportError:  # Script doğrudan çalıştırıldığında paket yolunu ayarla
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ex_4_ml_clustering_regression.data_utils import load_hotel_data

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class Ex4Clustering:
    """
    Hotel_data verisi üzerinde KMeans ve DBSCAN çalıştırır, metrikleri ve PCA grafiklerini üretir.
    """

    def __init__(self, n_clusters=3, dbscan_eps=0.8, dbscan_min_samples=5):
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=42)
        self._prepared_cache = None

    def _prepare_features(self):
        df = load_hotel_data()
        features = df[["rooms", "stars", "segment", "province"]].copy()
        features["rooms"] = pd.to_numeric(features["rooms"], errors="coerce")
        features["stars"] = pd.to_numeric(features["stars"], errors="coerce")
        features = features.dropna(subset=["rooms", "stars"])
        features = pd.get_dummies(features, columns=["segment", "province"], dummy_na=True)
        features = features.fillna(0)
        return features

    def _get_prepared_data(self):
        if self._prepared_cache is None:
            features = self._prepare_features()
            scaled = self.scaler.fit_transform(features)
            pca_coords = self.pca.fit_transform(scaled)
            self._prepared_cache = (features, scaled, pca_coords)
        return self._prepared_cache

    def _evaluate_clustering(self, X, labels):
        unique_labels = set(labels)
        valid_clusters = unique_labels - {-1}
        if len(valid_clusters) <= 1:
            return None, None
        silhouette = silhouette_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        return silhouette, davies

    def _plot_clusters(self, method_name, pca_coords, labels):
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            c=labels,
            cmap="tab10",
            s=35,
            alpha=0.85,
        )
        ax.set_title(f"{method_name} - PCA 2B Küme Görünümü")
        ax.set_xlabel("PCA Bileşeni 1")
        ax.set_ylabel("PCA Bileşeni 2")
        legend = ax.legend(*scatter.legend_elements(), title="Kümeler", loc="best")
        ax.add_artist(legend)
        fig.tight_layout()
        output_path = PLOTS_DIR / f"{method_name.lower()}_pca.png"
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def run_kmeans(self):
        _, scaled, pca_coords = self._get_prepared_data()
        model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        silhouette, davies = self._evaluate_clustering(scaled, labels)
        plot_path = self._plot_clusters("KMeans", pca_coords, labels)
        return {
            "method": "KMeans",
            "labels": labels,
            "silhouette": silhouette,
            "davies_bouldin": davies,
            "plot_path": plot_path,
            "distribution": pd.Series(labels).value_counts(sort=False).to_dict(),
        }

    def run_dbscan(self):
        _, scaled, pca_coords = self._get_prepared_data()
        model = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = model.fit_predict(scaled)
        silhouette, davies = self._evaluate_clustering(scaled, labels)
        plot_path = self._plot_clusters("DBSCAN", pca_coords, labels)
        return {
            "method": "DBSCAN",
            "labels": labels,
            "silhouette": silhouette,
            "davies_bouldin": davies,
            "plot_path": plot_path,
            "distribution": pd.Series(labels).value_counts(sort=False).to_dict(),
        }

    def run_all(self):
        return [self.run_kmeans(), self.run_dbscan()]


def print_clustering_results(results):
    for result in results:
        sil = f"{result['silhouette']:.4f}" if result["silhouette"] is not None else "Hesaplanamadı"
        db = f"{result['davies_bouldin']:.4f}" if result["davies_bouldin"] is not None else "Hesaplanamadı"
        print(f"=== {result['method']} ===")
        print(f"Silhouette Score: {sil}")
        print(f"Davies-Bouldin Score: {db}")
        print("Küme dağılımı:")
        for cluster_id, count in result["distribution"].items():
            label = "Gürültü (-1)" if cluster_id == -1 else f"Küme {cluster_id}"
            print(f"  {label}: {count} kayıt")
        print(f"PCA grafiği: {result['plot_path']}")
        print("-" * 40)


if __name__ == "__main__":
    clustering = Ex4Clustering(n_clusters=3, dbscan_eps=0.9, dbscan_min_samples=6)
    results = clustering.run_all()
    print_clustering_results(results)
