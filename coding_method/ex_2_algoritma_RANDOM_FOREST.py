"""
Banknote Authentication veri seti üzerinde Random Forest sınıflandırıcı oluşturur.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
COLUMN_NAMES = ["variance", "skewness", "curtosis", "entropy", "class"]

def load_banknote_data():
    """
    Banknote Authentication verisetini indirip DataFrame, özellikler ve hedef olarak döndürür.
    """
    df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)
    features = df[COLUMN_NAMES[:-1]]
    target = df["class"]
    return df, features, target


def train_random_forest(features, target, n_estimators=100, random_state=42):
    """
    Verilen özellikler ve hedef üzerinde Random Forest modelini eğitir.
    """
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(features, target)
    return rf_model


if __name__ == "__main__":
    df, X, y = load_banknote_data()
    print(f"Toplam örnek sayısı: {len(df)}")
    rf_model = train_random_forest(X, y)
    print("Random Forest modeli tüm veri seti üzerinde eğitildi.")
