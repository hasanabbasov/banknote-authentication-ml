"""
Banknote Authentication veri seti üzerinde SVM sınıflandırıcı oluşturur.
"""

import pandas as pd
from sklearn.svm import SVC

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


def train_svm_classifier(features, target, kernel="rbf", C=1.0, gamma="scale"):
    """
    Verilen özellikler ve hedef üzerinde SVM modelini eğitir.
    """
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma)
    svm_model.fit(features, target)
    return svm_model


if __name__ == "__main__":
    df, X, y = load_banknote_data()
    print(f"Toplam örnek sayısı: {len(df)}")
    svm_model = train_svm_classifier(X, y)
    print("SVM modeli tüm veri seti üzerinde eğitildi.")
