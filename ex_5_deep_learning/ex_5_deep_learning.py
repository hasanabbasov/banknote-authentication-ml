"""
Bardak / Kalem / Kitap fotoğraflarıyla 2 model (baseline CNN ve transfer learning) eğiten betik.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Veri yolları ve sabitler
BASE_DIR = Path(__file__).resolve().parent.parent / "data-image"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 10  # küçük veri olduğu için kısa tutuyoruz
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets():
    """
    data-image klasöründen train/val dataset'lerini oluşturur.
    """
    train_raw = image_dataset_from_directory(
        BASE_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_raw = image_dataset_from_directory(
        BASE_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_raw.class_names  # cache öncesi al

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_raw.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_raw.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names




def build_baseline_cnn(num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_v2(num_classes):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # önce donduruyoruz

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, model_name):
    acc = history.history.get("accuracy", []) or history.history.get("categorical_accuracy", [])
    val_acc = history.history.get("val_accuracy", []) or history.history.get("val_categorical_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].legend()

    fig.tight_layout()
    output_path = PLOTS_DIR / f"{model_name.lower().replace(' ', '_')}_history.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def train_and_evaluate():
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    # Baseline CNN
    cnn = build_baseline_cnn(num_classes)
    cnn_history = cnn.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    cnn_eval = cnn.evaluate(val_ds, verbose=0)
    cnn_history_path = plot_history(cnn_history, "Baseline CNN")

    # Transfer learning - MobileNetV2
    mobilenet = build_mobilenet_v2(num_classes)
    mobilenet_history = mobilenet.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    mobilenet_eval = mobilenet.evaluate(val_ds, verbose=0)
    mobilenet_history_path = plot_history(mobilenet_history, "MobileNetV2")

    results = [
        {
            "model": "Baseline CNN",
            "val_loss": cnn_eval[0],
            "val_accuracy": cnn_eval[1],
            "history_plot": cnn_history_path,
        },
        {
            "model": "MobileNetV2",
            "val_loss": mobilenet_eval[0],
            "val_accuracy": mobilenet_eval[1],
            "history_plot": mobilenet_history_path,
        },
    ]
    return results, class_names


def print_results(results, class_names):
    print(f"Sınıflar: {class_names}")
    for res in results:
        print(f"=== {res['model']} ===")
        print(f"Val Loss: {res['val_loss']:.4f}")
        print(f"Val Accuracy: {res['val_accuracy']:.4f}")
        print(f"Eğitim grafiği: {res['history_plot']}")
        print("-" * 40)


if __name__ == "__main__":
    results, class_names = train_and_evaluate()
    print_results(results, class_names)
