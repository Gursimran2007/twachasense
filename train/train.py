"""
train.py — SCALE-PATH SCAFFOLD (optional, not needed for the demo).

Transfer-learning trainer for a production skin-condition CNN. The browser app
runs fine on the in-browser KNN few-shot model (js/classifier.js); use this only
once a brand has accumulated a labelled Indian-skin dataset and wants a proper
model exported to TensorFlow.js.

Dataset layout (ImageFolder style):
    data/
      Clear/*.jpg  Acne/*.jpg  Fungal/*.jpg  Eczema/*.jpg
      Pigmentation/*.jpg  Dryness/*.jpg

Pipeline:  images -> MobileNetV2 backbone (frozen) -> small head -> train
           -> save Keras -> tensorflowjs_converter -> load in browser.

Run (use anaconda python if system python wheels are broken):
    pip install -r requirements.txt
    python train.py --data ./data --epochs 15 --out ./skin_model
    tensorflowjs_converter --input_format=keras \
        skin_model/model.keras web_model/    # drop web_model/ into the app
"""
import argparse
import os

CLASSES = ["Clear", "Acne", "Fungal", "Eczema", "Pigmentation", "Dryness"]
IMG = 224


def build_model(n_classes: int):
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze backbone; fine-tune later if needed
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_datasets(data_dir: str, batch: int):
    import tensorflow as tf
    common = dict(image_size=(IMG, IMG), batch_size=batch, label_mode="categorical",
                  class_names=CLASSES)
    train = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=42, **common)
    val = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=42, **common)
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomBrightness(0.1),  # lighting robustness for varied skin tones
    ])
    train = train.map(lambda x, y: (aug(norm(x)), y)).prefetch(1)
    val = val.map(lambda x, y: (norm(x), y)).prefetch(1)
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="./skin_model")
    args = ap.parse_args()

    train, val = make_datasets(args.data, args.batch)
    model = build_model(len(CLASSES))
    model.summary()
    model.fit(train, validation_data=val, epochs=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "model.keras"))
    print(f"Saved -> {args.out}/model.keras")
    print("Next: tensorflowjs_converter --input_format=keras "
          f"{args.out}/model.keras web_model/")


if __name__ == "__main__":
    main()
