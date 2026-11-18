"""Run all available banana models and print their predictions as JSON.

The models intentionally keep their original input contracts:
  * banana days model: image + 39 sensor features
  * banana ripeness model: image only
  * existing fusion TFLite model: image + 27 sensor features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DAYS_PATH = MODELS / "banana_days_to_ripe_multimodal_portable (3).keras"
RIPENESS_PATH = MODELS / "banana_ripeness_model2_finetuned_best (9).keras"
FUSION_PATH = MODELS / "fusion_blocked_full_float16.tflite"
SENSOR27_COLUMNS = [
    "Temperature", "Humidity", "Gas", "Temperature_Smoothed", "Humidity_Smoothed",
    "Gas_Smoothed", "Temperature_Change", "Humidity_Change", "Gas_Change",
    "Temperature_Rate", "Humidity_Rate", "Gas_Rate", "Temperature_Rolling_Mean",
    "Humidity_Rolling_Mean", "Gas_Rolling_Mean", "Temperature_Rolling_STD",
    "Humidity_Rolling_STD", "Gas_Rolling_STD", "Temperature_Humidity_Interaction",
    "Temperature_Gas_Interaction", "Humidity_Gas_Interaction", "Temperature_From_Initial",
    "Humidity_From_Initial", "Gas_From_Initial", "Gas_Increasing", "Temperature_Increasing",
    "Humidity_Increasing",
]


def vector(value: str, size: int, name: str) -> np.ndarray:
    """Parse a comma-separated or JSON sensor vector."""
    try:
        parsed = json.loads(value) if value.lstrip().startswith("[") else value.split(",")
        data = np.asarray(parsed, dtype="float32").reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a comma-separated list or JSON array") from exc
    if len(data) != size:
        raise ValueError(f"{name} needs exactly {size} values; received {len(data)}")
    return data[None, :]


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    # The banana Keras models include their own image preprocessing layers and expect
    # the conventional float image range used during their training.
    return np.asarray(image, dtype="float32")[None, ...] / 255.0


def invoke_tflite(interpreter, image: np.ndarray, sensor27: np.ndarray) -> float:
    details = interpreter.get_input_details()
    for detail in details:
        if len(detail["shape"]) == 4:
            value = image
        elif len(detail["shape"]) == 2:
            value = sensor27
        else:
            raise ValueError(f"unexpected TFLite input shape: {detail['shape']}")
        if detail["dtype"] in (np.int8, np.uint8):
            scale, zero = detail["quantization"]
            value = np.round(value / scale + zero).clip(
                np.iinfo(detail["dtype"]).min, np.iinfo(detail["dtype"]).max
            )
        interpreter.set_tensor(detail["index"], value.astype(detail["dtype"]))
    interpreter.invoke()
    output = interpreter.get_output_details()[0]
    raw = interpreter.get_tensor(output["index"])
    if output["dtype"] in (np.int8, np.uint8):
        scale, zero = output["quantization"]
        raw = (raw.astype("float32") - zero) * scale
    return float(np.asarray(raw).reshape(-1)[0])


def latest_record(dataset: Path | None):
    candidates = [dataset] if dataset else [ROOT / "artifacts" / "master_dataset.csv",
                                            ROOT / "sensor_features.csv"]
    candidates = [p for p in candidates if p and p.is_file()]
    if not candidates:
        raise FileNotFoundError("No CSV test dataset was found. Use --mode user or --dataset.")
    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    frame = pd.read_csv(chosen)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {chosen}")
    row = frame.iloc[-1]
    if "Timestamp" in frame:
        row = frame.sort_values("Timestamp").iloc[-1]
    image_name = row.get("image_file", row.get("Image_Path"))
    if not image_name:
        raise ValueError(f"Dataset {chosen} has no image_file/Image_Path column")
    image_path = Path(str(image_name))
    search = [image_path, chosen.parent / image_path, ROOT / "data_images" / image_path,
              ROOT / "data_images" / "Processed_Images" / image_path]
    image_path = next((p for p in search if p.is_file()), None)
    if image_path is None:
        raise FileNotFoundError(f"Latest dataset image was not found: {image_name}")
    missing = [name for name in SENSOR27_COLUMNS if name not in row.index]
    if missing:
        raise ValueError(f"Dataset {chosen} is missing 27-feature columns: {missing}")
    sensor27 = np.asarray([row[name] for name in SENSOR27_COLUMNS], dtype="float32")[None, :]
    sensor39_names = [f"sensor39_{i}" for i in range(39)]
    if not all(name in row.index for name in sensor39_names):
        raise ValueError("Latest dataset has the 27-feature vector but no compatible 39-feature vector "
                         "for the banana days model. Supply --sensor39 in user mode.")
    sensor39 = np.asarray([row[name] for name in sensor39_names], dtype="float32")[None, :]
    return image_path, sensor39, sensor27, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ask", "user", "latest"), default="ask",
                        help="ask for a source, use manual values, or use the newest dataset row")
    parser.add_argument("--dataset", type=Path, help="CSV for --mode latest")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--sensor39",
                        help="39 values for the banana days-to-ripe model")
    parser.add_argument("--sensor27",
                        help="27 values for the existing fusion TFLite model")
    args = parser.parse_args()

    mode = args.mode
    if mode == "ask":
        answer = input("Use manual uploaded input or latest test dataset? [user/latest]: ").strip().lower()
        mode = "latest" if answer in {"latest", "l", "no", "n"} else "user"
    source = None
    if mode == "latest":
        args.image, sensor39, sensor27, source = latest_record(args.dataset)
    else:
        args.image = args.image or Path(input("Image path: ").strip())
        args.sensor39 = args.sensor39 or input("39 sensor values (comma-separated): ").strip()
        args.sensor27 = args.sensor27 or input("27 sensor values (comma-separated): ").strip()
        sensor39 = vector(args.sensor39, 39, "--sensor39")
        sensor27 = vector(args.sensor27, 27, "--sensor27")
    image = load_image(args.image)

    days_model = tf.keras.models.load_model(DAYS_PATH, compile=False)
    ripeness_model = tf.keras.models.load_model(RIPENESS_PATH, compile=False)
    days = days_model.predict([image, sensor39], verbose=0).reshape(-1)[0]
    ripeness = ripeness_model.predict(image, verbose=0).reshape(-1)

    interpreter = tf.lite.Interpreter(model_path=str(FUSION_PATH))
    interpreter.allocate_tensors()
    fusion_days = invoke_tflite(interpreter, image * 255.0, sensor27)

    result = {
        "source": str(source or args.image),
        "banana_days_model": {"days_to_ripe": float(days)},
        "banana_ripeness_model": {
            "class_index": int(np.argmax(ripeness)),
            "probabilities": [float(x) for x in ripeness],
        },
        "fusion_tflite_model": {"days_to_ripe": fusion_days},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
