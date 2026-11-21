"""Run the classification, raw-to-ripe, and RSL fusion models.

Run preprocessinglayer.ipynb first. It writes `prepared_input.npz` with the image
and the two training-compatible sensor vectors. This script routes using the image
classification: unripe -> raw-to-ripe; ripe -> RSL fusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLASS_NAMES = ["unripe", "ripe", "overripe", "rotten"]


def sensor_vector(value, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must contain exactly {size} values; got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr[None, :]


def load_tflite(path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def quantize(value: np.ndarray, detail) -> np.ndarray:
    if detail["dtype"] in (np.int8, np.uint8):
        scale, zero = detail["quantization"]
        if not scale:
            raise ValueError("TFLite input has invalid quantization scale")
        limits = np.iinfo(detail["dtype"])
        value = np.round(value / scale + zero).clip(limits.min, limits.max)
    return value.astype(detail["dtype"])


def tflite_fusion(interpreter, image255: np.ndarray, sensor27: np.ndarray) -> float:
    """Supply preprocessed raw-range pixels and scaled 27-vector to fusion."""
    inputs = interpreter.get_input_details()
    image_detail = next((d for d in inputs if len(d["shape"]) == 4), None)
    sensor_detail = next((d for d in inputs if len(d["shape"]) == 2), None)
    if image_detail is None or sensor_detail is None:
        raise ValueError("Fusion TFLite model must expose image and sensor inputs")
    interpreter.set_tensor(image_detail["index"], quantize(image255, image_detail))
    interpreter.set_tensor(sensor_detail["index"], quantize(sensor27, sensor_detail))
    interpreter.invoke()
    detail = interpreter.get_output_details()[0]
    out = interpreter.get_tensor(detail["index"])
    if detail["dtype"] in (np.int8, np.uint8):
        scale, zero = detail["quantization"]
        out = (out.astype(np.float32) - zero) * scale
    return float(np.asarray(out).reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="prepared_input.npz written by preprocessinglayer.ipynb")
    ap.add_argument("--classification-model", type=Path,
                    default=HERE / "banana_ripeness_model2_finetuned_best.keras")
    ap.add_argument("--days-model", type=Path,
                    default=HERE / "banana_days_to_ripe_multimodal_portable.keras",
                    help="banana_days_to_ripe_multimodal_portable.keras")
    ap.add_argument("--fusion-model", type=Path,
                    default=ROOT / "ml" / "rsl_multimodal" / "models" / "fusion_blocked_full_float16.tflite",
                    help="fusion_blocked_full_float16.tflite")
    ap.add_argument("--output", type=Path, default=HERE / "predictions.json")
    args = ap.parse_args()

    with np.load(args.input, allow_pickle=False) as handoff:
        image01 = np.asarray(handoff["image_01"], dtype=np.float32)
        image255 = np.asarray(handoff["image_255"], dtype=np.float32)
        sensor39_raw = np.asarray(handoff["sensor39"], dtype=np.float32)
        sensor27_raw = np.asarray(handoff["sensor27"], dtype=np.float32)
        class_names = handoff["class_names"].astype(str).tolist()
        image_path = str(handoff["image_path"].item())
    if image01.shape != (1, 224, 224, 3) or image255.shape != (1, 224, 224, 3):
        raise ValueError(f"Prepared image arrays have invalid shapes: {image01.shape}, {image255.shape}")
    if not np.isfinite(image01).all() or not np.isfinite(image255).all():
        raise ValueError("Prepared image arrays contain NaN or infinite values")
    classifier_path = args.classification_model
    if not classifier_path.is_file():
        raise FileNotFoundError(f"Model asset missing: {classifier_path}")
    classifier = tf.keras.models.load_model(classifier_path, compile=False)
    class_prob = np.asarray(classifier(image255, training=False)).reshape(-1)
    if len(class_names) != class_prob.size:
        raise ValueError(f"class_names has {len(class_names)} items but model returns {class_prob.size}")
    class_index = int(np.argmax(class_prob))
    class_name = str(class_names[class_index]).strip().lower()

    selected = None
    if class_name == "unripe":
        if args.days_model is None:
            ap.error("--days-model is required when classification is unripe")
        if not args.days_model.is_file():
            raise FileNotFoundError(f"Model asset missing: {args.days_model}")
        sensor39 = sensor_vector(sensor39_raw, 39, "sensor39")
        days_model = tf.keras.models.load_model(args.days_model, compile=False)
        days = float(np.asarray(days_model(
            {"image_input": image01, "sensor_input": sensor39}, training=False
        )).reshape(-1)[0])
        selected = {"route": "raw_to_ripe", "days_to_ripe": days}
    elif class_name in ["ripe", "overripe", "rotten"]:
        if args.fusion_model is None:
            ap.error("--fusion-model is required for ripe, overripe, and rotten classes")
        if not args.fusion_model.is_file():
            raise FileNotFoundError(f"Model asset missing: {args.fusion_model}")
        sensor27 = sensor_vector(sensor27_raw, 27, "sensor27")
        fusion = tflite_fusion(load_tflite(args.fusion_model), image255, sensor27)
        selected = {"route": "rsl_fusion", "remaining_shelf_life_days": fusion}

    result = {
        "source": image_path,
        "ripeness_classification": {
            "class_index": class_index,
            "class_name": class_names[class_index],
            "probabilities": [float(x) for x in class_prob],
        },
        "selected_prediction": selected,
    }
    if selected is None:
        result["routing_note"] = (
            f"No days estimate configured for class '{class_names[class_index]}'; "
            "only unripe and ripe classes are routed."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
