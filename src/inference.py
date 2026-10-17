import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import (
    PERSON_CONFIDENCE,
    PPE_CONFIDENCE,
    PPE_CLASSES,
    CLASS_COLORS,
    SUPPORTED_EXTENSIONS,
    BOX_THICKNESS,
    FONT,
    FONT_SCALE,
    TEXT_THICKNESS,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Two-stage Person and PPE detection inference.")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where annotated images will be saved.")
    parser.add_argument("--person_det_model", type=str, required=True, help="Path to person detection model weights.")
    parser.add_argument("--ppe_detection_model", type=str, required=True, help="Path to PPE detection model weights.")

    return parser.parse_args()


def draw_detection(image, bbox, label, confidence):
    x1, y1, x2, y2 = map(int, bbox)
    color = CLASS_COLORS.get(label, (255, 255, 255))
    text = f"{label} {confidence:.2f}"
    text_x = x1
    text_y = max(y1 - 10, 20)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    cv2.putText(image, text, (text_x, text_y), FONT, FONT_SCALE, color, TEXT_THICKNESS, cv2.LINE_AA)


def process_image(image_path, output_path, person_model, ppe_model):
    try:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        image_height, image_width = image.shape[:2]

        person_results = person_model.predict(source=image, conf=PERSON_CONFIDENCE, verbose=False)

        if not person_results:
            raise RuntimeError(f"Person model returned no results for: {image_path}")

        person_result = person_results[0]

        if person_result.boxes is None or len(person_result.boxes) == 0:
            if not cv2.imwrite(str(output_path), image):
                raise IOError(f"Failed to save image: {output_path}")
            return

        for person_box in person_result.boxes:
            person_xyxy = person_box.xyxy[0].cpu().numpy()

            person_x1 = max(0, min(int(person_xyxy[0]), image_width))
            person_y1 = max(0, min(int(person_xyxy[1]), image_height))
            person_x2 = max(0, min(int(person_xyxy[2]), image_width))
            person_y2 = max(0, min(int(person_xyxy[3]), image_height))

            person_confidence = float(person_box.conf[0].cpu().item())

            if person_x2 <= person_x1 or person_y2 <= person_y1:
                continue

            person_crop = image[person_y1:person_y2, person_x1:person_x2]

            if person_crop.size == 0:
                continue

            crop_height, crop_width = person_crop.shape[:2]

            ppe_results = ppe_model.predict(source=person_crop, conf=PPE_CONFIDENCE, verbose=False)

            if not ppe_results:
                continue

            ppe_result = ppe_results[0]

            if ppe_result.boxes is not None and len(ppe_result.boxes) > 0:
                for ppe_box in ppe_result.boxes:
                    ppe_xyxy = ppe_box.xyxy[0].cpu().numpy()

                    crop_x1 = float(ppe_xyxy[0])
                    crop_y1 = float(ppe_xyxy[1])
                    crop_x2 = float(ppe_xyxy[2])
                    crop_y2 = float(ppe_xyxy[3])

                    ppe_confidence = float(ppe_box.conf[0].cpu().item())
                    ppe_class_id = int(ppe_box.cls[0].cpu().item())

                    ppe_class_name = PPE_CLASSES.get(ppe_class_id)

                    if ppe_class_name is None:
                        continue

                    original_x1 = person_x1 + crop_x1
                    original_y1 = person_y1 + crop_y1
                    original_x2 = person_x1 + crop_x2
                    original_y2 = person_y1 + crop_y2

                    original_x1 = max(0, min(original_x1, image_width))
                    original_y1 = max(0, min(original_y1, image_height))
                    original_x2 = max(0, min(original_x2, image_width))
                    original_y2 = max(0, min(original_y2, image_height))

                    if original_x2 <= original_x1 or original_y2 <= original_y1:
                        continue

                    draw_detection(image, (original_x1, original_y1, original_x2, original_y2), ppe_class_name, ppe_confidence)

            draw_detection(image, (person_x1, person_y1, person_x2, person_y2), "person", person_confidence)

        if not cv2.imwrite(str(output_path), image):
            raise IOError(f"Failed to save image: {output_path}")

    except Exception as error:
        print(f"ERROR processing {image_path.name}: {error}")


def run_pipeline():
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    person_model_path = Path(args.person_det_model)
    ppe_model_path = Path(args.ppe_detection_model)

    try:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        if not person_model_path.exists():
            raise FileNotFoundError(f"Person model not found: {person_model_path}")

        if not ppe_model_path.exists():
            raise FileNotFoundError(f"PPE model not found: {ppe_model_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading person detection model...")
        person_model = YOLO(str(person_model_path))

        print("Loading PPE detection model...")
        ppe_model = YOLO(str(ppe_model_path))

    except Exception as error:
        print(f"ERROR during initialization: {error}")
        return

    image_paths = [path for path in input_dir.iterdir() if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS]

    if not image_paths:
        print(f"No supported images found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} images.")

    successful = 0
    failed = 0

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Processing {image_path.name}")

        output_path = output_dir / image_path.name

        try:
            process_image(image_path, output_path, person_model, ppe_model)

            if output_path.exists():
                successful += 1
            else:
                failed += 1

        except Exception as error:
            failed += 1
            print(f"ERROR processing {image_path.name}: {error}")

    print("\nInference completed.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    run_pipeline()