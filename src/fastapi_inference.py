import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO
import os


from config import (
    PPE_MODEL_PATH,
    PERSON_MODEL_PATH,
    PERSON_CONFIDENCE,
    PPE_CONFIDENCE,
    PPE_CLASSES,
)







def format_detection_results(
    person_result,
    ppe_results_by_person,
    image_shape,):

    """
    Convert person and PPE model results into a JSON-compatible dictionary.

    Args:
        person_result: First result from the person model.
        ppe_results_by_person: List of PPE model results, aligned with
                               the valid person detections.
        image_shape: Shape of the original image (height, width, ...).

    Returns:
        A JSON-compatible dictionary containing persons and PPE detections.
    """
    image_height, image_width = image_shape[:2]

    persons = {}
    person_index = 0

    if person_result.boxes is None:
        return {
                "total_persons": 0,
                "persons": {}
                }

    ppe_result_index = 0

    for person_box in person_result.boxes:
        person_xyxy = person_box.xyxy[0].cpu().numpy()

        person_x1 = max(0, min(int(person_xyxy[0]), image_width))
        person_y1 = max(0, min(int(person_xyxy[1]), image_height))
        person_x2 = max(0, min(int(person_xyxy[2]), image_width))
        person_y2 = max(0, min(int(person_xyxy[3]), image_height))

        # Skip invalid person bounding boxes.
        if person_x2 <= person_x1 or person_y2 <= person_y1:
            continue

        person_confidence = float(person_box.conf[0].cpu().item())

        # Each valid person detection has one corresponding PPE result.
        ppe_result = (
            ppe_results_by_person[ppe_result_index]
            if ppe_result_index < len(ppe_results_by_person)
            else None
        )
        ppe_result_index += 1

        ppe_detections = []

        if ( ppe_result is not None and ppe_result.boxes is not None ):
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

                # Convert crop-relative coordinates to original-image
                # coordinates.
                original_x1 = max(0, min(person_x1 + crop_x1, image_width))
                original_y1 = max(0, min(person_y1 + crop_y1, image_height))
                original_x2 = max(0, min(person_x1 + crop_x2, image_width))
                original_y2 = max(0, min(person_y1 + crop_y2, image_height))

                if (original_x2 <= original_x1 or original_y2 <= original_y1):
                    continue

                ppe_detections.append({
                                        "class": ppe_class_name,
                                        "confidence": ppe_confidence,
                                        "bbox": {
                                            "x1": round(original_x1, 2),
                                            "y1": round(original_y1, 2),
                                            "x2": round(original_x2, 2),
                                            "y2": round(original_y2, 2),
                                                }
                                    })

        person_index += 1
        person_key = f"person_{person_index}"

        persons[person_key] = {
                                "class": "person",
                                "confidence": person_confidence,
                                "bbox": {
                                    "x1": person_x1,
                                    "y1": person_y1,
                                    "x2": person_x2,
                                    "y2": person_y2,
                                        },
                                "ppe": {
                                    "count": len(ppe_detections),
                                    "detections": ppe_detections,
                                        }
                            }

    return {
            "total_persons": len(persons),
            "persons": persons,
            }







def process_image(image_arr, person_model, ppe_model):
    try:
        if image_arr is None:
            raise ValueError("Unable to read image")

        image_height, image_width = image_arr.shape[:2]

        person_results = person_model.predict(
            source=image_arr,
            conf=PERSON_CONFIDENCE,
            verbose=False,
        )

        if not person_results:
            return {
                    "total_persons": 0,
                    "persons": {}
                    }

        person_result = person_results[0]
        ppe_results_by_person = []

        if person_result.boxes is None:
            return {
                    "total_persons": 0,
                    "persons": {}
                    }

        for person_box in person_result.boxes:
            person_xyxy = person_box.xyxy[0].cpu().numpy()

            person_x1 = max(0, min(int(person_xyxy[0]), image_width))
            person_y1 = max(0, min(int(person_xyxy[1]), image_height))
            person_x2 = max(0, min(int(person_xyxy[2]), image_width))
            person_y2 = max(0, min(int(person_xyxy[3]), image_height))

            # Keep the result list aligned with the original
            # person detections, including invalid detections.
            if person_x2 <= person_x1 or person_y2 <= person_y1:
                ppe_results_by_person.append(None)
                continue

            person_crop = image_arr[person_y1:person_y2, person_x1:person_x2]

            if person_crop.size == 0:
                ppe_results_by_person.append(None)
                continue

            ppe_results = ppe_model.predict(
                            source=person_crop,
                            conf=PPE_CONFIDENCE,
                            verbose=False,)

            ppe_result = (ppe_results[0] if ppe_results else None)
            ppe_results_by_person.append(ppe_result)

        # Format the inference results into the final JSON structure.
        final_results = format_detection_results(
                        person_result=person_result,
                        ppe_results_by_person=ppe_results_by_person,
                        image_shape=image_arr.shape,)

        return final_results

    except Exception as error:
        print(f"ERROR processing: {error}")
        raise









def run_pipeline(image_arr):

    person_model_path = Path(PERSON_MODEL_PATH)
    ppe_model_path = Path(PPE_MODEL_PATH)

    try:

        if not person_model_path.exists():
            raise FileNotFoundError(f"Person model not found: {person_model_path}")

        if not ppe_model_path.exists():
            raise FileNotFoundError(f"PPE model not found: {ppe_model_path}")

        print("Loading person detection model...")
        person_model = YOLO(str(person_model_path))

        print("Loading PPE detection model...")
        ppe_model = YOLO(str(ppe_model_path))

    except Exception as error:
        print(f"ERROR during initialization: {error}")
        return

    
    try:
        print("Processing image...")
        final_results = process_image(image_arr, person_model, ppe_model)
        return final_results

    except Exception as error:
        # print(f"ERROR processing : {error}")
        return {f"ERROR processing : {error}"}

    print("\nInference completed.")
