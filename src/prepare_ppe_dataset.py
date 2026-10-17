import argparse
import os
import shutil
from pathlib import Path

import cv2


# Original Roboflow class IDs
PERSON_CLASS_ID = 4

PPE_CLASS_IDS = {
    0: "boots",
    1: "gloves",
    2: "hard-hat",
    3: "mask",
    5: "ppe-suit",
    6: "vest",
}

# New PPE model class mapping
# person is removed and PPE classes are remapped to 0-5
PPE_CLASS_REMAP = {
    0: 0,  # boots
    1: 1,  # gloves
    2: 2,  # hard-hat
    3: 3,  # mask
    5: 4,  # ppe-suit
    6: 5,  # vest
}

PPE_CLASS_NAMES = [
    "boots",
    "gloves",
    "hard-hat",
    "mask",
    "ppe-suit",
    "vest",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate cropped-person dataset for PPE detection."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the original YOLO dataset."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path where the PPE dataset will be created."
    )

    return parser.parse_args()


def load_yolo_annotations(label_path):
    """
    Read YOLO annotation file.

    Returns:
        List of dictionaries containing class ID and normalized bbox.
    """

    annotations = []

    with open(label_path, "r") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                print(
                    f"Warning: Invalid annotation in "
                    f"{label_path}, line {line_number}"
                )
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            annotations.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            })

    return annotations


def yolo_to_xyxy(annotation, image_width, image_height):
    """
    Convert normalized YOLO bbox to absolute pixel coordinates.

    Returns:
        xmin, ymin, xmax, ymax
    """

    x_center = annotation["x_center"] * image_width
    y_center = annotation["y_center"] * image_height

    width = annotation["width"] * image_width
    height = annotation["height"] * image_height

    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2

    return xmin, ymin, xmax, ymax


def xyxy_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    """
    Convert absolute pixel bbox to normalized YOLO format.
    """

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    width = xmax - xmin
    height = ymax - ymin

    x_center /= image_width
    y_center /= image_height

    width /= image_width
    height /= image_height

    return x_center, y_center, width, height


def bbox_center(xmin, ymin, xmax, ymax):
    """
    Return the center point of a bounding box.
    """

    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    return center_x, center_y


def point_inside_bbox(point_x, point_y, bbox):
    """
    Check whether a point lies inside a bounding box.
    """
    xmin, ymin, xmax, ymax = bbox
    return ( xmin <= point_x <= xmax and ymin <= point_y <= ymax )


def find_person_for_ppe(ppe_bbox, person_bboxes):
    """
    Assign a PPE object to a person based on whether
    the PPE bounding-box center lies inside the person's bbox.

    Returns:
        person index or None
    """

    ppe_center_x, ppe_center_y = bbox_center(*ppe_bbox)

    candidate_persons = []

    for person_index, person_bbox in enumerate(person_bboxes):

        if point_inside_bbox(
            ppe_center_x,
            ppe_center_y,
            person_bbox
        ):
            candidate_persons.append(person_index)

    if not candidate_persons:
        return None

    # If multiple persons contain the PPE center,
    # select the person with the smallest bounding box.
    if len(candidate_persons) > 1:

        candidate_persons.sort(
            key=lambda index: (
                (person_bboxes[index][2] - person_bboxes[index][0])
                *
                (person_bboxes[index][3] - person_bboxes[index][1])
            )
        )

    return candidate_persons[0]


def transform_ppe_bbox_to_crop(
    ppe_bbox,
    person_bbox,
    crop_width,
    crop_height
):
    """
    Convert PPE bbox from full-image coordinates
    to normalized YOLO coordinates relative to the
    person crop.
    """

    person_xmin, person_ymin, _, _ = person_bbox

    ppe_xmin, ppe_ymin, ppe_xmax, ppe_ymax = ppe_bbox

    # Shift coordinates relative to crop origin
    crop_xmin = ppe_xmin - person_xmin
    crop_ymin = ppe_ymin - person_ymin
    crop_xmax = ppe_xmax - person_xmin
    crop_ymax = ppe_ymax - person_ymin

    # Clip to crop boundaries
    crop_xmin = max(0, min(crop_xmin, crop_width))
    crop_ymin = max(0, min(crop_ymin, crop_height))
    crop_xmax = max(0, min(crop_xmax, crop_width))
    crop_ymax = max(0, min(crop_ymax, crop_height))

    # Invalid after clipping
    if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
        return None

    return xyxy_to_yolo(
        crop_xmin,
        crop_ymin,
        crop_xmax,
        crop_ymax,
        crop_width,
        crop_height
    )


def process_split(input_dir, output_dir, split):
    """
    Process one dataset split.
    """

    input_images_dir = Path(input_dir) / split / "images"
    input_labels_dir = Path(input_dir) / split / "labels"

    output_images_dir = Path(output_dir) / split / "images"
    output_labels_dir = Path(output_dir) / split / "labels"

    output_images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not input_images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {input_images_dir}"
        )

    if not input_labels_dir.exists():
        raise FileNotFoundError(
            f"Labels directory not found: {input_labels_dir}"
        )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG"
    }

    image_files = [
        file
        for file in input_images_dir.iterdir()
        if file.suffix in image_extensions
    ]

    total_persons = 0
    total_ppe_annotations = 0
    total_crops = 0

    for image_path in image_files:

        label_path = input_labels_dir / (
            image_path.stem + ".txt"
        )

        if not label_path.exists():
            print(
                f"Warning: Annotation not found for "
                f"{image_path.name}"
            )
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Warning: Could not read image "
                f"{image_path.name}"
            )
            continue

        image_height, image_width = image.shape[:2]

        annotations = load_yolo_annotations(
            label_path
        )

        person_bboxes = []
        ppe_objects = []

        # Convert every annotation into pixel coordinates
        for annotation in annotations:

            bbox = yolo_to_xyxy(
                annotation,
                image_width,
                image_height
            )

            if annotation["class_id"] == PERSON_CLASS_ID:

                person_bboxes.append(bbox)

            elif annotation["class_id"] in PPE_CLASS_REMAP:

                ppe_objects.append({
                    "class_id": annotation["class_id"],
                    "bbox": bbox
                })

        if not person_bboxes:
            continue

        total_persons += len(person_bboxes)

        # Generate one crop per person
        for person_index, person_bbox in enumerate(person_bboxes):

            person_xmin, person_ymin, person_xmax, person_ymax = (
                person_bbox
            )

            # Clip person bbox to image boundaries
            person_xmin = max(
                0,
                min(person_xmin, image_width)
            )

            person_ymin = max(
                0,
                min(person_ymin, image_height)
            )

            person_xmax = max(
                0,
                min(person_xmax, image_width)
            )

            person_ymax = max(
                0,
                min(person_ymax, image_height)
            )

            person_xmin = int(round(person_xmin))
            person_ymin = int(round(person_ymin))
            person_xmax = int(round(person_xmax))
            person_ymax = int(round(person_ymax))

            crop_width = person_xmax - person_xmin
            crop_height = person_ymax - person_ymin

            if crop_width <= 0 or crop_height <= 0:
                print(
                    f"Warning: Invalid person bbox in "
                    f"{image_path.name}"
                )
                continue

            crop = image[
                person_ymin:person_ymax,
                person_xmin:person_xmax
            ]

            if crop.size == 0:
                continue

            crop_annotations = []

            # Find PPE belonging to this person
            for ppe_object in ppe_objects:

                assigned_person = find_person_for_ppe(
                    ppe_object["bbox"],
                    person_bboxes
                )

                if assigned_person != person_index:
                    continue

                transformed_bbox = transform_ppe_bbox_to_crop(
                    ppe_object["bbox"],
                    person_bbox,
                    crop_width,
                    crop_height
                )

                if transformed_bbox is None:
                    continue

                new_class_id = PPE_CLASS_REMAP[
                    ppe_object["class_id"]
                ]

                x_center, y_center, width, height = (
                    transformed_bbox
                )

                crop_annotations.append(
                    f"{new_class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

            total_ppe_annotations += len(crop_annotations)

            # -------------------------------------------------
            # Save crop
            # -------------------------------------------------

            crop_name = (
                f"{image_path.stem}_person_{person_index}"
                f"{image_path.suffix}"
            )

            crop_label_name = (
                f"{image_path.stem}_person_{person_index}.txt"
            )

            output_image_path = (
                output_images_dir / crop_name
            )

            output_label_path = (
                output_labels_dir / crop_label_name
            )

            cv2.imwrite(
                str(output_image_path),
                crop
            )

            # Save annotations.
            # Empty labels are allowed because a person may
            # have no visible PPE.
            with open(output_label_path, "w") as file:

                if crop_annotations:
                    file.write(
                        "\n".join(crop_annotations)
                    )

            total_crops += 1

    print(f"\n{split.upper()} split")
    print("-" * 40)
    print(f"Person instances : {total_persons}")
    print(f"Person crops     : {total_crops}")
    print(f"PPE annotations  : {total_ppe_annotations}")


def create_data_yaml(output_dir):
    """
    Create data.yaml for PPE model.
    """

    yaml_content = """path: .

    train: train/images
    val: valid/images

    names:
    - boots
    - gloves
    - hard-hat
    - mask
    - ppe-suit
    - vest
    """

    yaml_path = Path(output_dir) / "data.yaml"

    with open(yaml_path, "w") as file:
        file.write(yaml_content)

    print(f"\nCreated: {yaml_path}")


def main():

    args = parse_arguments()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input dataset does not exist: {input_dir}"
        )

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating PPE cropped-person dataset")
    print("=" * 60)

    print("\nOriginal class mapping:")
    print("0 -> boots")
    print("1 -> gloves")
    print("2 -> hard-hat")
    print("3 -> mask")
    print("4 -> person")
    print("5 -> ppe-suit")
    print("6 -> vest")

    print("\nPPE model mapping:")
    for class_id, class_name in enumerate(PPE_CLASS_NAMES):
        print(f"{class_id} -> {class_name}")

    process_split(
        input_dir,
        output_dir,
        "train"
    )

    process_split(
        input_dir,
        output_dir,
        "valid"
    )

    create_data_yaml(output_dir)

    print("\n" + "=" * 60)
    print("PPE dataset generation completed.")
    print(f"Output dataset: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()