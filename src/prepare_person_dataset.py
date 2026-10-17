import argparse
import os
import shutil


# Class mapping from the current Roboflow data.yaml
PERSON_CLASS_ID = 4

# Person becomes class 0 in the new person-only dataset
NEW_PERSON_CLASS_ID = 0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a person-only YOLO dataset from a multi-class YOLO dataset."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the original Roboflow YOLO dataset."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path where the person-only dataset will be created."
    )

    return parser.parse_args()


def process_split(input_dir, output_dir, split):
    input_images_dir = os.path.join(input_dir, split, "images")
    input_labels_dir = os.path.join(input_dir, split, "labels")

    output_images_dir = os.path.join(output_dir, split, "images")
    output_labels_dir = os.path.join(output_dir, split, "labels")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    if not os.path.isdir(input_labels_dir):
        raise FileNotFoundError(
            f"Labels directory not found: {input_labels_dir}"
        )

    if not os.path.isdir(input_images_dir):
        raise FileNotFoundError(
            f"Images directory not found: {input_images_dir}"
        )

    label_files = [
        file_name
        for file_name in os.listdir(input_labels_dir)
        if file_name.lower().endswith(".txt")
    ]

    for label_file in label_files:

        input_label_path = os.path.join(
            input_labels_dir,
            label_file
        )

        output_label_path = os.path.join(
            output_labels_dir,
            label_file
        )

        person_annotations = []

        with open(input_label_path, "r") as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) != 5:
                    print(
                        f"Warning: Invalid annotation in {label_file}: {line}"
                    )
                    continue

                class_id = int(values[0])

                # Keep only person annotations
                if class_id == PERSON_CLASS_ID:

                    # Change person class ID from 4 → 0
                    values[0] = str(NEW_PERSON_CLASS_ID)

                    person_annotations.append(
                        " ".join(values)
                    )

        # Skip images without a person annotation
        if not person_annotations:
            continue

        # Save person-only annotation
        with open(output_label_path, "w") as file:
            file.write("\n".join(person_annotations))

        # Find corresponding image
        image_base_name = os.path.splitext(label_file)[0]

        possible_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG"
        ]

        image_found = False

        for extension in possible_extensions:

            image_name = image_base_name + extension

            input_image_path = os.path.join(
                input_images_dir,
                image_name
            )

            if os.path.exists(input_image_path):

                output_image_path = os.path.join(
                    output_images_dir,
                    image_name
                )

                shutil.copy2(
                    input_image_path,
                    output_image_path
                )

                image_found = True
                break

        if not image_found:
            print(
                f"Warning: Image not found for {label_file}"
            )


def create_data_yaml(output_dir):

    yaml_content = """path: .

                    train: train/images
                    val: valid/images

                    names:
                    - person
                    """

    yaml_path = os.path.join(
        output_dir,
        "data.yaml"
    )

    with open(yaml_path, "w") as file:
        file.write(yaml_content)


def main():

    args = parse_arguments()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input dataset does not exist: {input_dir}"
        )

    os.makedirs(output_dir, exist_ok=True)

    print("Creating person-only dataset...")
    print(f"Original person class ID : {PERSON_CLASS_ID}")
    print(f"New person class ID      : {NEW_PERSON_CLASS_ID}")

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

    print("\nPerson dataset created successfully.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()