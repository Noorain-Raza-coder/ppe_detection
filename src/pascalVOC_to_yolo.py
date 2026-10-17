import argparse
import os
import shutil
import xml.etree.ElementTree as ET


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC annotations to YOLO format.")

    parser.add_argument("input_dir", type=str,
                        help="Path to the base input dataset directory containing images and annotations.")

    parser.add_argument("output_dir", type=str,
                        help="Path to the output directory where YOLO annotations will be saved.")

    return parser.parse_args()


def load_classes(classes_file):
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    return {class_name: index for index, class_name in enumerate(classes)}


def convert_bbox_to_yolo(image_width, image_height, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    x_center /= image_width
    y_center /= image_height
    bbox_width /= image_width
    bbox_height /= image_height

    return x_center, y_center, bbox_width, bbox_height


def convert_xml_to_yolo(xml_path, class_mapping):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()

        if class_name not in class_mapping:
            print(
                f"Warning: Unknown class '{class_name}' "
                f"in {os.path.basename(xml_path)}. Skipping."
            )
            continue

        class_id = class_mapping[class_name]

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Ensure coordinates are within image boundaries
        xmin = max(0, min(xmin, image_width))
        ymin = max(0, min(ymin, image_height))
        xmax = max(0, min(xmax, image_width))
        ymax = max(0, min(ymax, image_height))

        # Ignore invalid bounding boxes
        if xmax <= xmin or ymax <= ymin:
            print(
                f"Warning: Invalid bounding box in "
                f"{os.path.basename(xml_path)}. Skipping.")
            continue

        x_center, y_center, width, height = convert_bbox_to_yolo(
            image_width,
            image_height,
            xmin,
            ymin,
            xmax,
            ymax,)

        yolo_annotations.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}")

    return yolo_annotations


def main():
    args = parse_arguments()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    annotations_dir = os.path.join(input_dir, "labels")
    images_dir = os.path.join(input_dir, "images")
    classes_file = os.path.join(input_dir, "classes.txt")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(
            f"Annotations directory not found: {annotations_dir}"
        )

    if not os.path.isfile(classes_file):
        raise FileNotFoundError(
            f"classes.txt not found: {classes_file}"
        )

    class_mapping = load_classes(classes_file)

    print("Class mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_id}: {class_name}")

    os.makedirs(output_dir, exist_ok=True)

    output_annotations_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_annotations_dir, exist_ok=True)

    # Convert every Pascal VOC XML file
    xml_files = [
        file_name
        for file_name in os.listdir(annotations_dir)
        if file_name.lower().endswith(".xml")
    ]

    if not xml_files:
        raise RuntimeError(f"No XML annotation files found in {annotations_dir}")

    successful = 0
    failed = 0

    for xml_file in xml_files:
        xml_path = os.path.join(annotations_dir, xml_file)

        txt_file = os.path.splitext(xml_file)[0] + ".txt"
        txt_path = os.path.join(output_annotations_dir, txt_file)

        try:
            yolo_annotations = convert_xml_to_yolo(xml_path, class_mapping)

            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_annotations))

            successful += 1

        except Exception as e:
            failed += 1
            print(f"Error processing {xml_file}: {e}")

    # Copy classes.txt to output directory
    shutil.copy2(classes_file, os.path.join(output_dir, "classes.txt"))

    print("\nConversion completed.")
    print(f"Total XML files : {len(xml_files)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")
    print(f"Output labels   : {output_annotations_dir}")


if __name__ == "__main__":
    main()