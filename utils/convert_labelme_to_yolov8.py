import os
import json
import click

# Configurable global variable to determine which class to convert
CONVERT_CLASS = "Dolphin"


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def convert_labelme_to_yolo(input_dir, output_dir):
    """
    Convert LabelMe JSON files from INPUT_DIR to YOLOv8 format in OUTPUT_DIR.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List JSON files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for i, file_name in enumerate(files, start=1):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.txt")

        with open(input_path, 'r') as file:
            data = json.load(file)

        image_width = data['imageWidth']
        image_height = data['imageHeight']

        with open(output_path, 'w') as out_file:
            for shape in data['shapes']:
                if shape['label'] != CONVERT_CLASS:
                    continue

                # Extract points
                points = shape['points']
                group_id = shape['group_id']

                # Calculate bounding box
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)

                # Normalize coordinates
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                # Write to output
                out_file.write(f"{group_id} {x_center} {y_center} {width} {height}\n")

        # Basic progress report with carriage return
        print(f"Converted {i}/{len(files)}: {file_name}\r", end='')

    # Ensure the final line break is printed after the last file
    print()


if __name__ == "__main__":
    convert_labelme_to_yolo()