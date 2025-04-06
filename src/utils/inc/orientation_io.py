from PIL import Image
from pathlib import Path

def crop_image_with_bbox(image_path, bbox, bbox_format):
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if bbox_format == 'yolo':
        x_center, y_center, box_width, box_height = bbox
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x_top_left = x_center - box_width / 2
        y_top_left = y_center - box_height / 2
        x_bottom_right = x_center + box_width / 2
        y_bottom_right = y_center + box_height / 2
    elif bbox_format == 'pixels':
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = bbox
    else:
        raise ValueError("bbox_format must be either 'yolo' or 'pixels'")

    cropped_image = image.crop((x_top_left, y_top_left, x_bottom_right, y_bottom_right))
    return cropped_image