from ultralytics import YOLO
import motmetrics as mm
import os
import pandas as pd
import numpy as np
import click
import re
import torch
from tracking_metrics import TrackingMetrics
import yaml
import cv2
import math


@click.command()
@click.option('--dataset', required=True, help="Path to the dataset directory.")
@click.option('--model', required=True, help="Path to the model file.")
@click.option('--output', required=True, help="Path to the output directory.")
@click.option('--tracker', help="Tracker config file")
@click.option('--botsort', is_flag=True, help="Enable BotSort parameter.")
@click.option('--nopersist', is_flag=True, help="Disable persistence in tracking.")
def main(dataset, model, output, tracker, botsort, nopersist):
    results = run_tracking_and_evaluation(dataset, model, output, tracker, botsort, nopersist)

def run_tracking_and_evaluation(dataset, model, output, tracker, botsort=False, nopersist=False):
    print(f"Loading configuration files...")
    config = 'dolphin_tracker/cfg/settings.yaml'

    with open(config, 'r') as file:
        settings = yaml.safe_load(file)

    images_directory = os.path.join(dataset, "images")
    label_directory = os.path.join(dataset, "labels")

    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available.")
        device = '0'
    else:
        print("CUDA is not available.")
        device = 'cpu'

    # Load model and determine model name
    model_instance = YOLO(model)
    model_name = os.path.basename(model).split('.')[0]

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Extract the closest level directory name from the output path
    run_name = os.path.basename(os.path.normpath(output))

    # Construct output filenames using the closest level directory name
    gt_file_path = os.path.join(output, f'{run_name}_{settings["gt_file_suffix"]}')
    results_file_path = os.path.join(output, f'{run_name}_{settings["results_file_suffix"]}')
    metrics_file_path = os.path.join(output, f'{run_name}_{settings["metrics_file_suffix"]}')
    metrics_events_path = os.path.join(output, f'{run_name}_{settings["metrics_events_suffix"]}')
    researcher_output_path = os.path.join(output, f'{run_name}_{settings["researcher_output_suffix"]}')

    # Run tracking
    tracker = tracker or settings['ultralytics_bytetrack']
    if tracker == "Default":
        tracker = settings['ultralytics_bytetrack']
    if botsort:
        tracker = settings['ultralytics_botsort']

    with open(tracker, 'r') as file:
        tracker_settings = yaml.safe_load(file)

    if 'iou' in tracker_settings:
        iou = tracker_settings['iou']
    else:
        iou = 0.7

    files = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if f.endswith('.jpg')]
    files.sort()

    results = model_instance.track(source=images_directory, tracker=tracker, stream=True, device=device,
                                   persist=(not nopersist), iou=iou)
    # results = []
    # pattern = r"(.*?)_?(\d+)(?=[._](jpg))"
    # prefix = None
    # frame = None
    # empty_frames = 0
    # for file in files:
    #     if prefix is None:
    #         match = re.search(pattern, file)
    #         if match:
    #             prefix = match.group(1)
    #             frame = int(match.group(2)) - 1
    #
    #     match = re.search(pattern, file)
    #     frame += 1
    #
    #     if match.group(1) != prefix or int(match.group(2)) != frame:
    #         prefix = match.group(1)
    #         frame = int(match.group(2))
    #         model_instance.predictor.trackers[0].reset()
    #         print(f"Resetting tracker at frame {frame}")
    #
    #     result = model_instance.track(source=file, tracker=tracker, stream=True, device=device,
    #                                    persist=(not nopersist))
    #     objects = len(result[0].boxes)
    #     if objects == 0:
    #         empty_frames += 1
    #         if empty_frames == settings['tracker_empty_frames_reset']:
    #             print(f"{empty_frames} empty frames detected at {frame}. Resetting tracker.")
    #             model_instance.predictor.trackers[0].reset()
    #     else:
    #         empty_frames = 0
    #
    #     results.extend(result)

    save_tracker_results(images_directory, results_file_path, results, researcher_outpath=researcher_output_path)
    if os.path.exists(label_directory):
        save_ground_truth(gt_file_path, label_directory)

        df_gt = mm.io.loadtxt(gt_file_path)
        df_pred = mm.io.loadtxt(results_file_path)

        metrics = compute_metrics(metrics_file_path, metrics_events_path, df_gt, df_pred)
        return metrics
    else:
        print(f"No ground truth label directory found; looked for {label_directory}. Not running metrics calculations.")


def save_tracker_results(images_directory, file_path, results, researcher_outpath=None):
    files = [f for f in os.listdir(images_directory) if f.endswith('.jpg')]
    files.sort()
    pattern = r"(\d+)(?=[._](jpg))"

    first_image_path = os.path.join(images_directory, files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise ValueError(f"Could not read image file {first_image_path}")

    img_height, img_width = cv2.imread(first_image_path).shape[:2]

    with open(file_path, 'w') as f:
        if researcher_outpath:
            rf = open(researcher_outpath, 'w')
            rf.write(f'FrameID,ObjectID,Point1X,Point1Y,Point2X,Point2Y,Width,Height,CenterX,CenterY\n')

        for i, result in enumerate(results):
            match = re.search(pattern, files[i])
            frame_id = match.group(1)
            for box in result.boxes:
                if box.id:
                    bbox = box.xyxyn[0].tolist()
                    track_id = int(box.id.item())
                    conf = box.conf.item()
                    point_a_x, point_a_y, point_b_x, point_b_y = bbox
                    width = point_b_x - point_a_x
                    height = point_b_y - point_a_y
                    center_x = (point_a_x + point_b_x) / 2
                    center_y = (point_a_y + point_b_y) / 2
                    f.write(
                        f'{frame_id},{track_id},{center_x},{center_y},{width},{height},-1,-1,{conf}\n')
                    if rf:
                        rf.write(
                            f'{frame_id},{track_id},{point_a_x * img_width},{point_a_y * img_height},'
                            f'{point_b_x * img_width},{point_b_y * img_height},'
                            f'{width * img_width},{height * img_height},'
                            f'{center_x * img_width},{center_y * img_height}\n')

        if rf:
            rf.close()


def save_ground_truth(gt_file_path, label_directory):
    files = [f for f in os.listdir(label_directory) if f.endswith('.txt')]
    files.sort()

    data = []

    # Read and process each file
    for filename in files:
        pattern = r"(\d+)(?=[._](jpg|txt|json))"
        match = re.search(pattern, filename)
        if not match:
            raise ValueError(f"Could not process filename {label_directory}/{filename}")

        frame_id = match.group(1)

        filepath = os.path.join(label_directory, filename)
        with open(filepath, 'r') as infile:
            content = infile.read()
            next_id = 1
            empty_frame = True
            for line in content.split("\n"):
                if line.strip():
                    # Add frame number as the first element
                    line_data = line.split(" ")
                    # Replace any box IDs of 0
                    if line_data[0] == '0' or line_data[0] == 'None':
                        line_data[0] = str(next_id)
                        next_id += 1
                    data.append([frame_id] + line_data)
                    empty_frame = False
            if empty_frame:
                data.append([frame_id, None, None, None, None, None])

    # Convert list to DataFrame
    gt_data = pd.DataFrame(data, columns=['frame', 'id', 'x_min', 'y_min', 'x_max', 'y_max'])
    gt_data = gt_data.astype({'frame': int, 'id': 'Int64', 'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float})

    gt_data['mot15col1'] = -1
    gt_data['mot15col2'] = -1
    gt_data['mot15col3'] = 1
    gt_data.to_csv(gt_file_path, index=False, header=False)


def calculate_iou_shapely(box1, box2):
    from shapely.geometry import box
    poly1 = box(*box1)
    poly2 = box(*box2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0


def compute_metrics(evaluation_file, events_file, df_gt, df_pred):
    tm = TrackingMetrics()

    with open('dolphin_tracker/cfg/settings.yaml', 'r') as file:
        settings = yaml.safe_load(file)

    metrics = settings['tracking_metrics']

    frames = sorted(set(df_gt.index.get_level_values('FrameId')).intersection(
        set(df_pred.index.get_level_values('FrameId'))))

    for frame in frames:
        g = df_gt.loc[frame]
        t = df_pred.loc[frame]

        g.dropna(inplace=True)

        gt_ids = g.index.get_level_values('Id').values
        tr_ids = t.index.get_level_values('Id').values

        #print(f"Frame: {frame} IDs: GT: {gt_ids}, Pred: {tr_ids}")

        #gt_boxes = [(row['X'], row['Y'], row['Width'], row['Height']) for index, row in g.iterrows()]
        #tr_boxes = [(row['X'], row['Y'], row['Width'], row['Height']) for index, row in t.iterrows()]

        gt_boxes = [(row['X'], row['Y'], row['X'] + row['Width'], row['Y'] + row['Height']) for index, row in g.iterrows()]
        tr_boxes = [(row['X'], row['Y'], row['X'] + row['Width'], row['Y'] + row['Height']) for index, row in t.iterrows()]

        #print(f"GT Boxes: {gt_boxes}\n PR boxes: {tr_boxes}")

        distances = np.full((len(gt_ids), len(tr_ids)), np.inf)
        for i, gt_box in enumerate(gt_boxes):
            for j, tr_box in enumerate(tr_boxes):
                #print(f"Comparing gt box {i}:{gt_box} to pr box {j}:{tr_box}")
                iou = calculate_iou_shapely(gt_box, tr_box)
                #print(f"IOU: {iou}")
                if iou > 0.5:
                    distances[i, j] = 1 - iou

        # Before updating the accumulator, print the current frame and IDs
        print(f"Processing frame: {frame} Ground truth IDs: {gt_ids} Tracker IDs: {tr_ids}")

        # Update the accumulator
        tm.update(gt_ids, tr_ids, distances, frame)
    #tm.print_events()
    summary = tm.compute(metrics=metrics, outfile=evaluation_file, printsum=True, df_gt=df_gt, df_pred=df_pred)

    tm.write_events(events_file)

    print(f"\nResults written to {evaluation_file}")

    return summary

if __name__ == '__main__':
    main()