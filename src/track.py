from ultralytics import YOLO
import os
import pandas as pd
import numpy as np
import click
from tracking_metrics import TrackingMetrics


@click.command()
@click.option('--dataset', required=True, help="Path to the dataset directory.")
@click.option('--model', required=True, help="Path to the model file.")
@click.option('--output', required=True, help="Path to the output directory.")
def run_tracking_and_evaluation(dataset, model, output):
    images_directory = os.path.join(dataset, "images")
    label_directory = os.path.join(dataset, "labels")

    # Load model and determine model name
    model_instance = YOLO(model)
    model_name = os.path.basename(model).split('.')[0]

    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Construct output filenames
    gt_file = os.path.join(output, f'{model_name}_gt_formatted.txt')
    tracker_results_file = os.path.join(output, f'{model_name}_tracker_results.txt')
    evaluation_file = os.path.join(output, f'{model_name}_mota_evaluation.csv')

    # Run tracking
    results = model_instance.track(source=images_directory, tracker="bytetrack.yaml", stream=True)

    save_tracker_results(tracker_results_file, results)
    process_ground_truth(gt_file, label_directory)
    assign_group_ids_in_file(gt_file)
    
    gt_data = read_data(gt_file)
    tracker_data = read_data(tracker_results_file)

    assign_unique_ids_per_frame(gt_data)
    compute_metrics(gt_data, tracker_data, evaluation_file, gt_file, tracker_results_file)


def save_tracker_results(file_path, results):
    with open(file_path, 'w') as f:
        for frame_id, result in enumerate(results):
            for box in result.boxes:
                if box.id:
                    bbox = box.xyxyn[0].tolist()
                    track_id = int(box.id.item())
                    conf = box.conf.item()
                    f.write(
                        f'{frame_id + 1},{track_id},{bbox[0]},{bbox[1]},{bbox[2] - bbox[0]},{bbox[3] - bbox[1]},-1,-1,{conf}\n')


def read_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df.iloc[:, :6]
    df.columns = ['frame', 'id', 'x_min', 'y_min', 'x_max', 'y_max']
    df['bbox'] = df.apply(
        lambda row: (row['x_min'], row['y_min'], row['x_max'] - row['x_min'], row['y_max'] - row['y_min']), axis=1)
    return df[['frame', 'id', 'bbox']]


# Function to assign new IDs to zero entries while preserving existing non-zero IDs
def assign_unique_ids_per_frame(gt_data):
    # Group the data by 'frame'
    grouped = gt_data.groupby('frame')

    # Iterate over each group
    for frame, group in grouped:
        next_id = 1
        for index, row in group.iterrows():
            if row['id'] == 0:
                # Assign a new unique ID within the frame
                gt_data.at[index, 'id'] = next_id
                next_id += 1
            else:
                # Update `next_id` to ensure uniqueness within the frame
                next_id = max(next_id, row['id'] + 1)


def assign_group_ids_in_file(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df.iloc[:, :6]
    df.columns = ['frame', 'id', 'x_min', 'y_min', 'x_max', 'y_max']
    assign_unique_ids_per_frame(df)
    df['mot15col1'] = -1
    df['mot15col2'] = -1
    df['mot15col3'] = 1
    df.to_csv(file_path, index=False, header=False)


def process_ground_truth(gt_file_path, label_directory):

    # Combine label files
    files = [f for f in os.listdir(label_directory) if f.endswith('.txt')]
    files.sort()

    with open(gt_file_path, 'w') as outfile:
        for i, filename in enumerate(files, start=1):
            filepath = os.path.join(label_directory, filename)
            with open(filepath, 'r') as infile:
                content = infile.read()
                content = [f"{i}," + ",".join(x.split(" ")) for x in content.split("\n") if x.strip()]
                content = "\n".join(content)
                outfile.write(f"{content}\n")


def calculate_iou_shapely(box_1, box2):
    from shapely.geometry import box
    poly1 = box(*box_1)
    poly2 = box(*box2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0


def compute_metrics(gt_data, tracker_data, evaluation_file, gt_file, tracker_results_file):
    tm = TrackingMetrics()
    frames = sorted(set(gt_data['frame'].unique()).union(set(tracker_data['frame'].unique())))

    for frame in frames:
        g = gt_data[gt_data['frame'] == frame]
        t = tracker_data[tracker_data['frame'] == frame]

        gt_ids = g['id'].values
        tr_ids = t['id'].values
        gt_boxes = g['bbox'].values
        tr_boxes = t['bbox'].values

        distances = np.full((len(gt_ids), len(tr_ids)), np.inf)
        for i, gt_box in enumerate(gt_boxes):
            for j, tr_box in enumerate(tr_boxes):
                iou = calculate_iou_shapely(gt_box, tr_box)
                if iou > 0.5:
                    distances[i, j] = 1 - iou

        # Before updating the accumulator, print the current frame and IDs
        print(f"Processing frame: {frame}")
        print(f"Ground truth IDs: {gt_ids}")
        print(f"Tracker IDs: {tr_ids}")

        # Update the accumulator
        tm.update(gt_ids, tr_ids, distances)

    summary = tm.compute(metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                       'recall', 'precision', 'num_objects', \
                                       'mostly_tracked', 'partially_tracked', \
                                       'mostly_lost', 'num_false_positives', \
                                       'num_misses', 'num_switches', \
                                       'num_fragmentations', 'mota', 'motp', "id_global_assignment", \
                                       "obj_frequencies"
                                       ], outfile=evaluation_file, printsum=True, gt_path=gt_file, pred_path=tracker_results_file)

    #summary = tm.compute(metrics=["hota_alpha"], outfile=evaluation_file, printsum=True, gt_path=gt_file, pred_path=tracker_results_file)


if __name__ == '__main__':
    run_tracking_and_evaluation()