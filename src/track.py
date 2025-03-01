from ultralytics import YOLO
import motmetrics as mm
import pandas as pd
import numpy as np
import click
import re
import torch
import yaml
import cv2
from pathlib import Path
from shapely.geometry import box as shape_box

if __package__ is None or __package__ == '':
    from utils.settings import settings, project_path
    from utils.tracking_metrics import TrackingMetrics
else:
    from .utils.settings import settings, project_path
    from .utils.tracking_metrics import TrackingMetrics


@click.command()
@click.option('--dataset', required=True, help="Path to the dataset directory.")
@click.option('--model', required=True, help="Path to the model file.")
@click.option('--output', required=True, help="Path to the output directory.")
@click.option('--tracker', help="Tracker config file")
@click.option('--botsort', is_flag=True, help="Enable BotSort parameter.")
@click.option('--nopersist', is_flag=True, help="Disable persistence in tracking.")
def main(dataset, model, output, tracker, botsort, nopersist):
    results = run_tracking_and_evaluation(dataset, model, output, tracker, botsort, nopersist)


class DolphinTracker:

    def __init__(self, model_path, output_dir_path, tracker_path, botsort, nopersist):
        self.model_path = Path(model_path)
        self.output_dir_path = Path(output_dir_path)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.run_name = self.output_dir_path.name

        self.botsort = botsort
        self.nopersist = nopersist

        if tracker_path is None or tracker_path == "Default":
            self.tracker_path = project_path(settings['ultralytics_bytetrack'])
        if self.botsort:
            self.tracker_path = project_path(settings['ultralytics_botsort'])
        else:
            self.tracker_path = Path(tracker_path)

        # Check for CUDA availability
        if torch.cuda.is_available():
            print("CUDA is available.")
            self.device = '0'
        else:
            print("CUDA is not available.")
            self.device = 'cpu'

        # Load model and determine model name
        self.model_instance = YOLO(self.model_path)

        self.gt_file_path = self.output_dir_path / f'{self.run_name}_{settings["gt_file_suffix"]}'
        self.results_file_path = self.output_dir_path / f'{self.run_name}_{settings["results_file_suffix"]}'
        self.metrics_file_path = self.output_dir_path / f'{self.run_name}_{settings["metrics_file_suffix"]}'
        self.metrics_events_path = self.output_dir_path / f'{self.run_name}_{settings["metrics_events_suffix"]}'
        self.researcher_output_path = self.output_dir_path / f'{self.run_name}_{settings["researcher_output_suffix"]}'

        with open(self.tracker_path, 'r') as file:
            tracker_settings = yaml.safe_load(file)

        if 'iou' in tracker_settings:
            self.iou = tracker_settings['iou']
        else:
            self.iou = 0.7

    def track_from_images(self, image_dir_path, camera_df=None):
        results = self.model_instance.track(source=image_dir_path, tracker=self.tracker_path, device=self.device,
                                            persist=(not self.nopersist), iou=self.iou, stream=True)
        self.save_tracker_results(image_dir_path, results, camera_df)

    def save_tracker_results(self, image_dir_path, results, camera_df=None):
        files = list(image_dir_path.glob('*.jpg'))
        files.sort()
        pattern = r"(\d+)(?=[._](jpg))"

        first_image = cv2.imread(files[0])
        if first_image is None:
            raise ValueError(f"Could not read image file {files[0]}")

        img_height, img_width = first_image.shape[:2]

        with open(self.results_file_path, 'w') as f:
            if self.researcher_output_path:
                rf = open(self.researcher_output_path, 'w')
                rf.write(
                    f'FrameID,ObjectID,Point1X_px,Point1Y_px,Point2X_px,Point2Y_px,Width_px,Height_px,CenterX_px,CenterY_px')
                if camera_df is not None:
                    rf.write(
                        ',Point1X_m,Point1Y_m,Point2X_m,Point2Y_m,Width_m,Height_m,CenterX_m,CenterY_m,Altitude_m,GSD_cmpx')
                rf.write('\n')

            for i, result in enumerate(results):
                match = re.search(pattern, str(files[i]))
                frame_id = match.group(1)
                if camera_df is not None:
                    if i < len(camera_df):
                        camera_row = camera_df.iloc[i]
                    else:
                        camera_df = None
                        print(f"------------------------------------------\n"
                              f"Warning: Ran out of rows in SRT file at frame index {i + 1} (frame ID {frame_id}).\n"
                              f"Will use altitude and focal length from the last available row for the remaining frames.\n")
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
                            point_a_x_px = point_a_x * img_width
                            point_a_y_px = point_a_y * img_height
                            point_b_x_px = point_b_x * img_width
                            point_b_y_px = point_b_y * img_height
                            center_x_px = center_x * img_width
                            center_y_px = center_y * img_height
                            width_px = width * img_width
                            height_px = height * img_height
                            rf.write(
                                f'{frame_id},{track_id},{point_a_x_px},{point_a_y_px},'
                                f'{point_b_x_px},{point_b_y_px},'
                                f'{width_px},{height_px},'
                                f'{center_x_px},{center_y_px}')
                            if camera_df is not None and camera_row is not None:
                                gsd_mpx = camera_row['GSD_cmpx'] / 100
                                rf.write(
                                    f',{point_a_x_px * gsd_mpx},{point_a_y_px * gsd_mpx},'
                                    f'{point_b_x_px * gsd_mpx},{point_b_y_px * gsd_mpx},'
                                    f'{width_px * gsd_mpx},{height_px * gsd_mpx},'
                                    f'{center_x_px * gsd_mpx},{center_y_px * gsd_mpx},'
                                    f'{camera_row["est_alt_m"]},{camera_row["GSD_cmpx"]}')
                            rf.write('\n')

            if rf:
                rf.close()

    def save_ground_truth(self, label_dir_path):
        files = list(label_dir_path.glob('*.txt'))
        files.sort()

        data = []

        # Read and process each file
        for label_path in files:
            pattern = r"(\d+)(?=[._](txt))"
            match = re.search(pattern, str(label_path))
            if not match:
                raise ValueError(f"Could not process filename {label_path}")

            frame_id = match.group(1)

            with open(label_path, 'r') as f:
                content = f.read()
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
        gt_df = pd.DataFrame(data, columns=['frame', 'id', 'x_min', 'y_min', 'x_max', 'y_max'])
        gt_df = gt_df.astype(
            {'frame': int, 'id': 'Int64', 'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float})

        gt_df['mot15col1'] = -1
        gt_df['mot15col2'] = -1
        gt_df['mot15col3'] = 1
        gt_df.to_csv(self.gt_file_path, index=False, header=False)

    def _iou(self, box1, box2):
        poly1 = shape_box(*box1)
        poly2 = shape_box(*box2)
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union != 0 else 0

    def compute_metrics(self, gt_df, pred_df):
        tm = TrackingMetrics()

        metrics = settings['tracking_metrics']

        frames = sorted(set(gt_df.index.get_level_values('FrameId')).intersection(
            set(pred_df.index.get_level_values('FrameId'))))

        for frame in frames:
            g = gt_df.loc[frame]
            p = pred_df.loc[frame]

            g.dropna(inplace=True)

            gt_ids = g.index.get_level_values('Id').values
            pr_ids = p.index.get_level_values('Id').values

            gt_boxes = [(row['X'], row['Y'], row['X'] + row['Width'], row['Y'] + row['Height']) for index, row in
                        g.iterrows()]
            tr_boxes = [(row['X'], row['Y'], row['X'] + row['Width'], row['Y'] + row['Height']) for index, row in
                        p.iterrows()]

            distances = np.full((len(gt_ids), len(pr_ids)), np.inf)
            for i, gt_box in enumerate(gt_boxes):
                for j, tr_box in enumerate(tr_boxes):
                    iou = self._iou(gt_box, tr_box)
                    #TODO: Is this the same as match_threshold? How does it relate to other thresholds? Should it be hp?
                    if iou > 0.5:
                        distances[i, j] = 1 - iou

            # Before updating the accumulator, print the current frame and IDs
            print(f"Processing frame: {frame} Ground truth IDs: {gt_ids} Tracker IDs: {pr_ids}")

            # Update the accumulator
            tm.update(gt_ids, pr_ids, distances, frame)

        summary = tm.compute(metrics=metrics, outfile=self.metrics_file_path, printsum=True)

        tm.write_events(self.metrics_events_path)

        print(f"\nResults written to {self.metrics_file_path}")

        return summary

    def evaluate(self, label_dir_path):
        if label_dir_path.is_dir():
            self.save_ground_truth(label_dir_path)

            gt_df = mm.io.loadtxt(self.gt_file_path)
            pred_df = mm.io.loadtxt(self.results_file_path)

            metrics = self.compute_metrics(gt_df, pred_df)
            return metrics
        else:
            print(
                f"No ground truth label directory found; looked for {label_dir_path}. Not running metrics calculations.")


def run_tracking_and_evaluation(dataset_path, model_path, output_dir_path, tracker_path, botsort=False,
                                nopersist=False, camera_df=None):
    print(f"Loading configuration files...")

    dataset_path = Path(dataset_path)

    image_dir_path = dataset_path / settings['images_dir']
    label_dir_path = dataset_path / settings['labels_dir']

    tracker = DolphinTracker(model_path, output_dir_path, tracker_path, botsort, nopersist)
    tracker.track_from_images(image_dir_path, camera_df)
    metrics = tracker.evaluate(label_dir_path)

    return metrics


if __name__ == '__main__':
    main()
