from ultralytics import YOLO
import motmetrics as mm
import pandas as pd
import numpy as np
import re
import torch
import yaml
import cv2
from pathlib import Path
from shapely.geometry import box as shape_box
import argparse

if __package__ is None or __package__ == '':
    from utils.inc.settings import settings, project_path
    from utils.inc.reporting import TrackingMetrics
    from utils.inc.reporting import DataAccumulator
    from utils.inc.oriented_bounding_boxes import rotate_points
else:
    from .utils.inc.settings import settings, project_path
    from .utils.inc.reporting import TrackingMetrics
    from .utils.inc.reporting import DataAccumulator
    from .utils.inc.oriented_bounding_boxes import rotate_points


def run_tracking_and_evaluation(dataset_path, model_path, output_dir_path, tracker_path, botsort=False, nopersist=False,
                                srt_path=None, drone_profile=None, manual_altitude=None, calibration=None,
                                evaluate=True, researcher_data_accumulator=None):
    print(f"Loading configuration files...")

    dataset_path = Path(dataset_path)
    srt_path = Path(srt_path) if srt_path else None

    image_dir_path = dataset_path / settings['images_dir']
    label_dir_path = dataset_path / settings['labels_dir']

    tracker = DolphinTracker(model_path, output_dir_path, tracker_path, botsort, nopersist, researcher_data_accumulator)
    results = tracker.track_from_images(image_dir_path)
    tracker.save_tracker_results(image_dir_path, results, srt_path, drone_profile, manual_altitude, calibration)
    if evaluate:
        metrics = tracker.evaluate(image_dir_path, label_dir_path)
        return metrics
    else:
        return None


class DolphinTracker:

    def __init__(self, model_path, output_dir_path, tracker_path, botsort, nopersist, researcher_data_accumulator):
        self.model_path = Path(model_path)
        self.output_dir_path = Path(output_dir_path)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.run_name = self.output_dir_path.name
        self.researcher_data_accumulator = researcher_data_accumulator

        self.botsort = botsort
        self.nopersist = nopersist
        self.using_obb = False

        self.img_width = None
        self.img_height = None

        if self.botsort:
            self.tracker_path = project_path(settings['ultralytics_botsort'])
        elif tracker_path is None or tracker_path == "Default":
            self.tracker_path = project_path(settings['ultralytics_bytetrack'])
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
        self.images_index_file_path = self.output_dir_path / f'{self.run_name}_{settings["images_index_suffix"]}'
        self.gt_images_index_file_path = self.output_dir_path / f'{self.run_name}_{settings["gt_images_index_suffix"]}'

        with open(self.tracker_path, 'r') as file:
            tracker_settings = yaml.safe_load(file)

        if 'iou' in tracker_settings:
            self.iou = tracker_settings['iou']
        else:
            self.iou = 0.7 # this aligns with default value in ultralytics

        if 'conf' in tracker_settings:
            self.conf = tracker_settings['conf']
        else:
            self.conf = 0.25 # this aligns with default value in ultralytics

    def track_from_images(self, image_dir_path):
        return self.model_instance.track(source=image_dir_path, tracker=self.tracker_path, device=self.device,
                                         persist=(not self.nopersist), iou=self.iou, stream=True, conf=self.conf)

    def evaluate(self, image_dir_path, label_dir_path):
        if label_dir_path.is_dir():

            self._save_ground_truth(image_dir_path, label_dir_path)
            gt_df = mm.io.loadtxt(self.gt_file_path)
            pred_df = mm.io.loadtxt(self.results_file_path)

            gt_df.reset_index(inplace=True)
            pred_df.reset_index(inplace=True)

            if self.images_index_file_path.stat().st_size != 0:
                images_index_df = pd.read_csv(self.images_index_file_path, header=None)
            else:
                images_index_df = pd.DataFrame(columns=['file_stem'])

            gt_images_index_df = pd.read_csv(self.gt_images_index_file_path, header=None)

            images_index_df.columns = ['file_stem']
            gt_images_index_df.columns = ['file_stem']

            images_index_df['file_stem'] = [Path(f).stem for f in images_index_df['file_stem']]
            gt_images_index_df['file_stem'] = [Path(f).stem for f in gt_images_index_df['file_stem']]

            pred_df = pd.concat([images_index_df, pred_df], axis=1)
            gt_df = pd.concat([gt_images_index_df, gt_df], axis=1)

            metrics = self._compute_metrics(gt_df, pred_df)
            return metrics
        else:
            print(f"No ground truth label directory found; looked for {label_dir_path}. Not running metrics "
                  f"calculations.")

    def save_tracker_results(self, image_dir_path, results, srt_path=None, drone_profile=None, manual_altitude=None,
                             calibration=None):
        files = list(image_dir_path.glob('*.jpg'))
        files.sort()
        pattern = r"(\d+)(?=[._](jpg))"

        first_image = cv2.imread(files[0])
        if first_image is None:
            raise ValueError(f"Could not read image file {files[0]}")

        img_height, img_width = first_image.shape[:2]
        self.img_width = img_width
        self.img_height = img_height

        tracker_data_accumulator = DataAccumulator(bbox_type='xyxy', width=img_width, height=img_height, units='pct')
        if self.researcher_data_accumulator:
            self.researcher_data_accumulator.set_imgsize(img_width, img_height)
        image_files_index = []

        for i, result in enumerate(results):
            match = re.search(pattern, str(files[i]))
            frame_id = match.group(1)
            file_stem = files[i].stem

            if result.obb:
                for xywhr, xyxyxyxy in zip(result.obb.xywhr, result.obb.xyxyxyxy):
                    """ Explanation for posterity

                    This is a feature to work with oriented bounding boxes (OBB) in YOLO format. The goal was to use
                    this format to reflect dolphin orientation, which is a useful statistic for researchers. However,
                    the bounding boxes only record orientation in the range [0, 180) degrees, which is not sufficient
                    for orientation. I am preserving the code in the feature set for posterity, but it is not fully
                    functional and will require further development to be useful. Below are some notes on the work
                    remaining, but it should not be considered exhaustive. Intrepid travellers, beware.

                    - Track IDs aren't integrated yet -- need get them
                    - Right now this always sets the orientation to the bottom-rightest side. YOLO-OBB does some weird
                        things with the rotation (see https://docs.ultralytics.com/datasets/obb/#yolo-obb-format).
                    - Researcher data isn't fully built out yet.
                    - Evaluation against ground truth isn't built out yet.
                    """
                    self.using_obb = True
                    if self.researcher_data_accumulator:
                        self.researcher_data_accumulator.set_bbox_type('xywhr')
                        self.researcher_data_accumulator.set_units('px')

                    tracker_data_accumulator.set_bbox_type('xyxyxyxy')
                    tracker_data_accumulator.set_units('px')

                    _, _, _, _, rotation = torch.flatten(xywhr).tolist()
                    x1, y1, x2, y2, x3, y3, x4, y4 = torch.flatten(xyxyxyxy).tolist()
                    x1, y1, x2, y2, x3, y3, x4, y4 = rotate_points(x1, y1, x2, y2, x3, y3, x4, y4, rotation)

                    image_files_index.append(str(files[i]))

                    bbox = [x1, y1, x2, y2, x3, y3, x4, y4]
                    tracker_data_accumulator.add_object(i, frame_id, -1, bbox)
                    if self.researcher_data_accumulator:
                        researcher_bbox = torch.flatten(xywhr).tolist()
                        self.researcher_data_accumulator.add_object(i, frame_id, -1, researcher_bbox)
            else:
                for box in result.boxes:
                    if box.id:
                        researcher_bbox = box.xyxyn[0].tolist()
                        track_id = int(box.id.item())
                        conf = box.conf.item()

                        image_files_index.append(str(files[i]))

                        tracker_data_accumulator.add_object(i, frame_id, track_id, researcher_bbox, conf=conf)
                        if self.researcher_data_accumulator:
                            self.researcher_data_accumulator.add_object(i, file_stem, track_id, researcher_bbox)
        if self.researcher_data_accumulator:
            self.researcher_data_accumulator.finished_adding_objects()
        tracker_data_accumulator.finished_adding_objects()

        if not self.using_obb:
            if self.researcher_data_accumulator:
                self.researcher_data_accumulator.reformat_bbox('xywh', drop_original=True)
                self.researcher_data_accumulator.add_conversion_columns('px', drop_original=True)
            tracker_data_accumulator.reformat_bbox('xywh', drop_original=True)
        else:
            tracker_data_accumulator.add_conversion_columns('pct', drop_original=True)



        images_df = pd.DataFrame(image_files_index, columns=['ImageFile'])
        images_df.to_csv(self.images_index_file_path, index=False, header=False)

        #tracker_data_accumulator.add_conversion_columns('px', drop_original=True)

        tracker_data_accumulator.to_csv(self.results_file_path, mot15=True)
        print(f"Wrote raw results to {self.results_file_path}")

    def _save_ground_truth(self, image_dir_path, label_dir_path):
        # pattern = r"(\d+)(?=[._](txt|jpg\.rf))"  # Roboflow files have _jpg. followed by a hash then .txt; this
        # gets those
        pattern = r"(\d+)(?=[._](jpg|jpg\.rf))"
        files = list(image_dir_path.glob('*.jpg'))
        files.sort()

        ground_truth_data_accumulator = DataAccumulator(bbox_type='xyxy', units='pct', width=self.img_width, height=self.img_height)
        images_index = []

        # Define the tracks directory path
        tracks_dir_path = label_dir_path.parent / settings['tracks_dir']
        zero_ids_warning = False

        # Read and process each file
        for frame_index, image_path in enumerate(files):
            match = re.search(pattern, str(image_path))
            if not match:
                raise ValueError(f"Could not process filename {image_path}")

            frame_id = match.group(1)

            label_path = label_dir_path / image_path.with_suffix('.txt').name
            track_path = tracks_dir_path / label_path.name

            # Load track IDs if the track file exists
            track_ids = []
            if track_path.exists():
                with open(track_path, 'r') as track_file:
                    track_ids = [line.strip() for line in track_file.readlines()]

            if label_path.exists():
                with open(label_path, 'r') as f:
                    next_id = 1
                    empty_frame = True

                    file_name = label_path.name

                    for i, line in enumerate(f.readlines()):
                        if line.strip():
                            line_data = line.split(" ")
                            bbox = line_data[1:5]
                            # Use track ID from track file if available, otherwise use next_id if ID is 0 or None
                            if track_ids and i < len(track_ids):
                                track_id = int(track_ids[i])
                            # The logic below is only preserved as a vestige for use with older dataset iterations for the sake of comparison. It should not be necessary for datasets created after March 3, 2025. If it is, you need to reconvert the dataset.
                            elif line_data[0] == '0' or line_data[0] == 'None':
                                track_id = next_id
                                next_id += 1
                                zero_ids_warning = True
                            else:
                                track_id = int(line_data[0])

                            ground_truth_data_accumulator.add_object(frame_index, frame_id, track_id, bbox, conf=1)
                            images_index.append(file_name)
                            empty_frame = False

                    if empty_frame:
                        ground_truth_data_accumulator.add_object(frame_index, frame_id, -1, ['nan', 'nan', 'nan', 'nan'])
                        images_index.append(file_name)

        images_df = pd.DataFrame(images_index, columns=['file_stem'])
        images_df.to_csv(self.gt_images_index_file_path, index=False, header=False)

        ground_truth_data_accumulator.finished_adding_objects()

        #ground_truth_data_accumulator.add_conversion_columns('px', drop_original=True)

        ground_truth_data_accumulator.to_csv(self.gt_file_path, mot15=True)

        if zero_ids_warning:
            print("Warning: Found 0 or None IDs in ground truth labels. Assigned new IDs. If this is unexpected, "
                  "check that the track files are correct and present.")

    def _iou(self, box1, box2):
        poly1 = shape_box(*box1)
        poly2 = shape_box(*box2)
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union != 0 else 0

    def _compute_metrics(self, gt_df, pred_df):
        tm = TrackingMetrics()

        metrics = settings['tracking_metrics']

        frames = sorted(gt_df['file_stem'].unique())

        for idx, file_stem in enumerate(frames):

            g = gt_df[gt_df['file_stem'] == file_stem]
            p = pred_df[pred_df['file_stem'] == file_stem]

            g = g.dropna()

            gt_ids = g['Id'].values
            pr_ids = p['Id'].values

            gt_boxes = [(row['X'], row['Y'], row['X'] + row['Width'], row['Y'] + row['Height']) for index, row in
                        g.iterrows()]
            tr_boxes = [(row['X'], row['Y'], row['X'] + row['Width'], row['Y'] + row['Height']) for index, row in
                        p.iterrows()]

            distances = np.full((len(gt_ids), len(pr_ids)), np.inf)
            for i, gt_box in enumerate(gt_boxes):
                for j, tr_box in enumerate(tr_boxes):
                    iou = self._iou(gt_box, tr_box)
                    if iou > 0.5:
                        distances[i, j] = 1 - iou

            # Before updating the accumulator, print the current frame and IDs
            print(f"Processing frame: {file_stem} Ground truth IDs: {gt_ids} Tracker IDs: {pr_ids}")

            # Update the accumulator
            tm.update(gt_ids, pr_ids, distances, idx)

        summary = tm.compute(metrics=metrics, outfile=self.metrics_file_path, printsum=True)

        tm.write_events(self.metrics_events_path)

        print(f"\nResults written to {self.metrics_file_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Run dolphin tracking and evaluation.")
    parser.add_argument('--dataset', required=True, help="Path to the dataset directory.")
    parser.add_argument('--model', required=True, help="Path to the model file.")
    parser.add_argument('--output', required=True, help="Path to the output directory.")
    parser.add_argument('--tracker', help="Tracker config file")
    parser.add_argument('--botsort', action='store_true', help="Enable BotSort parameter.")
    parser.add_argument('--nopersist', action='store_true', help="Disable persistence in tracking.")
    parser.add_argument('--srt', help="Path to an SRT file corresponding to the video input.")
    parser.add_argument('--drone', help="Drone profile for GSD calculation.")
    parser.add_argument('--altitude', type=float,
                        help="Manual altitude in meters for GSD calculation. Overrides SRT altitude if present.")
    parser.add_argument('--calibration', type=float, help="Manual calibration factor for GSD calculation.")
    parser.add_argument('--skip_evaluation', action='store_true', help="Skip evaluation.")

    args = parser.parse_args()

    results = run_tracking_and_evaluation(dataset_path=args.dataset, model_path=args.model, output_dir_path=args.output,
                                          tracker_path=args.tracker, botsort=args.botsort, nopersist=args.nopersist,
                                          srt_path=args.srt, drone_profile=args.drone, manual_altitude=args.altitude,
                                          calibration=args.calibration, evaluate=not args.skip_evaluation)


if __name__ == '__main__':
    main()
