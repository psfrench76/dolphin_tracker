from ultralytics import YOLO
import motmetrics as mm
import pandas as pd
import numpy as np
import click
import re
import torch
import yaml
import cv2
import pysrt
from pathlib import Path
from shapely.geometry import box as shape_box
import math

if __package__ is None or __package__ == '':
    from utils.inc.settings import settings, project_path
    from utils.inc.tracking_metrics import TrackingMetrics
    from utils.inc.oriented_bounding_boxes import rotate_points
else:
    from .utils.inc.settings import settings, project_path
    from .utils.inc.tracking_metrics import TrackingMetrics
    from .utils.inc.oriented_bounding_boxes import rotate_points

# TODO: update codebase to use images_index_file and not assume frame numbers are unique
#  (may impact evaluation on composite datasets)

@click.command()
@click.option('--dataset', required=True, help="Path to the dataset directory.")
@click.option('--model', required=True, help="Path to the model file.")
@click.option('--output', required=True, help="Path to the output directory.")
@click.option('--tracker', help="Tracker config file")
@click.option('--botsort', is_flag=True, help="Enable BotSort parameter.")
@click.option('--nopersist', is_flag=True, help="Disable persistence in tracking.")
@click.option('--srt', help="Path to an SRT file corresponding to the video input.")
@click.option('--drone', help="Drone profile for GSD calculation.")
@click.option('--altitude', help="Manual altitude in meters for GSD calculation. Overrides SRT altitude if present.")
@click.option('--calibration', help="Manual calibration factor for GSD calculation.")
@click.option('--skip_evaluation', is_flag=True, help="Skip evaluation.")
def main(dataset, model, output, tracker, botsort, nopersist, srt, drone, altitude, calibration, skip_evaluation):
    results = run_tracking_and_evaluation(dataset, model, output, tracker, botsort, nopersist, srt_path=srt,
                                          drone_profile=drone, manual_altitude=altitude, calibration=calibration,
                                          evaluate=not skip_evaluation)


class DolphinTracker:

    def __init__(self, model_path, output_dir_path, tracker_path, botsort, nopersist):
        self.model_path = Path(model_path)
        self.output_dir_path = Path(output_dir_path)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.run_name = self.output_dir_path.name

        self.botsort = botsort
        self.nopersist = nopersist
        self.last_img_height = None
        self.using_obb = False

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
            self.iou = 0.7

    def track_from_images(self, image_dir_path):
        return self.model_instance.track(source=image_dir_path, tracker=self.tracker_path, device=self.device,
                                         persist=(not self.nopersist), iou=self.iou, stream=True)

    def save_tracker_results(self, image_dir_path, results, srt_path=None, drone_profile=None, manual_altitude=None,
                             calibration=None):
        files = list(image_dir_path.glob('*.jpg'))
        files.sort()
        pattern = r"(\d+)(?=[._](jpg))"

        first_image = cv2.imread(files[0])
        if first_image is None:
            raise ValueError(f"Could not read image file {files[0]}")

        img_height, img_width = first_image.shape[:2]
        self.last_img_height = img_height

        camera_df = self.load_srt_altitudes(srt_path) if srt_path else None

        drone_profile = drone_profile or settings['default_drone_profile']
        if manual_altitude:
            if camera_df is None:
                camera_df = pd.DataFrame(index=range(len(files)))
            self.load_manual_altitudes(camera_df, float(manual_altitude))

        if camera_df is not None:
            self.calculate_gsd(camera_df, drone_profile, calibration)

        data = []
        researcher_data = []
        image_files_index = []

        for i, result in enumerate(results):
            match = re.search(pattern, str(files[i]))
            frame_id = match.group(1)
            if camera_df is not None:
                if i < len(camera_df):
                    camera_row = camera_df.iloc[i]
                else:
                    print(f"Warning: Ran out of rows in SRT file at frame index {i + 1} (frame ID {frame_id})."
                          f"Will use altitude and focal length from the last available row for the remaining frames.")

            if result.obb:
                for xywhr, xyxyxyxy in zip(result.obb.xywhr, result.obb.xyxyxyxy):
                    """ Explanation for posterity
                    
                    This is a feature to work with oriented bounding boxes (OBB) in YOLO format. The goal was to use
                    this format to reflect dolphin orientation, which is a useful statistic for researchers. However,
                    the bounding boxes only record orientation in the range [0, 180) degrees, which is not sufficient
                    for orientation. I am preserving the code in the feature set for posterity, but it is not fully
                    functional and will require further development to be useful. Below are some notes on the work
                    remaining, but it should not be considered exhaustive. Intrepid travellers, beware.
                    
                    - Track IDs aren't available yet -- need get them
                    - Right now this always sets the orientation to the bottom-rightest side. YOLO-OBB does some weird
                        things with the rotation (see https://docs.ultralytics.com/datasets/obb/#yolo-obb-format).
                    - Researcher data isn't built out yet.
                    - Evaluation against ground truth isn't built out yet.
                    """
                    self.using_obb = True
                    center_x_px, center_y_px, width_px, height_px, rotation = torch.flatten(xywhr).tolist()
                    x1, y1, x2, y2, x3, y3, x4, y4 = torch.flatten(xyxyxyxy).tolist()
                    x1, y1, x2, y2, x3, y3, x4, y4 = rotate_points(x1, y1, x2, y2, x3, y3, x4, y4, rotation)

                    data.append(
                        [frame_id, -1, x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height, x3 / img_width,
                         y3 / img_height, x4 / img_width, y4 / img_height, -1, -1, -1])
                    image_files_index.append(str(files[i]))
                    researcher_data.append([frame_id, -1, center_x_px, center_y_px, width_px, height_px, rotation])
            else:
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
                        data.append([frame_id, track_id, center_x, center_y, width, height, -1, -1, conf])
                        image_files_index.append(str(files[i]))

                        point_a_x_px = point_a_x * img_width
                        point_a_y_px = point_a_y * img_height
                        point_b_x_px = point_b_x * img_width
                        point_b_y_px = point_b_y * img_height
                        center_x_px = center_x * img_width
                        center_y_px = center_y * img_height
                        width_px = width * img_width
                        height_px = height * img_height
                        researcher_data.append(
                            [frame_id, track_id, point_a_x_px, point_a_y_px, point_b_x_px, point_b_y_px, width_px,
                                height_px, center_x_px, center_y_px])
                        if camera_df is not None and camera_row is not None:
                            gsd_mpx = camera_row['GSD_cmpx'] / 100
                            researcher_data[-1].extend(
                                [point_a_x_px * gsd_mpx, point_a_y_px * gsd_mpx, point_b_x_px * gsd_mpx,
                                 point_b_y_px * gsd_mpx, width_px * gsd_mpx, height_px * gsd_mpx, center_x_px * gsd_mpx,
                                 center_y_px * gsd_mpx, camera_row["est_alt_m"], camera_row["GSD_cmpx"]])
        if self.using_obb:
            df_columns = ['FrameID', 'ObjectID', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'Unused1', 'Unused2',
                          'Unused3']
            researcher_columns = ['FrameID', 'ObjectID', 'CenterX_px', 'CenterY_px', 'Width_px', 'Height_px',
                                  'Rotation']
        else:
            df_columns = ['FrameID', 'ObjectID', 'CenterX', 'CenterY', 'Width', 'Height', 'Unused1', 'Unused2',
                          'Confidence']
            researcher_columns = ['FrameID', 'ObjectID', 'Point1X_px', 'Point1Y_px', 'Point2X_px', 'Point2Y_px',
                'Width_px', 'Height_px', 'CenterX_px', 'CenterY_px']
        df = pd.DataFrame(data, columns=df_columns)
        df.to_csv(self.results_file_path, index=False, header=False)
        print(f"Wrote raw results to {self.results_file_path}")

        images_df = pd.DataFrame(image_files_index, columns=['ImageFile'])
        images_df.to_csv(self.images_index_file_path, index=False, header=False)

        if researcher_data:
            if camera_df is not None:
                researcher_columns.extend(
                    ['Point1X_m', 'Point1Y_m', 'Point2X_m', 'Point2Y_m', 'Width_m', 'Height_m', 'CenterX_m',
                        'CenterY_m', 'Altitude_m', 'GSD_cmpx'])
            researcher_df = pd.DataFrame(researcher_data, columns=researcher_columns)
            self.add_custom_researcher_columns(researcher_df)
            researcher_df.to_csv(self.researcher_output_path, index=False)
            print(f"Wrote researcher output data to {self.researcher_output_path}")

    def add_custom_researcher_columns(self, researcher_df):
        # Calculate the count of individuals in the frame
        # There is only one count of individuals in the frame, and it's reflected in the number of rows per frame_id.
        # Simple enough to add as a column in the researcher output.
        researcher_df['IndividualCount'] = researcher_df.groupby('FrameID')['ObjectID'].transform('count')

        # Calculate the distance between each pair of individuals
        # This results in dimensional expansion. Perhaps best as an array within a cell?

        distances = []
        if 'GSD_cmpx' in researcher_df:
            center_x_col = 'CenterX_m'
            center_y_col = 'CenterY_m'
            distance_col = 'Distances_m'
            max_col = 'MaxDistance_m'
        else:
            center_x_col = 'CenterX_px'
            center_y_col = 'CenterY_px'
            distance_col = 'Distances_px'
            max_col = 'MaxDistance_px'

        for frame_id, group in researcher_df.groupby('FrameID'):
            centers = group[[center_x_col, center_y_col]].values
            object_ids = group['ObjectID'].values
            dist_matrix = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)
            for i in range(len(group)):
                dist_dict = {object_ids[j]: dist_matrix[i, j] for j in range(len(group)) if i != j}
                distances.append(dist_dict)

        researcher_df[distance_col] = distances

        # Calculate the furthest distance between any two individuals
        # This results in a single value per frame_id
        researcher_df[max_col] = researcher_df.groupby('FrameID')[distance_col].transform(
            lambda x: max((max(d.values()) for d in x if d), default=0))

        # NOT READY: waiting on features or explanations

        # Calculate the orientation of each individual relative to each other
        # This results in dimensional expansion. Also relies on orientations (to be developed).

        # Calculate the interval of appearance on the surface
        # Needs more detail -- is this per individual? Distances between dives between individuals? Dimensionality?

        pass

    def save_ground_truth(self, label_dir_path):
        files = list(label_dir_path.glob('*.txt'))
        files.sort()

        data = []
        images_index = []

        # Define the tracks directory path
        tracks_dir_path = label_dir_path.parent / settings['tracks_dir']

        zero_ids_warning = False

        # Read and process each file
        for label_path in files:
            pattern = r"(\d+)(?=[._](txt|jpg\.rf))"  # Roboflow files have _jpg. followed by a hash then .txt; this
            # gets those
            match = re.search(pattern, str(label_path))
            if not match:
                raise ValueError(f"Could not process filename {label_path}")

            frame_id = match.group(1)
            track_path = tracks_dir_path / label_path.name

            # Load track IDs if the track file exists
            track_ids = []
            if track_path.exists():
                with open(track_path, 'r') as track_file:
                    track_ids = [line.strip() for line in track_file.readlines()]

            with open(label_path, 'r') as f:
                content = f.read()
                next_id = 1
                empty_frame = True

                file_name = label_path.name

                for i, line in enumerate(content.split("\n")):
                    if line.strip():
                        line_data = line.split(" ")
                        # Use track ID from track file if available, otherwise use next_id if ID is 0 or None
                        if track_ids and i < len(track_ids):
                            line_data[0] = track_ids[i]
                        # The logic below is only preserved as a vestige for use with older dataset iterations for
                        # the sake
                        # of comparison. It should not be necessary for datasets created after March 3, 2025. If it is,
                        # you need to reconvert the dataset.
                        elif line_data[0] == '0' or line_data[0] == 'None':
                            line_data[0] = str(next_id)
                            next_id += 1
                            zero_ids_warning = True

                        data.append([frame_id] + line_data)
                        images_index.append(file_name)
                        empty_frame = False
                if empty_frame:
                    data.append([frame_id, None, None, None, None, None])
                    images_index.append(file_name)

        # Convert list to DataFrame
        gt_df = pd.DataFrame(data, columns=['frame', 'id', 'x_min', 'y_min', 'x_max', 'y_max'])
        gt_df = gt_df.astype(
            {'frame': int, 'id': 'Int64', 'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float})

        gt_df['mot15col1'] = -1
        gt_df['mot15col2'] = -1
        gt_df['mot15col3'] = 1
        gt_df.to_csv(self.gt_file_path, index=False, header=False)

        images_df = pd.DataFrame(images_index, columns=['file_stem'])
        images_df.to_csv(self.gt_images_index_file_path, index=False, header=False)

        if zero_ids_warning:
            print("Warning: Found 0 or None IDs in ground truth labels. Assigned new IDs. If this is unexpected, "
                  "check that the track files are correct and present.")

    def _iou(self, box1, box2):
        poly1 = shape_box(*box1)
        poly2 = shape_box(*box2)
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union != 0 else 0

    def compute_metrics(self, gt_df, pred_df):
        tm = TrackingMetrics()

        metrics = settings['tracking_metrics']

        frames = sorted(gt_df['file_stem'].unique())

        for frame in frames:

            g = gt_df[gt_df['file_stem'] == frame]
            p = pred_df[pred_df['file_stem'] == frame]

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
                    # TODO: Experiment with changing this to self.iou or self.match_thresh?
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

            gt_df.reset_index(inplace=True)
            pred_df.reset_index(inplace=True)

            images_index_df = pd.read_csv(self.images_index_file_path, header=None)
            gt_images_index_df = pd.read_csv(self.gt_images_index_file_path, header=None)

            images_index_df.columns = ['file_stem']
            gt_images_index_df.columns = ['file_stem']

            images_index_df['file_stem'] = [Path(f).stem for f in images_index_df['file_stem']]
            gt_images_index_df['file_stem'] = [Path(f).stem for f in gt_images_index_df['file_stem']]

            pred_df = pd.concat([images_index_df, pred_df], axis=1)
            gt_df = pd.concat([gt_images_index_df, gt_df], axis=1)

            metrics = self.compute_metrics(gt_df, pred_df)
            return metrics
        else:
            print(
                f"No ground truth label directory found; looked for {label_dir_path}. Not running metrics "
                f"calculations.")

    def load_manual_altitudes(self, camera_df, manual_altitude):
        camera_df['est_alt_m'] = manual_altitude

    def load_srt_altitudes(self, srt_path):
        srt = pysrt.open(srt_path)
        altitudes = []
        frame_indexes = []
        focal_lengths = []
        rel_alt_pattern = r"\[rel_alt\ ?:\ (\S*)"
        focal_len_pattern = r"\[focal_len\ ?:\ (\d*)"

        for sub in srt:
            rel_alt_match = re.search(rel_alt_pattern, sub.text)
            focal_len_match = re.search(focal_len_pattern, sub.text)

            if rel_alt_match and focal_len_match:
                altitudes.append(float(rel_alt_match.group(1)))
                focal_lengths.append(float(focal_len_match.group(1)))
                frame_indexes.append(sub.index)
            else:
                raise ValueError(
                    f"Could not find relative altitude or focal length in frame {sub.index} subtitle: {sub.text}")

        df = pd.DataFrame({'frame_index': frame_indexes, 'rel_alt_m': altitudes, 'focal_len_raw': focal_lengths})
        df['est_alt_m'] = df['rel_alt_m'] + settings['estimated_drone_starting_altitude_m']

        return df

    def calculate_gsd(self, df, drone_profile, calibration=None):
        drone_profile = Path(drone_profile)
        drone_profile_path = project_path(settings['drone_profile_dir']) / f"{drone_profile.stem}.yaml"
        with open(drone_profile_path, 'r') as file:
            drone_settings = yaml.safe_load(file)

        if drone_settings['gsd_calculation_mode'] == 'fov':
            # For 90 deg overhead videos:
            # tan(fov/2) = (field_height_m/2) / altitude_m
            # equivalently:
            # field_height_m = 2 * altitude_m * tan(fov/2)
            # GSD_mpx = field_height_m / image_height_px
            # GSD_cmpx = GSD_mpx * 100
            if 'camera_vertical_fov_deg' in drone_settings:
                df['GSD_cmpx'] = 100 * (2 * df['est_alt_m'] * np.tan(
                    np.radians(drone_settings['camera_vertical_fov_deg'] / 2))) / self.last_img_height
            else:
                raise ValueError(
                    "camera_vertical_fov_deg must be provided in the drone profile for fov-mode GSD calculation")

            print(f"\nSRT data for first frame before calibration factor: GSD: {df['GSD_cmpx'][0]} cm/px. "
                  f"Image height: {self.last_img_height} px. "
                  f"Estimated altitude: {df['est_alt_m'][0]} m. Camera vertical FOV: "
                  f"{drone_settings['camera_vertical_fov_deg']} deg.")
            print(f"\nFormula for GSD: 100 * (2 * estimated altitude * tan(vertical FOV/2)) / image height")

        elif drone_settings['gsd_calculation_mode'] == 'sensor' or drone_settings['gsd_calculation_mode'] == 'focal':
            if 'sensor_height_mm' in drone_settings:
                sensor_height = drone_settings['sensor_height_mm']
            else:
                raise ValueError(
                    "sensor_height_mm must be provided in the drone profile for sensor-mode or focal-mode GSD "
                    "calculation.")

            if drone_settings['gsd_calculation_mode'] == 'sensor':
                if 'focal_length_multiplier' in drone_settings:
                    focal_length_multiplier = drone_settings['focal_length_multiplier']
                else:
                    raise ValueError(
                        "focal_length_multiplier must be provided in the drone profile for sensor-mode GSD "
                        "calculation.")

                if 'crop_factor' in drone_settings:
                    crop_factor = drone_settings['crop_factor']
                else:
                    raise ValueError(
                        "crop_factor must be provided in the drone profile for sensor-mode GSD calculation.")

                if 'focal_len_raw' not in df:
                    raise ValueError(
                        "focal length must be provided in the SRT file for sensor-mode GSD calculation. Did you forget"
                        " to specify the srt file? If you do not have one, use the --altitude argument to specify"
                        " the altitude manually, and use 'fov' or 'focal' as the GSD calculation mode in the drone "
                        "profile.")

                df['focal_len_mm'] = df['focal_len_raw'] * focal_length_multiplier / crop_factor
            else:
                if 'focal_length_mm' in drone_settings:
                    df['focal_len_mm'] = drone_settings['focal_length_mm']
                else:
                    raise ValueError(
                        "focal_length_mm must be provided in the drone profile for focal-mode GSD calculation.")

            df['GSD_cmpx'] = (df['est_alt_m'] * 100 * sensor_height) / (df['focal_len_mm'] * self.last_img_height)
            print(f"\nSRT data for first frame before calibration factor: GSD: {df['GSD_cmpx'][0]} cm/px. "
                  f"Image height: {self.last_img_height} px. "
                  f"Focal length: {df['focal_len_mm'][0]} mm. "
                  f"Estimated altitude: {df['est_alt_m'][0]} m. Sensor height: {settings['drone_sensor_height_mm']} "
                  f"mm.")
            print(f"\nFormula for GSD: (estimated altitude*100 * sensor height/10) / (focal length/10 * image height)")

        else:
            raise ValueError(f"Invalid GSD calculation mode: {drone_settings['gsd_calculation_mode']}")

        if not calibration and 'manual_calibration_factor' in drone_settings:
            calibration = drone_settings['manual_calibration_factor']

        if calibration:
            df['GSD_cmpx'] *= calibration
            print(f"Applied manual calibration factor of {calibration}")


def run_tracking_and_evaluation(dataset_path, model_path, output_dir_path, tracker_path, botsort=False, nopersist=False,
                                camera_df=None, srt_path=None, drone_profile=None, manual_altitude=None,
                                calibration=None, evaluate=True):
    print(f"Loading configuration files...")

    dataset_path = Path(dataset_path)
    srt_path = Path(srt_path) if srt_path else None

    image_dir_path = dataset_path / settings['images_dir']
    label_dir_path = dataset_path / settings['labels_dir']

    tracker = DolphinTracker(model_path, output_dir_path, tracker_path, botsort, nopersist)
    results = tracker.track_from_images(image_dir_path)
    tracker.save_tracker_results(image_dir_path, results, srt_path, drone_profile, manual_altitude, calibration)
    if evaluate:
        metrics = tracker.evaluate(label_dir_path)
        return metrics
    else:
        return None


if __name__ == '__main__':
    main()
