import motmetrics as mm
import pandas as pd
import numpy as np
import re
import yaml
import pysrt
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict
from .settings import settings, project_path


class TrackingMetrics:

    def __init__(self):
        self.acc = [mm.MOTAccumulator(auto_id=False)]
        self.acc_names = ['Default']
        self.mh = mm.metrics.create()
        self.metrics = None

    def update(self, gt_ids, tr_ids, distances, frame_id):
        self.acc[0].update(gt_ids, tr_ids, distances, frameid=frame_id)

    def compute(self, metrics, outfile=None, printsum=False):
        self.metrics = metrics

        summary = self.mh.compute_many(self.acc, metrics=self.metrics, names=self.acc_names)

        if outfile:
            summary.to_csv(outfile)

        if printsum:
            self.print_formatted(summary)

        return summary

    def print_formatted(self, summary):
        # modified from example here: https://github.com/cheind/py-motmetrics
        strsummary = mm.io.render_summary(summary.iloc[[-1], :],  # Use list to preserve `DataFrame` type
            formatters=self.mh.formatters, namemap=settings['tracking_metrics_names'])
        print("\n" + strsummary)

    def print_events(self):
        print(self.acc[0].mot_events)

    def write_events(self, filename):
        self.acc[0].mot_events.to_csv(filename)

# TODO: Validate negative frames
class HOTAContainer:
    def __init__(self, input_folder):
        self.benchmark = 'dolphin_tracker'
        self.split = 'test'
        self.challenge_name = f"{self.benchmark}-{self.split}"
        self.tracker_name = 'DolphinTracker'

        self.input_folder = input_folder
        self.run_name = input_folder.name
        self.gt_filepath = self.input_folder / f"{self.run_name}_{settings['gt_file_suffix']}"
        self.pred_filepath = self.input_folder / f"{self.run_name}_{settings['results_file_suffix']}"

        self.script_path = self._track_eval_path('scripts/run_mot_challenge.py')
        self.seqmaps_folder = self._track_eval_path('data/gt/mot_challenge/seqmaps')
        self.gt_folder = self._track_eval_path(f'data/gt/mot_challenge/{self.challenge_name}/{self.run_name}')
        self.pred_folder = self._track_eval_path(f'data/trackers/mot_challenge/{self.challenge_name}/{self.tracker_name}')
        self.pred_input_folder = self.pred_folder / 'data'
        self.output_sub_folder = f"output/{self.run_name}"
        self.output_folder = self.pred_folder / self.output_sub_folder
        self.output_file_path = self.output_folder / f"pedestrian_summary.txt"
        self.final_output_file_path = self.input_folder / f"{self.run_name}_{settings['hota_file_suffix']}"

        self.seq_info_ini_path = self.gt_folder / "seqinfo.ini"
        self.seq_map_path = self.seqmaps_folder / f"{self.challenge_name}.txt"
        self.track_eval_gt_file_path = self.gt_folder / "gt/gt.txt"
        self.track_eval_input_path = self.pred_input_folder / f"{self.run_name}.txt"

        self.track_eval_input_path.parent.mkdir(parents=True, exist_ok=True)
        self.track_eval_gt_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.seq_map_path.parent.mkdir(parents=True, exist_ok=True)

        self.sequence_length = self._get_sequence_length()
        self.frames_with_gt = set()

        self._update_seqinfo_ini()
        self._update_seq_map()
        self._update_gt_file()
        self._update_input_file()


    def run(self):
        args = [
            '--BENCHMARK', self.benchmark,
            '--SPLIT_TO_EVAL', self.split,
            '--TRACKERS_TO_EVAL', self.tracker_name,
            '--DO_PREPROC', 'False',
            '--OUTPUT_SUMMARY', 'False',
            '--OUTPUT_EMPTY_CLASSES', 'False',
            '--OUTPUT_DETAILED', 'False',
            '--PLOT_CURVES', 'False',
            '--PRINT_CONFIG', 'False',
            '--PRINT_RESULTS', 'False',
            '--TIME_PROGRESS', 'False',
            '--OUTPUT_SUB_FOLDER', self.output_sub_folder,
        ]
        #print(f"Running TrackEval with command: {' '.join(['python', str(self.script_path), *args])}")
        subprocess.run(['python', self.script_path, *args])
        shutil.copy(self.output_file_path, self.final_output_file_path)
        print(f"TrackEval finished. Results saved to {self.final_output_file_path}")

    def get_hota(self):
        if not self.final_output_file_path.exists():
            raise ValueError(f"Output file {self.final_output_file_path} does not exist. Did you run the evaluation?")

        df = pd.read_csv(self.final_output_file_path, sep=' ')
        hota = float(df['HOTA'].values[0])
        print(f"HOTA:\n{hota}")
        return hota

    def _update_seqinfo_ini(self):
        with open(self.seq_info_ini_path, 'w') as file:
            file.write(f"[Sequence]\n"
                       f"name={self.run_name}\n"
                       f"seqLength={self.sequence_length}")

    def _get_sequence_length(self):
        gt_df = pd.read_csv(self.gt_filepath, header=None)
        pred_df = pd.read_csv(self.pred_filepath, header=None)

        gt_df.columns = settings['bbox_file_columns']
        pred_df.columns = settings['bbox_file_columns']

        gt_max = gt_df['frame'].max()
        pred_max = pred_df['frame'].max()

        return max(gt_max, pred_max) + 1

    def _update_seq_map(self):
        with open(self.seq_map_path, 'w') as file:
            file.write(f"name\n{self.run_name}")

    def _update_gt_file(self):
        shutil.copy(self.gt_filepath, self.track_eval_gt_file_path)
        df = pd.read_csv(self.track_eval_gt_file_path, header=None)
        df.columns = settings['bbox_file_columns']
        self._increment_frame_ids(df)
        self.frames_with_gt = set(df['frame'].unique())
        df.insert(6, 'confidence', 1)
        df = df.dropna()
        df.to_csv(self.track_eval_gt_file_path, index=False, header=False)

    def _update_input_file(self):
        shutil.copy(self.pred_filepath, self.track_eval_input_path)
        df = pd.read_csv(self.track_eval_input_path, header=None)
        df.columns = settings['bbox_file_columns']
        self._increment_frame_ids(df)
        df = df[df['frame'].isin(self.frames_with_gt)]
        df.insert(6, 'confidence', 1)
        df['confidence'] = df['visibility']

        df.to_csv(self.track_eval_input_path, index=False, header=False)

    def _increment_frame_ids(self, df):
        df['frame'] = df['frame'].astype(int) + 1

    def _track_eval_path(self, file):
        return project_path('src/utils/inc/third_party/TrackEval/' + file)

class OrientationMetrics:
    def __init__(self, pred_df, gt_df):
        self.pred_df = pred_df
        self.gt_df = gt_df
        self.metrics = {}

        original_length = len(self.gt_df)
        self.lines = self.gt_df.copy()
        self.lines = self.lines.merge(self.pred_df, on=['dataloader_index', 'object_id', 'filename'], suffixes=['_gt', '_pred'])
        if len(self.lines) != original_length:
            raise ValueError(f"Mismatch in length of line results: {len(self.lines)} vs {original_length}")
        self.lines = self.lines[['filename', 'dataloader_index', 'object_id', 'angle_gt', 'angle_pred', 'x_val_gt', 'y_val_gt', 'x_val_pred', 'y_val_pred']]

    def calculate_metrics(self):

        print(self.lines)

        pred_angles_rad = np.radians(self.lines['angle_pred'])
        gt_angles_rad = np.radians(self.lines['angle_gt'])

        cos_diff = np.cos(pred_angles_rad) - np.cos(gt_angles_rad)
        self.lines['cosine_diff'] = cos_diff
        self.metrics['MAE_Cos'] = np.mean(np.abs(cos_diff))

        angular_distance = np.degrees(np.arccos(np.clip(np.cos(pred_angles_rad - gt_angles_rad), -1.0, 1.0)))
        self.lines['angular_distance'] = angular_distance
        self.metrics['MeanAngularDistance'] = np.mean(angular_distance)

        # Calculate vector norm error
        x_diff = self.lines['x_val_pred'] - self.lines['x_val_gt']
        y_diff = self.lines['y_val_pred'] - self.lines['y_val_gt']
        vector_norm_error = np.sqrt(x_diff ** 2 + y_diff ** 2)
        self.lines['vector_norm_error'] = vector_norm_error
        self.metrics['MeanVectorNormError'] = np.mean(vector_norm_error)

        return self.metrics

    def print_results(self):
        print("\nOrientation Metrics:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")

    def write_results(self, filename):
        df = pd.DataFrame([self.metrics])
        df.to_csv(filename, index=False)

    def write_line_results(self, filename):
        self.lines.to_csv(filename, index=False)

class DataAccumulator:
    BBOX_TYPES = {'xywhr': ['CenterX', 'CenterY', 'Width', 'Height', 'Rotation'],
                  'xyxyxyxy': ['Point1X', 'Point1Y', 'Point2X', 'Point2Y', 'Point3X', 'Point3Y', 'Point4X', 'Point4Y'],
                  'xyxy': ['Point1X', 'Point1Y', 'Point2X', 'Point2Y'],
                  'xywh': ['CenterX', 'CenterY', 'Width', 'Height'], }
    BBOX_UNITS = ['px', 'pct', 'm']

    def __init__(self, bbox_type, units, width=None, height=None):
        self.bbox_type = None
        self.position_columns = None
        self.units = None
        self.df = None
        self.finished = False

        self.img_width = width
        self.img_height = height

        self.data = []

        self.set_bbox_type(bbox_type)
        self.set_units(units)

    def set_imgsize(self, width, height):
        if self.img_width is not None and self.img_height is not None:
            raise ValueError("Image dimensions already set. Cannot change after data has been added.")
        elif len(self.data) > 0:
            raise ValueError("Cannot change image dimensions after data has been added.")
        else:
            self.img_width = width
            self.img_height = height

    def to_csv(self, filepath, ignore_columns=None, mot15=False, header=True, only_columns=None):
        if not self.finished:
            raise ValueError("Cannot write to csv before calling finished_adding_objects().")
        df = self.df.copy()
        if mot15:
            header=False
            df['Unused1'] = -1
            df['Unused2'] = -1
            if 'Confidence' not in df.columns:
                df['Confidence'] = -1

            only_columns = ['FrameIndex', 'ObjectID'] + self.position_columns + ['Unused1', 'Unused2', 'Confidence']

        if ignore_columns is not None:
            df.drop(columns=ignore_columns, inplace=True)

        if only_columns is not None:
            df = df[only_columns]

        df.to_csv(filepath, index=False, header=header)

    def set_bbox_type(self, bbox_type):
        if bbox_type not in self.BBOX_TYPES:
            raise ValueError(f"Invalid bbox_type: {bbox_type}. Must be one of {self.BBOX_TYPES}.")
        elif len(self.data) > 0 and self.bbox_type != bbox_type:
            raise ValueError(f"Cannot change bbox_type from {self.bbox_type} to {bbox_type} after data has been added.")
        else:
            self.bbox_type = bbox_type
            self.position_columns = self.BBOX_TYPES[self.bbox_type]

    def set_units(self, units):
        if units not in self.BBOX_UNITS:
            raise ValueError(f"Invalid units: {units}. Must be one of {self.BBOX_UNITS}.")
        elif self.units == units:
            return
        elif len(self.data) > 0:
            raise ValueError(f"Cannot change input units from {self.units} to {units} after data has been added.")
        else:
            self.units = units

    def add_object(self, frame_index, frame_id, object_id, bbox, conf=-1):
        if len(bbox) != len(self.BBOX_TYPES[self.bbox_type]):
            raise ValueError(
                f"Invalid bbox length: {len(bbox)}. Must be {len(self.BBOX_TYPES[self.bbox_type])} for bbox type "
                f"{self.bbox_type}.")
        if len(self.position_columns) != len(self.BBOX_TYPES[self.bbox_type]):
            raise ValueError(
                f"Column count has changed to: {len(self.position_columns)}. Must be "
                f"{len(self.BBOX_TYPES[self.bbox_type])} for bbox type {self.bbox_type} while adding objects.")
        if self.finished:
            raise ValueError("Cannot add objects after calling finished_adding_objects().")

        self.data.append({'FrameIndex': frame_index, 'FrameID': frame_id, 'ObjectID': object_id,
                          **{col: float(bbox[i]) for i, col in enumerate(self.position_columns)}, 'Confidence': conf})

    def finished_adding_objects(self):
        self.df = self._convert_to_df()
        self.finished = True

    def get_columns(self):
        return self._all_columns()

    def add_conversion_columns(self, units, drop_original=False):
        if not self.finished:
            raise ValueError("Cannot add conversion columns before calling finished_adding_objects().")
        if units not in self.BBOX_UNITS:
            raise ValueError(f"Invalid units: {units}. Must be one of {self.BBOX_UNITS}.")
        if self.img_width is None or self.img_height is None:
            raise ValueError("Image dimensions not set. Cannot convert units.")
        if self.units == 'pct' and units == 'px':
            self._convert_data_columns('pct', 'px', drop_original)
        elif self.units == 'px' and units == 'pct':
            self._convert_data_columns('px', 'pct', drop_original)
        elif units == 'm':
            self._convert_data_columns('px', 'm', drop_original)
        else:
            raise NotImplementedError(f"Conversion from {self.units} to {units} not implemented.")

    def reformat_bbox(self, bbox_type, drop_original=True):
        if not self.finished:
            raise ValueError("Cannot reformat bbox before calling finished_adding_objects().")
        if bbox_type not in self.BBOX_TYPES:
            raise ValueError(f"Invalid bbox_type: {bbox_type}. Must be one of {self.BBOX_TYPES}.")
        if self.bbox_type == bbox_type:
            return

        if self.bbox_type == 'xyxy' and bbox_type == 'xywh':
            self._reformat_bbox_columns('xyxy', 'xywh', drop_original)
        else:
            raise NotImplementedError(f"Conversion from {self.bbox_type} to {bbox_type} not implemented.")

    def load_srt_altitudes(self, srt_path):
        if not self.finished:
            raise ValueError("Cannot load SRT file before calling finished_adding_objects().")

        rel_alt_pattern = r"\[rel_alt\ ?:\ (\S*)"
        focal_len_pattern = r"\[focal_len\ ?:\ (\d*)"

        srt = pysrt.open(srt_path)

        sub_base_index = srt[0].index
        base_altitude = float(settings['estimated_drone_starting_altitude_m'])

        srt_data = []

        for sub in srt:
            alt_match = re.search(rel_alt_pattern, sub.text)
            focal_match = re.search(focal_len_pattern, sub.text)
            if not alt_match or not focal_match:
                raise ValueError(f"Could not find relative altitude or focal length in subtitle: {sub.text}")

            srt_data.append({'FrameIndex': int(sub.index - sub_base_index), 'RelAltitude_m': float(alt_match.group(1)),
                'FocalLengthRaw': float(focal_match.group(1)), })

        srt_df = pd.DataFrame(srt_data)
        srt_df['EstAltitude_m'] = srt_df['RelAltitude_m'] + base_altitude

        new_df = self.df.merge(srt_df, on='FrameIndex', how='left')

        new_columns = ['RelAltitude_m', 'FocalLengthRaw', 'EstAltitude_m']

        affected_rows = new_df[new_columns].isna().any(axis=1).sum()
        if affected_rows > 0:
            print(
                f"Number of missing SRT frames to be filled with closest preceding (proceeding if preceding does not "
                f"exist) SRT data: {affected_rows}")

        # This will fill in NaN values (generated by the left merge if the srt file is shorter than the video)
        new_df[new_columns] = new_df[new_columns].ffill()
        new_df[new_columns] = new_df[new_columns].bfill()

        self.df = new_df

    def load_manual_altitudes(self, manual_altitude):
        if not self.finished:
            raise ValueError("Cannot load altitudes before calling finished_adding_objects().")
        self.df['EstAltitude_m'] = float(manual_altitude)

    def load_orientations(self, orientation_outfile_path):
        if not self.finished:
            raise ValueError("Cannot load orientations before calling finished_adding_objects().")
        if not Path(orientation_outfile_path).exists():
            raise ValueError(f"Orientation file {orientation_outfile_path} does not exist.")
        orientation_df = pd.read_csv(orientation_outfile_path)

        self.df['Angle_deg'] = orientation_df['angle']

        # orientation_df.rename(columns={'angle': 'Angle_deg'}, inplace=True)
        # # images_index_filename = orientation_outfile_path.parent / orientation_outfile_path.name.replace(
        # #     settings['orientations_results_suffix'], settings['images_index_suffix'])
        # # images_index = pd.read_csv(images_index_filename, header=None)
        # # images_index.columns = ['filename']
        # # images_index['filename'] = [Path(f).stem for f in images_index['filename']]
        #
        #
        #
        #
        # orientation_df['FrameIndex'] = orientation_df['FrameIndex'].astype(int)
        # self.df = self.df.merge(orientation_df, on='FrameIndex', how='left')
        # self.df['Rotation'] = self.df['Rotation'].fillna(0)

    def add_gsd_column(self, drone_profile, calibration):
        drone_profile = drone_profile or settings['default_drone_profile']
        drone_profile = Path(drone_profile)
        drone_profile_path = project_path(settings['drone_profile_dir']) / f"{drone_profile.stem}.yaml"
        with open(drone_profile_path, 'r') as file:
            drone_settings = yaml.safe_load(file)

        if not calibration and 'manual_calibration_factor' in drone_settings:
            calibration = drone_settings['manual_calibration_factor']

        if drone_settings['gsd_calculation_mode'] == 'fov':
            # For 90 deg overhead videos:
            # tan(fov/2) = (field_height_m/2) / altitude_m
            # equivalently:
            # field_height_m = 2 * altitude_m * tan(fov/2)
            # GSD_mpx = field_height_m / image_height_px
            # GSD_cmpx = GSD_mpx * 100
            if 'camera_vertical_fov_deg' in drone_settings:
                self.df['GSD_cmpx'] = 100 * (2 * self.df['EstAltitude_m'] * np.tan(
                    np.radians(drone_settings['camera_vertical_fov_deg'] / 2))) / self.img_height
            else:
                raise ValueError(
                    "camera_vertical_fov_deg must be provided in the drone profile for fov-mode GSD calculation")

            print(f"\nSRT data for first frame before calibration factor: GSD: {self.df['GSD_cmpx'][0]} cm/px. "
                  f"Image height: {self.img_height} px. "
                  f"Estimated altitude: {self.df['EstAltitude_m'][0]} m. Camera vertical FOV: "
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

                if 'FocalLengthRaw' not in self.df:
                    raise ValueError(
                        "focal length must be provided in the SRT file for sensor-mode GSD calculation. Did you forget"
                        " to specify the srt file? If you do not have one, use the --altitude argument to specify"
                        " the altitude manually, and use 'fov' or 'focal' as the GSD calculation mode in the drone "
                        "profile.")

                self.df['FocalLength_mm'] = self.df['FocalLengthRaw'] * focal_length_multiplier / crop_factor
            else:
                if 'focal_length_mm' in drone_settings:
                    self.df['FocalLength_mm'] = drone_settings['focal_length_mm']
                else:
                    raise ValueError(
                        "focal_length_mm must be provided in the drone profile for focal-mode GSD calculation.")

            self.df['GSD_cmpx'] = (self.df['EstAltitude_m'] * 100 * sensor_height) / (
                        self.df['FocalLength_mm'] * self.img_height)
            print(f"\nSRT data for first frame before calibration factor: GSD: {self.df['GSD_cmpx'][0]} cm/px. "
                  f"Image height: {self.img_height} px. "
                  f"Focal length: {self.df['FocalLength_mm'][0]} mm. "
                  f"Estimated altitude: {self.df['EstAltitude_m'][0]} m. Sensor height: "
                  f"{settings['drone_sensor_height_mm']} "
                  f"mm.")
            print(f"\nFormula for GSD: (estimated altitude*100 * sensor height/10) / (focal length/10 * image height)")

        else:
            raise ValueError(f"Invalid GSD calculation mode: {drone_settings['gsd_calculation_mode']}")

        if calibration:
            self.df['GSD_cmpx'] *= calibration
            print(f"Applied manual calibration factor of {calibration}")

    def add_individual_count_column(self):
        # Calculate the count of individuals in the frame
        # There is only one count of individuals in the frame, and it's reflected in the number of rows per frame_id.
        # Simple enough to add as a column in the researcher output.
        self.df['IndividualCount'] = self.df.groupby('FrameIndex')['ObjectID'].transform('count')

    def add_distances_columns(self):
        # Calculate the distance between each pair of individuals
        col_index = self._get_data_column_index()
        for units, columns in col_index.items():
            if units == 'px' or units == 'm':
                distances = []
                for frame_index, group in self.df.groupby('FrameIndex'):
                    centers = group[[columns['CenterX'], columns['CenterY']]].values
                    object_ids = group['ObjectID'].values
                    dist_matrix = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)
                    for i in range(len(group)):
                        dist_dict = {object_ids[j]: dist_matrix[i, j] for j in range(len(group)) if i != j}
                        distances.append(dist_dict)

                self.df[f"Distances_{units}"] = distances
                self.df[f"MaxDistance_{units}"] = self.df.groupby('FrameID')[f"Distances_{units}"].transform(
                    lambda x: max((max(d.values()) for d in x if d), default=0))
                self.position_columns.append(f"Distances_{units}")
                self.position_columns.append(f"MaxDistance_{units}")

    def _convert_to_df(self):
        df = pd.DataFrame(self.data, columns=self._all_columns())
        rename_cols = {}
        for col in self.position_columns:
            rename_cols[col] = f"{col}_{self.units}"
            df[col] = df[col].astype(float)
        df['FrameIndex'] = df['FrameIndex'].astype(int)
        df['FrameID'] = df['FrameID'].astype(int)
        df['ObjectID'] = df['ObjectID'].astype('Int64')
        self.position_columns = [val for _, val in rename_cols.items()]
        df.rename(columns=rename_cols, inplace=True)
        return df

    def _convert_data_columns(self, from_units, to_units, drop_original):
        new_columns = self.position_columns.copy()
        converted_columns = []

        col_index = self._get_data_column_index()
        for units, columns in col_index.items():
            if units == from_units:
                for col_base_name, full_col_name in columns.items():
                    new_col_name = f"{col_base_name}_{to_units}"

                    if from_units == 'pct' and to_units == 'px':
                        self.df[new_col_name] = self._convert_values_pct_to_px(self.df[full_col_name], col_base_name)
                    elif from_units == 'px' and to_units == 'm':
                        self.df[new_col_name] = self._convert_values_px_to_m(self.df[full_col_name], col_base_name)
                    elif from_units == 'px' and to_units == 'pct':
                        self.df[new_col_name] = self._convert_values_px_to_pct(self.df[full_col_name], col_base_name)
                    else:
                        raise NotImplementedError(f"Conversion from {from_units} to {to_units} not implemented.")

                    converted_columns.append(full_col_name)
                    new_columns.append(new_col_name)

        if drop_original:
            self.df.drop(columns=converted_columns, inplace=True)
            for col in converted_columns:
                new_columns.remove(col)

        self.position_columns = new_columns

    def _get_data_column_index(self):
        column_pattern = re.compile(r'(\S+)_([a-z]+)\s?')
        col_index = defaultdict(dict)

        for full_col_name in self.position_columns:
            match = column_pattern.match(full_col_name)
            if not match:
                raise ValueError(f"Invalid column name: {full_col_name}. Must match pattern: {column_pattern.pattern}")
            col_base_name = match.group(1)
            col_units = match.group(2)

            col_index[col_units][col_base_name] = full_col_name

        return col_index

    def _reformat_bbox_columns(self, from_type, to_type, drop_original):
        bbox_col_index = self._get_data_column_index()
        new_columns = self.position_columns.copy()
        converted_columns = []
        for units, columns in bbox_col_index.items():
            for col_base_name, full_col_name in columns.items():
                if col_base_name in self.BBOX_TYPES[from_type]:
                    converted_columns.append(full_col_name)

        if from_type == 'xyxy' and to_type == 'xywh':
            for unit, columns in bbox_col_index.items():
                self.df[f'CenterX_{unit}'] = (self.df[columns['Point1X']] + self.df[columns['Point2X']]) / 2
                self.df[f'CenterY_{unit}'] = (self.df[columns['Point1Y']] + self.df[columns['Point2Y']]) / 2
                self.df[f'Width_{unit}'] = self.df[columns['Point2X']] - self.df[columns['Point1X']]
                self.df[f'Height_{unit}'] = self.df[columns['Point2Y']] - self.df[columns['Point1Y']]
                new_columns.extend([f'CenterX_{unit}', f'CenterY_{unit}', f'Width_{unit}', f'Height_{unit}'])
        else:
            raise NotImplementedError(f"Conversion from {from_type} to {to_type} not implemented.")

        if drop_original:
            self.df.drop(columns=converted_columns, inplace=True)
            for col in converted_columns:
                new_columns.remove(col)

        self.position_columns = new_columns

    def _convert_values_pct_to_px(self, value, col_name):
        if col_name == 'Width':
            return value * self.img_width
        elif col_name == 'Height':
            return value * self.img_height
        elif col_name.endswith('X'):
            return value * self.img_width
        elif col_name.endswith('Y'):
            return value * self.img_height
        else:
            raise ValueError(f"Invalid column name: {col_name}. Must end with X or Y if not Width or Height.")

    def _convert_values_px_to_pct(self, value, col_name):
        if col_name == 'Width':
            return value / self.img_width
        elif col_name == 'Height':
            return value / self.img_height
        elif col_name.endswith('X'):
            return value / self.img_width
        elif col_name.endswith('Y'):
            return value / self.img_height
        else:
            raise ValueError(f"Invalid column name: {col_name}. Must end with X or Y if not Width or Height.")

    def _convert_values_px_to_m(self, value, col_name):
        ground_sampling_distance_column = 'GSD_cmpx'
        if ground_sampling_distance_column in self.df.columns:
            if col_name == 'Distances':
                raise NotImplementedError(
                    "Distance conversion from px to m not implemented. Please convert px to m BEFORE calculating "
                    "distances.")
            else:
                return value * self.df[ground_sampling_distance_column] / 100
        else:
            raise ValueError(
                f"Ground sampling distance column '{ground_sampling_distance_column}' not found. Cannot convert "
                f"pixels to meters.")

    def _all_columns(self):
        columns = ['FrameIndex', 'FrameID', 'ObjectID']
        columns.extend(self.position_columns)
        columns.append('Confidence')
        return columns
