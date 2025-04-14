import motmetrics as mm
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from .settings import settings


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
        strsummary = mm.io.render_summary(
            summary.iloc[[-1], :],  # Use list to preserve `DataFrame` type
            formatters=self.mh.formatters,
            namemap=settings['tracking_metrics_names']
        )
        print("\n" + strsummary)

    def print_events(self):
        print(self.acc[0].mot_events)

    def write_events(self, filename):
        self.acc[0].mot_events.to_csv(filename)


class ResearcherData:
    BBOX_TYPES = {'obb': ['CenterX', 'CenterY', 'Width', 'Height', 'Rotation'],
                  'xyxy': ['Point1X', 'Point1Y', 'Point2X', 'Point2Y'],
                  'xywh': ['CenterX', 'CenterY', 'Width', 'Height'],}
    BBOX_UNITS = ['px', 'pct', 'm']

    def __init__(self, bbox_type, width, height, units):
        self.bbox_type = None
        self.data_columns = None
        self.units = None
        self.df = None
        self.finished = False


        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}. Must be positive.")

        self.img_width = width
        self.img_height = height

        self.data = []

        self.set_bbox_type(bbox_type)
        self.set_units(units)


    def to_csv(self, filepath):
        self.df.to_csv(filepath, index=False)

    def set_bbox_type(self, bbox_type):
        if bbox_type not in self.BBOX_TYPES:
            raise ValueError(f"Invalid bbox_type: {bbox_type}. Must be one of {self.BBOX_TYPES}.")
        elif len(self.data) > 0 and self.bbox_type != bbox_type:
            raise ValueError(f"Cannot change bbox_type from {self.bbox_type} to {bbox_type} after data has been added.")
        else:
            self.bbox_type = bbox_type
            self.data_columns = self.BBOX_TYPES[self.bbox_type]

    def set_units(self, units):
        if units not in self.BBOX_UNITS:
            raise ValueError(f"Invalid units: {units}. Must be one of {self.BBOX_UNITS}.")
        elif len(self.data) > 0:
            raise ValueError(f"Cannot change input units from {self.units} to {units} after data has been added.")
        else:
            self.units = units

    def add_object(self, frame_id, object_id, bbox):
        if len(bbox) != len(self.BBOX_TYPES[self.bbox_type]):
            raise ValueError(f"Invalid bbox length: {len(bbox)}. Must be {len(self.BBOX_TYPES[self.bbox_type])} for bbox type {self.bbox_type}.")
        if len(self.data_columns) != len(self.BBOX_TYPES[self.bbox_type]):
            raise ValueError(f"Column count has changed to: {len(self.data_columns)}. Must be {len(self.BBOX_TYPES[self.bbox_type])} for bbox type {self.bbox_type} while adding objects.")
        if self.finished:
            raise ValueError("Cannot add objects after calling finished_adding_objects().")

        self.data.append({
            'FrameID': frame_id,
            'ObjectID': object_id,
            **{col: bbox[i] for i, col in enumerate(self.data_columns)}
        })

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
        if self.units == 'pct' and units == 'px':
            self._convert_data_columns('pct', 'px', drop_original)
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

    def _convert_to_df(self):
        df = pd.DataFrame(self.data, columns=self._all_columns())
        rename_cols = {}
        for col in self.data_columns:
            rename_cols[col] = f"{col}_{self.units}"
            df[col] = df[col].astype(float)
        self.data_columns = [val for _, val in rename_cols.items()]
        df.rename(columns=rename_cols, inplace=True)
        return df

    def _convert_data_columns(self, from_units, to_units, drop_original):
        column_pattern = re.compile(r'(\S+)_([a-z]+)\s?')
        new_columns = self.data_columns.copy()
        converted_columns = []

        for full_col_name in self.data_columns:
            match = column_pattern.match(full_col_name)
            if not match:
                raise ValueError(f"Invalid column name: {full_col_name}. Must match pattern: {column_pattern.pattern}")
            col_base_name = match.group(1)
            col_units = match.group(2)

            if col_units != from_units:
                continue

            if drop_original:
                new_columns.remove(full_col_name)

            new_col_name = f"{col_base_name}_{to_units}"

            if from_units == 'pct' and to_units == 'px':
                self.df[new_col_name] = self._convert_values_pct_to_px(self.df[full_col_name], col_base_name)
            else:
                raise NotImplementedError(f"Conversion from {from_units} to {to_units} not implemented.")

            converted_columns.append(full_col_name)
            new_columns.append(new_col_name)

        if drop_original:
            self.df.drop(columns=converted_columns, inplace=True)

        self.data_columns = new_columns

    def _reformat_bbox_columns(self, from_type, to_type, drop_original):
        column_pattern = re.compile(r'(\S+)_([a-z]+)\s?')
        new_columns = self.data_columns.copy()
        converted_columns = []
        bbox_col_index = defaultdict(dict)

        if from_type != self.bbox_type:
            raise ValueError(f"Invalid from bbox_type: {from_type}. Must be {self.bbox_type}.")

        for full_col_name in self.data_columns:
            match = column_pattern.match(full_col_name)
            if not match:
                raise ValueError(f"Invalid column name: {full_col_name}. Must match pattern: {column_pattern.pattern}")
            col_base_name = match.group(1)
            col_units = match.group(2)

            if col_base_name not in self.BBOX_TYPES[from_type]:
                continue

            if drop_original:
                new_columns.remove(full_col_name)

            converted_columns.append(full_col_name)
            bbox_col_index[col_units][col_base_name] = full_col_name

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

        self.data_columns = new_columns

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

    def _all_columns(self):
        columns = ['FrameID', 'ObjectID']
        columns.extend(self.data_columns)
        return columns