import numpy as np
import motmetrics as mm


class TrackingMetrics:
    METRIC_NAMES = {
        'hota_alpha': 'HOTA',
        'assa_alpha': 'ASSA',
        'deta_alpha': 'DETA',
        'num_frames': 'Num_Frames',
        'idf1': 'ID_F1',
        'idp': 'ID_Precision',
        'idr': 'ID_Recall',
        'recall': 'Recall',
        'precision': 'Precision',
        'num_objects': 'Objects',
        'mostly_tracked': 'Mostly_Tracked',
        'partially_tracked': 'Part_Tracked',
        'mostly_lost': 'Mostly_Lost',
        'num_false_positives': 'False_Pos',
        'num_misses': 'Misses',
        'num_switches': 'Switches',
        'num_fragmentations': 'Fragmentations',
        'mota': 'MOTA',
        'motp': 'MOTP',
        'id_global_assignment': 'ID_Global_Assn',
        'obj_frequencies': 'Object_Freq',
        'num_unique_objects': 'Num_Unique_Objects',
        'idfp': 'ID_False_Pos',
        'idfn': 'ID_False_Neg',
        'idtp': 'ID_True_Pos',
        'num_matches': 'Num_Matches'
    }
    HOTA_METRICS = ['hota_alpha', 'assa_alpha', 'deta_alpha']

    def __init__(self):
        self.acc = [mm.MOTAccumulator(auto_id=False)]
        self.acc_names = ['Default']
        self.mh = mm.metrics.create()
        self.metrics = None

    def update(self, gt_ids, tr_ids, distances, frame_id):
        self.acc[0].update(gt_ids, tr_ids, distances, frameid=frame_id)

    # modified from example here: https://github.com/cheind/py-motmetrics
    def compute_hota(self, df_gt, df_pred):
        # Require different thresholds for matching
        th_list = np.arange(0.05, 0.99, 0.05)
        res_list = mm.utils.compare_to_groundtruth_reweighting(df_gt, df_pred, "iou", distth=th_list)
        return res_list

    def compute(self, metrics, outfile=None, df_gt=None, df_pred=None, printsum=False):
        self.metrics = metrics

        if any([m in self.metrics for m in self.HOTA_METRICS]):
            if df_gt and df_pred:

                hota_acc = self.compute_hota(df_gt, df_pred)
                self.acc.extend(hota_acc)
                #self.acc = hota_acc
                self.acc_names.append("HOTA")

            else:
                raise "GT or Pred results not provided in call to 'TrackingMetrics.compute()'"
        print(self.acc)
        print(self.metrics)
        print(self.acc_names)

        summary = self.mh.compute_many(self.acc, metrics=self.metrics, names=self.acc_names)
        #summary = self.mh.compute_many(self.acc, self.metrics, generate_overall=True)
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
            namemap=self.METRIC_NAMES,
            #namemap={"hota_alpha": "HOTA", "assa_alpha": "ASSA", "deta_alpha": "DETA"},
        )
        print(strsummary)

    def print_events(self):
        print(self.acc[0].mot_events)

    def write_events(self, filename):
        self.acc[0].mot_events.to_csv(filename)

