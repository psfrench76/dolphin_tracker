import motmetrics as mm
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

