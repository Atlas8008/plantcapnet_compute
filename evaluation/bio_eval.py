from .utils.biological_analyses import run_bio_analysis2
from .utils.time_series import correlation_metrics_from_files
from .evaluation import PostEvaluationMethod


class BioEvaluation(PostEvaluationMethod):
    """BioEvaluation class for evaluating biological data.

    Includes evaluation of DCA-Procrustes-Correlation among other metrics.
    """
    def __init__(self, *args, name=None, **kwargs):
        if name is None:
            name = "bio_eval"

        super().__init__(name)

        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return run_bio_analysis2(*self.args, **self.kwargs)


class CorrelationEvaluation(PostEvaluationMethod):
    """Class for evaluating the correlation of predicted and observed values.
    """
    def __init__(self, *args, name=None, **kwargs):
        if name is None:
            name = "corr_eval"

        super().__init__(name)

        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        print(self.name)
        return correlation_metrics_from_files(*self.args, **self.kwargs)