from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from sklearn import neighbors

class NeighborsBase(BaseEstimator, metaclass=ABCMeta):
    """Base class for nearest neighbors estimators."""

    @abstractmethod
    def __init__(self, n_neighbors=None,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, metric_params=None, n_jobs=None):
        
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs
        self._check_algorithm_metric()

    def _check_algorithm_metric(self):
        if self.algorithm not in ['auto', 'brute',
                                  'kd_tree', 'ball_tree']:
            raise ValueError("unrecognized algorithm: '%s'" % self.algorithm)

        if self.algorithm == 'auto':
            if (callable(self.metric) or
                  self.metric in neighbors.VALID_METRICS['ball_tree']):
                alg_check = 'ball_tree'
            else:
                alg_check = 'brute'
        else:
            alg_check = self.algorithm

        if callable(self.metric):
            if self.algorithm == 'kd_tree':
                # callable metric is only valid for brute force and ball_tree
                raise ValueError(
                    "kd_tree does not support callable metric '%s'"
                    "Function call overhead will result"
                    "in very poor performance."
                    % self.metric)
        elif self.metric not in neighbors.VALID_METRICS[alg_check]:
            raise ValueError("Metric '%s' not valid. Use "
                             "sorted(sklearn.neighbors.VALID_METRICS['%s']) "
                             "to get valid options. "
                             "Metric can also be a callable function."
                             % (self.metric, alg_check))

        if self.metric_params is not None and 'p' in self.metric_params:
            if self.p is not None:
                warnings.warn("Parameter p is found in metric_params. "
                              "The corresponding parameter from __init__ "
                              "is ignored.", SyntaxWarning, stacklevel=3)
            effective_p = self.metric_params['p']
        else:
            effective_p = self.p

        if self.metric in ['wminkowski', 'minkowski'] and effective_p < 1:
            raise ValueError("p must be greater than one for minkowski metric")
