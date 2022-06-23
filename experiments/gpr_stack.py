import tensorflow as tf

from trieste.models.interfaces import TrainableModelStack
from trieste.models.gpflow import GaussianProcessRegression
from trieste.types import TensorType

class GPRStack(TrainableModelStack):
    def __init__(
        self,
        model_with_event_size: tuple[GaussianProcessRegression, int],
        *models_with_event_sizes: tuple[GaussianProcessRegression, int],
    ):
        r"""
        The order of individual models specified at :meth:`__init__` determines the order of the
        :class:`ModelStack` output dimensions.

        :param model_with_event_size: The first model, and the size of its output events.
            **Note:** This is a separate parameter to ``models_with_event_sizes`` simply so that the
            method signature requires at least one model. It is not treated specially.
        :param \*models_with_event_sizes: The other models, and sizes of their output events.
        """

        super().__init__(model_with_event_size, *models_with_event_sizes)
        # if not all([isinstance(m, GaussianProcessRegression) for m in self._models]):
        #     raise ValueError(
        #         f"""All models in GPRStack have to be GaussianProcessRegression instances"""
        #     )

    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ):
        """Compute the posterior covariance between sets of query points.

        If query_points_1 is (N, D) and query_points_2 is (M, D), and we have K models in the stack,
        the resulting object shape is (K, N, M).
        """
        covs_between_points = [model.covariance_between_points(query_points_1, query_points_2) for model in self._models]
        return tf.stack(covs_between_points, axis=-1)
