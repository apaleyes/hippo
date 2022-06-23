from typing import Optional, cast
import tensorflow as tf

from trieste.models.interfaces import ProbabilisticModel
from trieste.data import Dataset
from trieste.types import TensorType

from trieste.acquisition.function import ExpectedHypervolumeImprovement
from trieste.acquisition.interface import (
    AcquisitionFunction, SingleModelGreedyAcquisitionBuilder, SingleModelAcquisitionBuilder
)


class MOLocalPenalizationAcquisitionFunction(SingleModelGreedyAcquisitionBuilder):
    def __init__(
        self
    ):
        self._base_builder: SingleModelAcquisitionBuilder = ExpectedHypervolumeImprovement()
        self._base_acquisition_function: Optional[AcquisitionFunction] = None
        self._penalization: Optional[mo_penalizer] = None
        self._penalized_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: 
        :raise tf.errors.InvalidArgumentError: If the ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        acq = self._update_base_acquisition_function(model, dataset)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_penalization(acq, model, pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(self._base_acquisition_function is not None, [])

        if new_optimization_step:
            self._update_base_acquisition_function(model, dataset)

        if pending_points is None or len(pending_points) == 0:
            # no penalization required if no pending_points
            return cast(AcquisitionFunction, self._base_acquisition_function)

        return self._update_penalization(function, model, pending_points)

    def _update_penalization(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._penalized_acquisition is not None and isinstance(
            self._penalization, mo_penalizer
        ):
            # if possible, just update the penalization function variables
            self._penalization.update(pending_points)
            return self._penalized_acquisition
        else:
            # otherwise construct a new penalized acquisition function
            self._penalization = mo_penalizer(model, pending_points)

        # # handy plotting for acquisition function
        # import matplotlib.pyplot as plt

        # def plot_fn(fn, label, c):
        #     import numpy as np
        #     import math 
            
        #     x = np.linspace(0, 2*math.pi, 1000)
        #     y = fn(x.reshape(-1, 1, 1))

        #     plt.plot(x, y, label=label, c=c)

        @tf.function
        def penalized_acquisition(x: TensorType) -> TensorType:
            log_acq = tf.math.log(
                cast(AcquisitionFunction, self._base_acquisition_function)(x)
            ) + tf.math.log(self._penalization(x))
            return tf.math.exp(log_acq)

        # # plot acquisition function and batch points
        # plt.figure()
        # plt.vlines(tf.squeeze(pending_points), ymin=0, ymax=1, label="batch points so far", colors="green")
        # plot_fn(self._base_acquisition_function, "base function", c="blue")
        # plot_fn(self._penalization, "penalization", c="red")
        # plot_fn(penalized_acquisition, "penalized acquisition", c="purple")
        # # tf.print("------------------------------------")
        # # tf.print(pending_points)
        # # tf.print(self._base_acquisition_function(tf.expand_dims(pending_points, axis=1)))
        # # tf.print(self._penalization(tf.expand_dims(pending_points, axis=1)))
        # # tf.print(penalized_acquisition(tf.expand_dims(pending_points, axis=1)))
        # # tf.print("------------------------------------")
        # plt.legend()
        # plt.show()

        self._penalized_acquisition = penalized_acquisition
        return penalized_acquisition

    def _update_base_acquisition_function(
        self, model: ProbabilisticModel, dataset: Dataset
    ) -> AcquisitionFunction:
        if self._base_acquisition_function is None:
            self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                model,
                dataset=dataset,
            )
        else:
            self._base_acquisition_function = self._base_builder.update_acquisition_function(
                self._base_acquisition_function,
                model,
                dataset=dataset,
            )
        return self._base_acquisition_function

class mo_penalizer():
    def __init__(self, model: ProbabilisticModel, pending_points: TensorType):
        tf.debugging.Assert(pending_points is not None and len(pending_points) != 0, [])

        self._model = model
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means = tf.Variable(pending_means, shape=[None, *pending_means.shape[1:]])
        self._pending_vars = tf.Variable(pending_vars, shape=[None, *pending_vars.shape[1:]])

    def update(
        self,
        pending_points: TensorType
    ) -> None:
        """Update the local penalizer with new pending points."""
        tf.debugging.Assert(pending_points is not None and len(pending_points) != 0, [])

        self._pending_points.assign(pending_points)
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means.assign(pending_means)
        self._pending_vars.assign(pending_vars)

    # @tf.function
    # def __call__(self, x: TensorType) -> TensorType:
    #     # x is [N, 1, D]
    #     x = tf.squeeze(x, axis=1) # x is now [N, D]

    #     # pending_points is [B, D] where B is the size of the batch collected so far
    #     cov_with_pending_points = self._model.covariance_between_points(x, self._pending_points) # [N, B, K], K is the number of models in the stack
    #     pending_means, pending_covs = self._model.predict(self._pending_points) # pending_means is [B, K], pending_covs is [B, K]
    #     x_means, x_covs = self._model.predict(x) # x_mean is [N, K], x_cov is [N, K]

    #     tf.debugging.assert_shapes(
    #         [
    #             (x, ["N", "D"]),
    #             (self._pending_points, ["B", "D"]),
    #             (cov_with_pending_points, ["N", "B", "K"]),
    #             (pending_means, ["B", "K"]),
    #             (pending_covs, ["B", "K"]),
    #             (x_means, ["N", "K"]),
    #             (x_covs, ["N", "K"])
    #         ],
    #         message="uh-oh"
    #     )

    #     # N = tf.shape(x)[0]
    #     # B = tf.shape(self._pending_points)[0]
    #     # K = tf.shape(cov_with_pending_points)[-1]

    #     x_means_expanded = x_means[:, None, :]
    #     x_covs_expanded = x_covs[:, None, :]
    #     pending_means_expanded = pending_means[None, :, :]
    #     pending_covs_expanded = pending_covs[None, :, :]

    #     # tf.print(x_covs_expanded)
    #     # tf.print(pending_covs_expanded)
    #     # tf.print(cov_with_pending_points)
    #     # tf.print(x_covs_expanded + pending_covs_expanded - 2.0 * cov_with_pending_points)

    #     CLAMP_LB = 1e-12
    #     variance = x_covs_expanded + pending_covs_expanded - 2.0 * cov_with_pending_points
    #     variance = tf.clip_by_value(variance, CLAMP_LB, variance.dtype.max)

    #     # mean = tf.clip_by_value(pending_means_expanded - x_means_expanded, CLAMP_LB, x_means_expanded.dtype.max)
    #     # stddev = tf.clip_by_value(tf.math.sqrt(variance), CLAMP_LB, variance.dtype.max)
    #     mean = pending_means_expanded - x_means_expanded
    #     stddev = tf.math.sqrt(variance)

    #     # print(variance)
    #     # print(stddev)

    #     f_diff_normal = tfp.distributions.Normal(loc=mean, scale=stddev)
    #     cdf = f_diff_normal.cdf(0.0)

    #     # print(cdf)

    #     tf.debugging.assert_shapes(
    #         [
    #             (x, ["N", "D"]),
    #             (self._pending_points, ["B", "D"]),
    #             (mean, ["N", "B", "K"]),
    #             (stddev, ["N", "B", "K"]),
    #             (cdf, ["N", "B", "K"])
    #         ],
    #         message="uh-oh-oh"
    #     )

    #     # tf.print(mean)
    #     # tf.print(stddev)
    #     # tf.print(cdf)
    #     # penalty = tf.reduce_prod((1.0 - tf.reduce_prod(1 - cdf, axis=-1)), axis=-1)
    #     penalty = tf.reduce_prod(1.0 - tf.reduce_prod(cdf, axis=-1), axis=-1)

    #     return tf.reshape(penalty, (-1, 1))

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        # x is [N, 1, D]
        x = tf.squeeze(x, axis=1) # x is now [N, D]
        x_means, x_vars = self._model.predict(x) # x_means is [N, K], x_vars is [N, K], where K is the number of models/objectives

        # self._pending_points is [B, D] where B is the size of the batch collected so far
        tf.debugging.assert_shapes(
            [
                (x, ["N", "D"]),
                (self._pending_points, ["B", "D"]),
                (self._pending_means, ["B", "K"]),
                (self._pending_vars, ["B", "K"]),
                (x_means, ["N", "K"]),
                (x_vars, ["N", "K"])
            ],
            message="Encountered unexpected shapes while calculating mean and variance of given point x and pending points"
        )

        x_means_expanded = x_means[:, None, :]
        pending_means_expanded = self._pending_means[None, :, :]
        pending_vars_expanded = self._pending_vars[None, :, :]
        pending_stddevs_expanded = tf.sqrt(pending_vars_expanded)
        standardize_mean_diff = tf.abs(x_means_expanded - pending_means_expanded) / pending_stddevs_expanded # [N, B, K]

        d = tf.norm(standardize_mean_diff, axis=-1) # [N, B]

        # warp the distance so that resulting value is from 0 to nearly 1
        # warped_d = 2 * (1.0 / (1.0 + tf.exp(-d)) - 0.5) # [N, B]
        # warped_d = 1.0 - 1.0 / (1.0 + d) # [N, B]
        import math
        warped_d = (2.0/math.pi) * tf.math.atan(d)
        penalty = tf.reduce_prod(warped_d, axis=-1) # [N,]

        return tf.reshape(penalty, (-1, 1))
