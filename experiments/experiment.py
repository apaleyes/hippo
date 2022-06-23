from typing import Callable
import tensorflow as tf
import gpflow
import numpy as np
import time
import pathlib
from dataclasses import dataclass, field



from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.interfaces import TrainablePredictJointReparamModelStack
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE

from trieste.acquisition import BatchMonteCarloExpectedHypervolumeImprovement, Fantasizer, ExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.ask_tell_optimization import AskTellOptimizer

from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point


from test_functions import TestFunction, get_test_function
from generate_true_pareto_fronts import read_true_pf
# from mo_penalization import MOLocalPenalizationAcquisitionFunction
from trieste.acquisition.function.multi_objective import HIPPO


def get_acquisition_function(name):
    if name == "BatchMC":
        return BatchMonteCarloExpectedHypervolumeImprovement(sample_size=100).using(OBJECTIVE)
    elif name == "HIPPO":
        base_acq = ExpectedHypervolumeImprovement().using(OBJECTIVE)
        return HIPPO(objective_tag=OBJECTIVE, base_acquisition_function_builder=base_acq)
    elif name == "KB":
        return Fantasizer(ExpectedHypervolumeImprovement())
    else:
        raise ValueError(f"Unknown method {name}")


@dataclass
class Config:
    acquisition_method_name: str
    test_function_name: str
    test_function: Callable = field(init=False)
    n_initial_points: int = 3
    n_query_points: int = 4
    n_optimization_steps: int = 3
    n_repeats: int = 5
    seed: int = None
    filename_prefix: str = None

    def __post_init__(self):
        # it's ok to create it once as re-use
        # because test functions are supposed to be stateless
        self.test_function = get_test_function(self.test_function_name)

    def create_acquisition_function(self):
        # acquisition functions can be stateful
        # so we need to re-create it each time
        return get_acquisition_function(self.acquisition_method_name)

    def get_filename(self):
        return ("" if self.filename_prefix is None else self.filename_prefix + "_") + \
               f"{self.acquisition_method_name}_" \
               f"{self.test_function_name}_" \
               f"n_initial_points_{self.n_initial_points}_" \
               f"n_query_points_{self.n_query_points}_" \
               f"n_optimization_steps_{self.n_optimization_steps}_" \
               f"n_repeats_{self.n_repeats}_" \
               f"seed_{self.seed}"

    @classmethod
    def from_dict(cls, args):
        config = Config(**args)
        return config


def build_stacked_independent_objectives_model(data, n_obj):
    gprs = []
    for idx in range(n_obj):
        single_obj_data = Dataset(
            data.query_points, tf.gather(data.observations, [idx], axis=1)
        )
        variance = tf.math.reduce_variance(single_obj_data.observations)
        kernel = gpflow.kernels.Matern52(variance, tf.constant(0.2, tf.float64))
        gpr = gpflow.models.GPR(single_obj_data.astuple(), kernel, noise_variance=1e-5)
        gpflow.utilities.set_trainable(gpr.likelihood, False)
        gprs.append((GaussianProcessRegression(gpr), 1))

    return TrainablePredictJointReparamModelStack(*gprs)


def get_hv_regret(true_points, observed_points, num_initial_points, reference_point=None):
    ref_point = get_reference_point(observed_points) if reference_point is None else reference_point
    ideal_hv = Pareto(true_points).hypervolume_indicator(ref_point)

    hv_regret = []
    for i in range(num_initial_points, len(observed_points)+1):
        observations = observed_points[:i, :]
        observed_hv = Pareto(observations).hypervolume_indicator(ref_point)

        hv_regret.append((ideal_hv - observed_hv).numpy())
    
    return hv_regret


def single_run(config: Config, save_to_file=False):
    print(f"Running {config.acquisition_method_name} on {config.test_function_name}")

    if config.seed is not None:
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    test_function: TestFunction = config.test_function
    true_pf = read_true_pf(test_function.true_pf_filename)
    observer = mk_observer(test_function, OBJECTIVE)

    hv_regret = []
    i = 0
    times = []
    while i < config.n_repeats:
        try:
            print(f"Repeat #{i}")
            initial_query_points = test_function.search_space.sample(config.n_initial_points)
            initial_data = observer(initial_query_points)

            model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE], test_function.n_objectives)
            acq_fn = config.create_acquisition_function()
            acq_rule = EfficientGlobalOptimization(acq_fn, num_query_points=config.n_query_points)

            ask_tell = AskTellOptimizer(test_function.search_space, initial_data, {OBJECTIVE: model}, acq_rule)

            repeat_times = []
            for step in range(config.n_optimization_steps):
                start = time.time()
                new_point = ask_tell.ask()
                new_data = observer(new_point)
                ask_tell.tell(new_data)
                stop = time.time()
                repeat_times.append(stop-start)
            result = ask_tell.to_result()

            # start = time.time()
            # result = BayesianOptimizer(observer, test_function.search_space).optimize(config.n_optimization_steps,
            #                                                                         initial_data,
            #                                                                         {OBJECTIVE: model},
            #                                                                         acq_rule)
            # stop = time.time()
            print(f"Finished in {sum(repeat_times):.2f}s")

            dataset = result.try_get_final_datasets()[OBJECTIVE]
            hv_regret.append(get_hv_regret(true_pf, dataset.observations, config.n_initial_points))
            times.append(repeat_times)
            i += 1
        except Exception as e:
            # this is really lazy
            # above code seems to work almost always
            # but occasionally may fail with Cholesky errors
            # this exception isn't fatal, and usually works fine upon restart
            # but since i can't remember the exact type of exception
            # let's just catch them all for now
            print(f"Failed with error: {e}")
            continue
            # raise

    if save_to_file:
        current_dir = pathlib.Path(__file__).parent
        file_path = current_dir.joinpath("results", config.get_filename()).resolve()
        np.savetxt(str(file_path), hv_regret, delimiter=",")
        print(f"Saved results to {file_path}")

        times_file_path = current_dir.joinpath("results", config.get_filename() + "_time").resolve()
        np.savetxt(str(times_file_path), times, delimiter=",")
        print(f"Saved times to {times_file_path}")

    return hv_regret

