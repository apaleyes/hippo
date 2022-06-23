# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os

import trieste

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import ray
import numpy as np
import time

import trieste
from trieste.acquisition.rule import AsynchronousGreedy
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.observer import OBJECTIVE
from trieste.models.gpflow import SparseVariational, build_svgp
from trieste.models.interfaces import TrainablePredictJointReparamModelStack
from trieste.data import Dataset



from lunar_lander import LunarLander
from controller import demo_heuristic_lander

reward_max = 250.0
reward_min = -350.0
fuel_factor = 100.0

N_RUNS = 10

def lander_objective(x, sleep=True):
    # for each point compute average reward and fuel consumed over N_RUNS runs
    all_rewards = []
    all_fuel = []
    for w in x:
        results = [demo_heuristic_lander(LunarLander(), w) for _ in range(N_RUNS)]
        all_rewards.append([r.total_reward for r in results])
        all_fuel.append([r.total_fuel for r in results])

        if sleep:
            # insert some artificial delay that
            # increases linearly with the absolute value of points
            # which means our evaluations will take different time
            delay = 3 * np.sum(w)
            time.sleep(delay)

    # rewards_tensor = tf.convert_to_tensor(all_rewards, dtype=tf.float64)
    rewards_mean = np.reshape(np.mean(np.array(all_rewards), axis=1), (-1, 1))
    # fuel_tensor = tf.convert_to_tensor(all_fuel, dtype=tf.float64)
    fuel_mean = np.reshape(np.mean(np.array(all_fuel), axis=1), (-1, 1))

    # normalizing these values is tricky
    # we want them to be between 0 and 1
    # but we have only rough idea of the min/max ranges
    # reward seems to be between -350 and 250
    # fuel is normally between 0 and 100 (actually in theory it can go to 1000*0.33=330, but this seems to never happen in practice)

    rewards_mean = (reward_max - rewards_mean)/(reward_max - reward_min)
    fuel_mean = fuel_mean / fuel_factor

    objectives_tensor = np.concatenate([rewards_mean, fuel_mean], axis=-1)
    return (x, objectives_tensor)

@ray.remote
def ray_objective(points, sleep=True):
    return lander_objective(points, sleep)



search_space = trieste.space.Box([0.0] * 12, [1.5] * 12)

num_initial_points = 2 * search_space.dimension
initial_query_points = search_space.sample(num_initial_points)
_, initial_observations = lander_objective(initial_query_points.numpy(), sleep=False)
initial_data = Dataset(query_points=initial_query_points, observations=tf.convert_to_tensor(initial_observations, dtype=tf.float64))



def build_stacked_independent_objectives_model(data, n_obj):
    models = []
    for idx in range(n_obj):
        single_obj_data = Dataset(
            data.query_points, tf.gather(data.observations, [idx], axis=1)
        )
        svgp = build_svgp(single_obj_data, search_space)
        models.append((SparseVariational(svgp), 1))

    return TrainablePredictJointReparamModelStack(*models)

model = build_stacked_independent_objectives_model(initial_data, 2)

# Number of worker processes to run simultaneously
# Setting this to 1 will reduce our optimization to non-batch sequential
num_workers = 5
# Number of observations to collect
num_observations = 10
# Batch size of the acquisition function. We will wait for that many workers to return before launching a new batch
batch_size = 5
# Set this flag to False to disable sleep delays in case you want the notebook to execute quickly
enable_sleep_delays = False

from trieste.acquisition.rule import AsynchronousGreedy
from trieste.acquisition.function import HIPPO
from trieste.ask_tell_optimization import AskTellOptimizer

acquisition_function = HIPPO()
async_rule = AsynchronousGreedy(acquisition_function, num_query_points=batch_size)  # type: ignore
async_bo = AskTellOptimizer(search_space, initial_data, model, async_rule)




ray.init(ignore_reinit_error=True)

points_observed = 0
workers = []

# a helper function to launch a worker for a numpy array representing a single point
def launch_worker(x):
    worker = ray_objective.remote(np.atleast_2d(x), enable_sleep_delays)
    workers.append(worker)

start = time.time()
# get first couple of batches of points and init all workers
for _ in range(int(num_workers / batch_size)):
    points = async_bo.ask().numpy()
    np.apply_along_axis(launch_worker, axis=1, arr=points)


finished_workers = []
while points_observed < num_observations:
    ready_workers, remaining_workers = ray.wait(workers, timeout=0)
    finished_workers += ready_workers
    workers = remaining_workers

    if len(finished_workers) < batch_size:
        continue

    # we saw enough results to ask for a new batch

    new_observations = [
        ray.get(worker) for worker in finished_workers
    ]

    # new_observations is a list of tuples (point, observation value)
    # here we turn it into a Dataset and tell it to Trieste
    points_observed += len(new_observations)
    new_data = Dataset(
        query_points=tf.convert_to_tensor(
            np.concatenate([x[0] for x in new_observations], axis=0),
            dtype=tf.float64
        ),
        observations=tf.constant(
            np.concatenate([x[1] for x in new_observations], axis=0),
            dtype=tf.float64
        ),
    )
    print(new_data.query_points)
    async_bo.tell(new_data)

    # get a new batch of points
    # and launch workers for each point in the batch
    points = async_bo.ask().numpy()
    np.apply_along_axis(launch_worker, axis=1, arr=points)
    finished_workers = []

stop = time.time()

dataset = async_bo.to_result().try_get_final_dataset()
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

np.savetxt("query_points.txt", query_points, delimiter=',')
np.savetxt("observations.txt", observations, delimiter=',')
np.savetxt("time.txt", np.array([stop-start]), delimiter=',')