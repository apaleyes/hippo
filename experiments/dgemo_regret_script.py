# this assumes DGEMO is available locally and an experiment was run
# furthermore, we did following modifications to the DGEMO source code (following git notation for changes)

# mobo/mobo.py
# @@ -88,6 +88,7 @@ class MOBO:
#              print('========== Iteration %d ==========' % i)
#
#              timer = Timer()
# +            iteration_timer = Timer()
#
#              # data normalization
#              self.transformation.fit(self.X, self.Y)
# @@ -116,6 +117,7 @@ class MOBO:
#              timer.log('New samples evaluated')
#
#              # statistics
# +            iteration_timer.log("Total step time")
#              global_timer.log('Total runtime', reset=False)
#              print('Total evaluations: %d, hypervolume: %.4f\n' % (self.sample_num, self.status['hv']))

# scripts/run.py
# @@ -23,7 +23,9 @@ def main():
#      parser.add_argument('--n-iter', type=int, default=20)
#      parser.add_argument('--n-var', type=int, default=6)
#      parser.add_argument('--n-obj', type=int, default=2)
# +    parser.add_argument('--n-init-sample', type=int, default=50)
#      args = parser.parse_args()
# +    # args, unknown = parser.parse_known_args()
#
#      # get reference point for each problem first, make sure every algorithm and every seed run using the same reference point
#      ref_dict = {}
# @@ -53,6 +55,7 @@ def main():
#                          --batch-size {args.batch_size} --n-iter {args.n_iter} \
#                          --ref-point {ref_dict[problem]} \
#                          --n-process {args.n_inner_process} \
# +                        --n-init-sample {args.n_init_sample} \
#                          --subfolder {args.subfolder} --log-to-file'
#                      if algo != 'dgemo':
#                          command += ' --n-gen 200'

# with these changes in place, following lines were used to run the experiment itself
# python scripts/run.py --problem dtlz2 --algo dgemo --n-seed 10 --n-iter 10 --n-var 4 --n-obj 3 --batch-size 4 --n-init-sample 3
# python scripts/run.py --problem zdt3 --algo dgemo --n-seed 10 --n-iter 5 --n-var 6 --n-obj 2 --batch-size 4 --n-init-sample 3

import numpy as np
from experiment import get_hv_regret
from generate_true_pareto_fronts import read_true_pf
from trieste.acquisition.multi_objective.pareto import get_reference_point
import pathlib
import re

from test_functions import Gardner2D, ScaledHartmannAckley6D, BraninGoldsteinPrice, VLMOP2

current_dir = pathlib.Path(__file__).parent

problems = [
    # {"name": "dtlz2", "n_steps": 10, "n_seeds": 10, "n_init_points": 3, "n_query_points": 4, "n_vars": 4, "n_obj": 3},
    # {"name": "zdt3", "n_steps": 5, "n_seeds": 10, "n_init_points": 3, "n_query_points": 4, "n_vars": 6, "n_obj": 2},
    {"name": "gardner", "n_steps": 7, "n_seeds": 10, "n_init_points": 3, "n_query_points": 4, "n_vars": 2, "n_obj": 2,
    "true_pf_filename": Gardner2D().true_pf_filename},
    # {"name": "ha", "n_steps": 20, "n_seeds": 10, "n_init_points": 6, "n_query_points": 4, "n_vars": 6, "n_obj": 2,
    # "true_pf_filename": ScaledHartmannAckley6D().true_pf_filename},
    # {"name": "bgp", "n_steps": 10, "n_seeds": 10, "n_init_points": 3, "n_query_points": 4, "n_vars": 2, "n_obj": 2,
    # "true_pf_filename": BraninGoldsteinPrice().true_pf_filename},
    # {"name": "vlmop2", "n_steps": 10, "n_seeds": 10, "n_init_points": 3, "n_query_points": 4, "n_vars": 2, "n_obj": 2}
]

dgemo_root = current_dir.joinpath("results/dgemo_results")
save_to_file = True

def process_result(problem):
    if "true_pf_filename" in problem:
        true_pf = read_true_pf(problem["true_pf_filename"])
    else:
        true_pareto_file = str(dgemo_root.joinpath(problem["name"], "default/TrueParetoFront.csv").resolve())
        true_pf = np.genfromtxt(true_pareto_file, delimiter=',', skip_header=True)
    
    evaluated_pareto_file_format = str(dgemo_root.joinpath(problem["name"], "default/dgemo/{0}/EvaluatedSamples.csv").resolve())
    log_file_format = str(dgemo_root.joinpath(problem["name"], "default/dgemo/{0}/log.txt").resolve())
   

    hv_regret = []
    times = []
    for seed in range(problem["n_seeds"]):
        evaluated_pareto_file = evaluated_pareto_file_format.format(seed)
        observed_data = np.genfromtxt(evaluated_pareto_file, delimiter=',', skip_header=True)

        observed_pf = observed_data[:, (1 + problem["n_vars"]):(1 + problem["n_vars"] + problem["n_obj"])]
        ref_point = get_reference_point(np.concatenate((true_pf, observed_pf)))
        regret = get_hv_regret(true_pf, observed_pf, problem["n_init_points"], reference_point=ref_point)
        hv_regret.append(regret)

        with open(log_file_format.format(seed)) as f:
            log_lines = f.readlines()

        # log file has a special line for logging the step time
        step_time_lines = [line for line in log_lines if line.startswith("Total step time")]
        step_times_str = [re.findall("\d+\.\d+", line) for line in step_time_lines]
        times.append([float(t[0]) for t in step_times_str])

    if save_to_file:
        filename = f"DGEMO_{problem['name']}_" \
                   f"n_initial_points_{problem['n_init_points']}_" \
                   f"n_query_points_{problem['n_query_points']}_" \
                   f"n_optimization_steps_{problem['n_steps']}_" \
                   f"n_repeats_{problem['n_seeds']}_" \
                   f"seed_{None}"

        file_path = current_dir.joinpath("results", filename).resolve()
        np.savetxt(str(file_path), hv_regret, delimiter=",")

        file_path = current_dir.joinpath("results", filename + "_time").resolve()
        np.savetxt(str(file_path), times, delimiter=",")
    else:
        print(hv_regret)
        print(times)

if __name__ == "__main__":
    for problem in problems:
        process_result(problem)
