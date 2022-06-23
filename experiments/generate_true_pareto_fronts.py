import numpy as np
import pathlib

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2

import test_functions


class PymooProblem(Problem):
    def __init__(self, search_space, f, n_obj):
        n_var = int(search_space.dimension)
        xl = search_space.lower.numpy()
        xu = search_space.upper.numpy()
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
        
        self._f = f
        
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._f(x)


def get_true_pf(problem, n_gen=1000):
    algorithm = NSGA2(pop_size=100)
    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=1, verbose=False)

    return res.F


def save_true_pf(true_pf, filename):
    current_dir = pathlib.Path(__file__).parent
    file_path = current_dir.joinpath("true_pf", filename).resolve()
    np.savetxt(str(file_path), true_pf, delimiter=",")
    return str(file_path)


def read_true_pf(filename):
    current_dir = pathlib.Path(__file__).parent
    file_path = current_dir.joinpath("true_pf", filename).resolve()
    return np.genfromtxt(str(file_path), delimiter=',')


def generate_true_pareto_front(test_f: test_functions.TestFunction):
    print("Generating true Pareto front for " + test_f.name)
    problem = PymooProblem(test_f.search_space, test_f, test_f.n_objectives)
    true_pf = get_true_pf(problem)
    file_path = save_true_pf(true_pf, test_f.true_pf_filename)
    print("Saved to " + file_path)


if __name__ == "__main__":
    # generate_true_pareto_front(test_functions.Simple1D())
    # generate_true_pareto_front(test_functions.Gardner2D())
    # generate_true_pareto_front(test_functions.ZDT3())
    # generate_true_pareto_front(test_functions.HartmannAckley6D())
    # generate_true_pareto_front(test_functions.ScaledHartmannAckley6D())
    # generate_true_pareto_front(test_functions.DTLZ2())
    # generate_true_pareto_front(test_functions.VLMOP2())
    # generate_true_pareto_front(test_functions.BraninGoldsteinPrice())
    generate_true_pareto_front(test_functions.RosenbrockAlpine2())
