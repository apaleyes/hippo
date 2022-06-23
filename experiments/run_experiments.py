import test_functions
from experiment import single_run, Config

def simple_1d():
    config_dict = {
        "test_function_name": test_functions.Simple1D.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 3,
        "n_repeats": 10
    }

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "HIPPO"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "KB"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


def gardner():
    config_dict = {
        "test_function_name": test_functions.Gardner2D.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 7,
        "n_repeats": 10
    }

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "HIPPO"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "KB"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


def hartmann_ackley():
    config_dict = {
        "test_function_name": test_functions.ScaledHartmannAckley6D.name,
        "n_initial_points": 6,
        "n_query_points": 4,
        "n_optimization_steps": 20,
        "n_repeats": 10
    }

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config_dict["n_optimization_steps"] = 15 # BatchMC really struggles
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    config_dict["acquisition_method_name"] = "HIPPO"
    config_dict["n_optimization_steps"] = 20
    config = Config.from_dict(config_dict)
    single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "KB"
    # config_dict["n_optimization_steps"] = 20
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


def zdt3():
    config_dict = {
        "test_function_name": test_functions.ZDT3.name,
        "n_initial_points": 3,
        "n_query_points": 2,
        "n_optimization_steps": 5,
        "n_repeats": 1
    }

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "HIPPO"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "KB"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)



def DTLZ2_3_objectives():
    config_dict = {
        "test_function_name": test_functions.DTLZ2.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 10,
        "n_repeats": 10
    }

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    config_dict["acquisition_method_name"] = "HIPPO"
    config = Config.from_dict(config_dict)
    single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "KB"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


def VLMOP2():
    config_dict = {
        "test_function_name": test_functions.VLMOP2.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 10,
        "n_repeats": 10
    }

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    config_dict["acquisition_method_name"] = "HIPPO"
    config = Config.from_dict(config_dict)
    single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "KB"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


def BraninGoldsteinPrice():
    config_dict = {
        "test_function_name": test_functions.BraninGoldsteinPrice.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 10,
        "n_repeats": 10
    }

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "HIPPO"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "KB"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


def RosenbrokAlpine2():
    config_dict = {
        "test_function_name": test_functions.RosenbrockAlpine2.name,
        "n_initial_points": 3,
        "n_query_points": 4,
        "n_optimization_steps": 10,
        "n_repeats": 10
    }

    # config_dict["acquisition_method_name"] = "HIPPO"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "KB"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)

    # config_dict["acquisition_method_name"] = "BatchMC"
    # config = Config.from_dict(config_dict)
    # single_run(config, save_to_file=True)


def many_batches():
    test_func_names = [
        test_functions.ScaledHartmannAckley6D.name,
        # test_functions.VLMOP2.name,
    ]

    batch_n_steps = [
        # (5, 30),
        (10, 15),
        (25, 6),
        (50, 3)
    ]

    config_dict = {
        "n_initial_points": 6,
        "n_repeats": 6,
        "filename_prefix": "batch-size-exp"
    }

    for test_func_name in test_func_names:
        config_dict["test_function_name"] = test_func_name
        for batch_size, steps in batch_n_steps:
            config_dict["n_query_points"] = batch_size
            config_dict["n_optimization_steps"] = steps

            config_dict["acquisition_method_name"] = "HIPPO"
            config = Config.from_dict(config_dict)
            single_run(config, save_to_file=True)

            # config_dict["acquisition_method_name"] = "KB"
            # config = Config.from_dict(config_dict)
            # single_run(config, save_to_file=True)


if __name__ == '__main__':
    # simple_1d()
    # gardner()
    # hartmann_ackley()
    # zdt3()
    # DTLZ2_3_objectives()
    # VLMOP2()
    # BraninGoldsteinPrice()
    # RosenbrokAlpine2()
    many_batches()
