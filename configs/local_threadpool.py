from parsl.executors import ThreadPoolExecutor


def get_config(path_internal, py_env):
    executors = [
        ThreadPoolExecutor(
            label="training",
            max_threads=1,
            working_dir=str(path_internal),
        ),
        ThreadPoolExecutor(
            label="default", max_threads=1, working_dir=str(path_internal)
        ),
        ThreadPoolExecutor(
            label="model", max_threads=4, working_dir=str(path_internal)
        ),
        ThreadPoolExecutor(
            label="reference", max_threads=4, working_dir=str(path_internal)
        ),
    ]

    return executors
