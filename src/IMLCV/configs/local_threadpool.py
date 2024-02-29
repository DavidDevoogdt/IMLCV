from parsl.executors import ThreadPoolExecutor


def get_config(path_internal, ref_threads=2):
    executors = [
        ThreadPoolExecutor(
            label="training",
            max_threads=1,
            working_dir=str(path_internal),
        ),
        ThreadPoolExecutor(
            label="default",
            max_threads=1,
            working_dir=str(path_internal),
        ),
        ThreadPoolExecutor(
            label="reference",
            max_threads=ref_threads,
            working_dir=str(path_internal),
        ),
    ]

    return executors, [["default"], ["trainig"], ["reference"]]
