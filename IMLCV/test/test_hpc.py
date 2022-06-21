
import parsl
from IMLCV.launch.parsl_conf.config import config

if __name__ == '__main__':
    print("starting on slacking cluster")

    config(cluster='slaking',spawnjob=True)

    @parsl.bash_app
    def test_scheme(outputs=[], stdout='hpc.stdout', stderr='hpc.stderr'):

        return f"python -u /user/gent/436/vsc43693/scratch_vo/projects/IMLCV/IMLCV/test/test_scheme.py >> {outputs[0].filepath}"

    future = test_scheme(outputs=[parsl.File("./test_scheme.out")])
    future.outputs[0].result()
