from IMLCV.external.parsl_conf.config import config


def do_conf():
    config(cluster="doduo", max_blocks=10)
