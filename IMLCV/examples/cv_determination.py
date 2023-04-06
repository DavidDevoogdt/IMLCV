import argparse
from importlib import import_module

from keras.api._v2 import keras as KerasAPI
from molmod.units import angstrom

keras: KerasAPI = import_module("tensorflow.keras")


from configs.config_general import ROOT_DIR, config
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerAutoEncoder, TranformerUMAP
from IMLCV.base.rounds import Rounds


def test_cv_discovery(
    name,
    cvd="AE",
    out_dim=3,
):
    from IMLCV.scheme import Scheme

    rnds = Rounds(folder=name, new_folder=False)
    scheme0 = Scheme.from_rounds(rnds)

    tf_kwargs = {
        "outdim": out_dim,
        "descriptor": "sb",
        "descriptor_kwargs": {
            "r_cut": 5 * angstrom,
            "sti": scheme0.rounds.round_information().tic,
            "n_max": 5,
            "l_max": 5,
        },
    }

    if cvd == "AE":
        tf = TranformerAutoEncoder(**tf_kwargs)

        kwargs = {}
    elif cvd == "UMAP":
        tf = TranformerUMAP(**tf_kwargs)

        kwargs = dict(
            n_neighbors=60,
            min_dist=0.8,
            nunits=200,
            nlayers=4,
            metric="l2",
            densmap=True,
            parametric_reconstruction=True,
            parametric_reconstruction_loss_fcn=keras.losses.MSE,
            decoder=True,
        )
    else:
        raise ValueError

    cvd = CVDiscovery(
        transformer=tf,
    )

    config()

    scheme0.update_CV(
        samples=1e3,
        cvd=cvd,
        **kwargs,
    )


if __name__ == "__main__":

    test_cv_discovery(
        name=ROOT_DIR / "IMLCV" / "examples" / "output" / "ala_02", cvd="AE"
    )

    raise

    parser = argparse.ArgumentParser(
        description="CLI interface to set simulation params",
        epilog="Question? ask David.Devoogdt@ugent.be",
    )

    parser.add_argument("-f", "--folder", type=str)

    parser.add_argument("-s", "--output_dim", type=int, default=3)
    parser.add_argument(
        "-m", "--method", type=str, default="UMAP", choices=["AE", "UMAP"]
    )

    group = parser.add_argument_group("AE")

    group = parser.add_argument_group("UMAP")

    args = parser.parse_args()
    args.folder = ROOT_DIR / "IMLCV" / "examples" / "output" / args.folder

    test_cv_discovery(name=args.folder, cvd=args.method)
