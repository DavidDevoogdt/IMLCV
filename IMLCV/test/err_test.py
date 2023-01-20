# from IMLCV.base.MdEngine import SystemParams
import jax.numpy as jnp


class EnergyError(Exception):
    pass


class AseError(EnergyError):
    pass


def handle():
    try:
        raise AssertionError
    except:
        raise AseError("yeet")


class myclass:
    def __init__(self) -> None:
        self.sp = (jnp.zeros((22, 3)), None)

    def run(self):
        try:
            return handle()
        except EnergyError as be:
            raise EnergyError(
                f"""
An error occured during the energy calculation {self.__class__}
The lates coordinates were {self.sp}.                  
raised exception from calculator:{be}"""
            )


if __name__ == "__main__":
    a = myclass()
    try:
        a.run()
    except EnergyError as e:
        print(e)

    print("finished gracefully")
