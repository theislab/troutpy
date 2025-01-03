from importlib.metadata import version

from . import pl, pp, tl,read

__all__ = ["pl", "pp", "tl","read"]

__version__ = version("troutpy")
