from importlib.metadata import version

from pandas.core.frame import DataFrame
from spatialdata._core.spatialdata import SpatialData

from . import pl, pp, read, tl

__all__ = ["pl", "pp", "tl", "read"]

__version__ = version("troutpy")
