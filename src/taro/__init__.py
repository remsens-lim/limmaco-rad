
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = "limmaco-rad_" + version("limmaco-rad")
except PackageNotFoundError:
    # package is not installed
    pass

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
