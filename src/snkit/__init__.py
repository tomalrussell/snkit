"""snkit - a spatial networks toolkit
"""
import pkg_resources

# Define what is accessible directly on snkit, when a client writes::
#   from snkit import Network
from snkit.network import Network

__author__ = "Tom Russell"
__copyright__ = "Tom Russell"
__license__ = "mit"


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    __version__ = 'unknown'


# Define what should be imported as * when a client writes::
#   from snkit import *
__all__ = ['Network']
