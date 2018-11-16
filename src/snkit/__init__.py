"""snkit - a spatial networks toolkit
"""
import pkg_resources

__author__ = "Tom Russell"
__copyright__ = "Tom Russell"
__license__ = "mit"


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = 'unknown'
