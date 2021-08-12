"""Generic utilities
"""


def tqdm_standin(iterator, *_, **__):
    """Alternative to tqdm, with no progress bar - ignore any arguments after the first"""
    return iterator
