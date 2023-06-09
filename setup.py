#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup snkit package
"""
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages
from setuptools import setup


def readme():
    """Read README contents"""
    with open("README.md", encoding="utf8") as f:
        return f.read()


setup(
    name="snkit",
    use_scm_version=True,
    license="MIT License",
    description="a spatial networks toolkit",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Tom Russell",
    author_email="tomalrussell@gmail.com",
    url="https://github.com/tomalrussell/snkit",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Utilities",
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    setup_requires=["setuptools_scm"],
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
        "geopandas>=0.13",
        "shapely>=2.0",
    ],
    extras_require={
        "dev": ["black", "nbstripout", "pytest", "pytest-cov"],
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
        "networkx": ["networkx>=3.0"],
    },
    entry_points={
        "console_scripts": [
            # eg: 'snkit = snkit.cli:main',
        ]
    },
)
