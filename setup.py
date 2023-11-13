#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup snkit package
"""
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages
from setuptools import setup


def readme() -> str:
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
        "geopandas>=0.13",
        "shapely>=2.0",
    ],
    extras_require={
        "dev": [
            "black",
            "mypy",
            "nbstripout",
            "pre-commit",
            "pytest",
            "pytest-cov",
            "ruff",
        ],
        "docs": ["sphinx", "m2r2"],
        "networkx": ["networkx>=3.0"],
    },
    entry_points={
        "console_scripts": [
            # eg: 'snkit = snkit.cli:main',
        ]
    },
)
