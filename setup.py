"""Setup configuration for demand_forecasting_walmart package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="demand_forecasting_walmart",
    version="1.0.0",
    author="Mudit R",
    author_email="",
    description="Multi-model demand forecasting system for Walmart M5 dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mudit-R/demand-forecasting-walmart",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
