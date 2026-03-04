from setuptools import setup, find_packages

setup(
    name="quantlib-mm",
    version="0.1.0",
    description="Python quantitative finance library — Monte Carlo, option pricing, portfolio optimization, and risk analysis",
    author="Maisy Mylod",
    author_email="maisymylod@gmail.com",
    url="https://github.com/maisymylod/quantlib-mm",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "matplotlib>=3.7.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
