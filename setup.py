from setuptools import find_packages, setup

setup(
    name="jaxfg",
    version="0.0",
    description="Factor graphs in Jax",
    url="http://github.com/brentyi/jaxfg",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=find_packages(),
    package_data={"jaxfg": ["py.typed"]},
    python_requires=">=3.7",
    install_requires=[
        "dcargs",
        "frozendict",
        "jax>=0.2.13",
        "jaxlib",
        "jaxlie>=1.0.0",
        "jax_dataclasses>=1.0.0",
        "overrides",
        "scikit-sparse",
        "termcolor",
        "tqdm",
        "typing_utils",  # We can phase this out if we drop support for Python 3.7
        "matplotlib",
    ],
    extras_require={
        "testing": [
            "pytest",
            "pytest-cov",
            # "hypothesis",
            # "hypothesis[numpy]",
        ],
        "type-checking": [
            "mypy",
            "types-termcolor",
        ],
    },
)
