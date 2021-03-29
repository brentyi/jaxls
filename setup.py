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
    package_data={"jaxlie": ["py.typed"]},
    install_requires=[
        "datargs",
        "jax",
        "jaxlib",
        "jaxlie>=1.0.0",
        "overrides",
        "termcolor",
        "tqdm",
        "matplotlib",
    ],
    extras_require={
        "testing": [
            "pytest",
            # "pytest-cov",
            # "hypothesis",
            # "hypothesis[numpy]",
        ]
    },
)
