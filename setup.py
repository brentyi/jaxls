from setuptools import setup

setup(
    name="jaxfg",
    version="0.0",
    description="Factor graphs in Jax",
    url="http://github.com/brentyi/jaxfg",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=["jaxfg"],
    install_requires=[
        "jax",
        "jaxlib",
        "overrides",
    ],
)
