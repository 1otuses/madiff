from setuptools import find_packages, setup

setup(
    name="madiff",
    description="Multi-agent Diffusion Model.",
    packages=find_packages(include=["diffuser*", "mode_consistent*"]),
)
