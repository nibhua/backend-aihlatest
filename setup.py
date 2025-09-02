# D:\ADOBE\core\setup.py
from setuptools import setup, find_packages

setup(
    name="core",
    version="0.1.0",
    packages=find_packages(include=["outline_extractor", "outline_extractor.*"]),
)
