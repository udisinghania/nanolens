from setuptools import setup, find_packages

setup(
    name="nanolens",
    version="0.1.0",
    author="Udit Singhania",
    description="A lightweight, fault-tolerant PyTorch hook to detect variance collapse in continuous world models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)