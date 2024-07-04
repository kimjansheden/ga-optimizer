from setuptools import setup, find_packages

setup(
    name="ga-optimizer-wrapper",
    version="0.1.0",
    description="Gradient Accumulation Optimizer Wrapper for Keras/TensorFlow",
    author="Kim Jansheden",
    author_email="kim.jansheden@gmail.com",
    url="https://github.com/kimjansheden/GAOptimizer",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
