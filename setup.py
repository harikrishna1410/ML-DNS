from setuptools import setup, find_packages

setup(
    name="ml_dns",
    version="0.1.0",
    author="Harikrishna Tummalapalli",
    author_email="harikrishnatummalapalli@gmail.com",
    description="A machine learning-enhanced Direct Numerical Simulation solver",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/harikrishna1410/ML-DNS",
    packages=["ml_dns"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords='machine learning, DNS, fluid dynamics, combustion',
    python_requires='>=3.10',
    install_requires=[
        'torch',
        'numpy',
        'mpi4py',
        'h5py',
        'matplotlib',
    ],
    extras_require={
    },
)