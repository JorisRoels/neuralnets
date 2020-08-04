import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuralnets",
    version="1.1.4.4",
    author="Joris Roels",
    author_email="jorisb.roels@ugent.be",
    description="A library, based on PyTorch, that implements basic neural network algorithms and useful functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saeyslab/neuralnets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "torchvision",
        "future",
        "tensorboard",
        "numpy",
        "tifffile",
        "h5py",
        "scikit-image",
        "opencv-python",
        "progress",
        "six",
        "sklearn"
    ]
)
