import setuptools

with open("README.md", "r", encoding="utf-8") as fhand:
    long_description = fhand.read()

setuptools.setup(
    name="dwi-pipeline",
    version="0.0.1",
    author="Atul Phadke",
    author_email="atulphadke8@gmail.com",
    description=("A command line utility that allows easy "
                "modifications on DWI volumes and Datasets."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AtulPhadke/pipelineRepo",
    project_urls={
        "Bug Tracker": "https://github.com/AtulPhadke/pipelineRepo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires= ["keyboard==0.13.5", "brukerapi==0.1.8",
    "dipy==1.6.0", "nibabel==5.0.0", "numpy==1.24.2",
    "matplotlib==3.6.3", "simpleitk==2.2.1", "tkscrolledframe==1.0.4",
    "PyQt5==5.15.9", "scipy==1.10.0"],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "pipeline = src.cli:main",
        ]
    }
)