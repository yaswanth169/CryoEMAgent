from setuptools import setup, find_packages

setup(
    name="cryoemagent",
    version="0.2.0",
    description="Autonomous AI Agent for Cryo-EM GPCR Structure Determination",
    author="CryoEM Research Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "cryosparc-tools~=4.7.0",
        "openai>=1.0.0",
        "anthropic>=0.25.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "click>=8.0.0",
        "mrcfile>=1.4.0",
        "starfile>=0.5.0",
    ],
    entry_points={
        "console_scripts": [
            "cryoem-agent=cryoemagent.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
