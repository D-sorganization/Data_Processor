"""
Setup script for Data Processor Pro.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent.parent.parent / "docs" / "data_processor_pro" / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="data-processor-pro",
    version="1.0.0",
    author="Data Processor Pro Team",
    author_email="support@dataprocessorpro.com",
    description="Professional-grade data processing and analysis platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/D-sorganization/Data_Processor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda11x>=12.0.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
            "mypy>=1.5.0",
            "black>=23.9.0",
            "ruff>=0.0.291",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-processor-pro=data_processor_pro.app:run",
        ],
    },
    include_package_data=True,
    package_data={
        "data_processor_pro": [
            "config/*.yaml",
            "ui/assets/*",
        ],
    },
    zip_safe=False,
    keywords="data-processing signal-processing data-analysis visualization polars numba",
    project_urls={
        "Bug Reports": "https://github.com/D-sorganization/Data_Processor/issues",
        "Source": "https://github.com/D-sorganization/Data_Processor",
        "Documentation": "https://github.com/D-sorganization/Data_Processor/tree/main/docs/data_processor_pro",
    },
)
