from setuptools import setup, find_packages

setup(
    name="mogonet",  # Name of your package (must be unique on PyPI)
    version="0.1.0",  # Version of your package
    author="Lamine TOURE",
    author_email="laminetoure626@gmail.com",
    description="A Python package for MOGONET-related tasks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/laminetourelab/mogonet",
    packages=find_packages(),  # Automatically finds all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Minimum Python version required
    install_requires=[  # List of dependencies
        "numpy>=1.20.0",
        "scikit-learn>=1.0",
        "pandas>=1.3.0",
        "torch>=1.10.0",  
        "seaborn>=0.11.0",  
    ],
    extras_require={  # Optional dependencies
        "dev": ["pytest", "flake8", "twine"],
    },
    entry_points={  # Optional: Define command-line scripts
        "console_scripts": [
            "main-biomarker=mogonet.scripts.main_biomarker:main",
            "main-mogonet=mogonet.scripts.main_mogonet:main",
        ],
    },
    include_package_data=True,  # Include additional data files if needed
)
