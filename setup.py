import setuptools

required = [
    "numpy>=1.21.6",
    "pandas>=1.1.0",
    "statsmodels>=0.13.0",
    "scikit-learn>=1.0.0",
]

extras = {
    'additional': [
        "holidays>=0.9",
        'prophet>=1.0.0',
        'fredapi',
        'tensorflow',
        'xgboost',
        'lightgbm',
        'joblib',
        'scipy',
        'arch',
        'numexpr',
        'bottleneck',
        'yfinance',
        'pytrends',
        'matplotlib',
        'requests',
    ],
    'mcp': [
        "holidays>=0.9",
        'fredapi',
        'lightgbm',
        'joblib',
        'scipy',
        'numexpr',
        'bottleneck',
        'yfinance',
        'pytrends',
        'matplotlib',
        'requests',
        'seaborn',
        "mcp>=1.0.0",
    ]
}

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autots",
    version="1.0.2",
    author="Colin Catlin",
    author_email="colin.catlin@syllepsis.live",
    description="Automated Time Series Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/winedarksea/AutoTS",
    packages=setuptools.find_packages(),
    license="MIT",
    include_package_data=True,
    install_requires=required,
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    entry_points={
        "console_scripts": [
            "autots-mcp = autots.mcp.server:serve",
        ],
    },
)
