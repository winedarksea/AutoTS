import setuptools

required = [
    "numpy", "pandas>=0.23.*", "statsmodels>=0.9.*", "scikit-learn>=0.20.*","holidays>=0.9"
]

extras = {
    'additional' : ['fbprophet>=0.3.*', 'fredapi']
}

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoTS",
    version="0.0.3",
    author="Colin Catlin",
    author_email="colin.catlin@syllepsis.live",
    description="Automated Time Series Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/winedarksea/AutoTS",
    packages=setuptools.find_packages(),
    license="MIT",
    include_package_data=True,
    install_requires = required,
    extras_require = extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)