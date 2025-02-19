from setuptools import find_packages, setup

setup(
    name="review-bot-detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.7",
        "spacy>=3.4",
        "fastapi>=0.85.0",
        "mlflow>=2.0.0",
    ],
)
