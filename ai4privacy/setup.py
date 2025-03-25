from setuptools import setup, find_packages

setup(
    name="ai4privacy",
    version="0.1.3",
    author="Michael Anthony",
    author_email="developers@ai4privacy.com",
    description="A package to mask PII in text using transformers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MikeDoes/ai4privacy",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.0.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)