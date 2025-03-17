from setuptools import setup, find_packages


setup(
    name="entropy-analyzer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "python-dotenv==1.0.1",
        "pydantic==2.10.6",
        "pydantic-settings==2.7.1",
        "pydantic_core==2.27.2",
        "rich==13.9.4",
        "openai>=1.66.3",
        "openai-agents>=0.0.4",
        "setuptools==75.8.0",
    ],
    author="Aditya Patange (AdiPat)",
    author_email="contact.adityapatange@gmail.com",
    description="Perform a complete, and detailed entropy analysis on your data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thehackersplaybook/entropy-analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license_files=("LICENSE",),
)
