from setuptools import setup, find_packages

setup(
    name="xia_cog_models",
    version="0.1",
    packages=find_packages(),
    url="https://github.com/dddd1007/xia_cog_models",
    author="Xia Xiaokai",
    author_email="xia@xiaokai.me",
    description="My workflow of cognitive control models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=["numpy", "pandas", "matplotlib"],
)
