import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="connoFramer",
    version="0.00.01",
    author="Maria Antoniak",
    author_email="mariaa@allenai.org",
    description="A toolkit for using verb lexicons like connotation frames",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maartensap/connotationFramer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)