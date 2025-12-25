# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from setuptools import find_packages, setup

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = {}  # type: ignore
with open("llmlangstral/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

INSTALL_REQUIRES = [
    "transformers>=4.40.0",  # Support Mistral 3
    "accelerate>=0.27.0",
    "torch>=2.1.0",
    "sentencepiece",  # Tokenizer Mistral
    "protobuf",
    "nltk",
    "numpy",
    "tiktoken",  # Gardé pour compatibilité OpenAI
]
QUANLITY_REQUIRES = [
    "black==21.4b0",
    "flake8>=3.8.3",
    "isort>=5.5.4",
    "pre-commit",
    "pytest",
    "pytest-xdist",
]
DEV_REQUIRES = INSTALL_REQUIRES + QUANLITY_REQUIRES

setup(
    name="llmlangstral",
    version=VERSION["VERSION"],
    author="The LLMLangstral team",
    author_email="",
    description="Compress prompts up to 20x while preserving semantic information for LLMs. Fork of Microsoft LLMLingua with Mistral AI models.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    keywords="Prompt Compression, LLMs, Inference Acceleration, Mistral AI, Efficient LLMs",
    license="MIT License",
    url="https://github.com/sauron/LLMLangstral",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "."},
    packages=find_packages("."),
    extras_require={
        "dev": DEV_REQUIRES,
        "quality": QUANLITY_REQUIRES,
    },
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    python_requires=">=3.8.0",
    zip_safe=False,
)
