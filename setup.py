#!/usr/bin/env python
#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import os

import setuptools


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), "r", encoding="utf-8") as fh:
        return fh.read()


setuptools.setup(
    name="pinecone-client",
    version=read("pinecone/__version__").strip(),
    description="Pinecone client and SDK",
    license="Proprietary License",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://www.pinecone.io/",
    project_urls={
        "Homepage": "https://www.pinecone.io",
        "Documentation": "https://docs.beta.pinecone.io/en/latest/index.html",
        "Contact": "https://www.pinecone.io/contact/",
        "End-User License Agreement": "https://www.pinecone.io/thin-client-user-agreement/"
    },
    author="Pinecone Systems, Inc.",
    author_email="support@pinecone.io",
    keywords="Pinecone vector database cloud",
    packages=setuptools.find_packages(),
    install_requires=read("requirements.txt"),
    include_package_data=True,
    python_requires=">=3.6",
    entry_points={
        'console_scripts': ['pinecone=pinecone.cli:main'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Database",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)
