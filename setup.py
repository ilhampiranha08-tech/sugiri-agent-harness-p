#!/usr/bin/env python3
"""Sugiri — AI Coding Agent. Copyright (c) 2025 Ilham Sugiri. MIT License."""

from setuptools import setup, find_packages

setup(
    name="sugiri",
    version="1.2.3",
    description="Sugiri — AI Coding Agent, created by Ilham Sugiri",
    author="Ilham Sugiri",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["cli"],
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.27",
        "rich>=13.0",
        "click>=8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8",
            "pytest-asyncio>=0.23",
        ],
        "images": [
            "Pillow>=10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sugiri=cli:cli",
        ],
    },
)
