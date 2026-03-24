import os
from setuptools import setup, find_packages


def find_all_packages(root="vllm"):
    """Find all packages including namespace packages without __init__.py."""
    packages = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # Include any directory that contains .py files
        if any(f.endswith(".py") for f in filenames):
            pkg = dirpath.replace(os.sep, ".")
            packages.add(pkg)
    return sorted(packages)


setup(
    name="vllm",
    version="0.15.0",
    description="vLLM 0.15.0 + weighted LoRA chunked exact accumulation",
    packages=find_all_packages("vllm"),
    package_data={"": ["*.so", "*.pyd", "*.json", "*.jinja", "*.js", "*.css", "*.html"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "vllm=vllm.entrypoints.cli.main:main",
        ],
        "vllm.general_plugins": [
            "lora_filesystem_resolver=vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver",
        ],
    },
    python_requires=">=3.10,<3.14",
)
