import re
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("nerf/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)


with open("requirements.txt", "r") as f:
    reqs = [x.strip() for x in f.readlines()]


setuptools.setup(
    name="nerf",
    version=version,
    author="FateScript",
    author_email="wangfeng02@megvii.com",
    description="Nerf implemented by megengine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    install_requires=reqs,
)
