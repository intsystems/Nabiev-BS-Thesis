import io
import re
from setuptools import setup, find_packages

from mylib import __version__


def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('../docs/source/index.md')
# clearing local versions
# https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$', read('requirements.txt'),
               flags=re.MULTILINE))

setup(
    # metadata
    name='mylib',
    version=__version__,
    license='MIT',
    author='Greg Psoi',
    author_email="psoi.greg@gmail.com",
    description='mylib, python package',
    long_description=readme,
    url='https://github.com/intsystems/SoftwareTemplate-simplified',
    # options
    packages=find_packages(),
    install_requires=requirements,
)
