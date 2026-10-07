from setuptools import setup, find_packages


setup(
    name='src',
    packages=find_packages(exclude=('config*', 'scripts*', 'tools*')),

)
