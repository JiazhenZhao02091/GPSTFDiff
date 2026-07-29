from setuptools import setup, find_packages


setup(
    name='stf',
    packages=find_packages(exclude=('config*', 'scripts*', 'tools*')),
    # version='0.1.0',
)  # 创建库的名称  # 版本号
