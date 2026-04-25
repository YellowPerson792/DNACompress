from setuptools import setup

setup(
    name='nugget',
    version='0.1.2',
    packages=['nugget', 'nugget.adaptors', 'nugget.utils', 'nugget.inspect'],
    install_requires=['torch', 'transformers>=4.41,<4.42'],
    url='https://github.com/hiaoxui/nugget',
    license='MIT',
    author='Guanghui Qin',
    description='Nugget text representation',
)
