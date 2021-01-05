from setuptools import setup, find_packages

setup(
    name='sun-position',
    version='1.0.1',
    url='https://github.com/nastasi/sun-position',
    author='Matteo Nastasi',
    author_email='nastasi@alternativeoutput.it',
    description='Solar position algorithm for solar radiation applications',
    packages=find_packages(),    
    install_requires=['numpy >= 1.19.4'],
)
