from setuptools import setup, find_packages

setup(
    name='DependentlyRandomForest',
    version='0.1',
    packages=find_packages(),
    description='A modified Random Forest with a new parameter added',
    long_description=open('README.md').read(),
    author='Quoc Viet Le',
    author_email='quocvietlework@gmail.com',
    license='',
    install_requires=[
        # Appropriate Python versions
    ],
)
