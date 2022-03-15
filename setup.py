from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='lmpy',
      version='0.0.1',
      description='Yet another Python toolbox for Linear Regression.',
      author='Ziwei Huang',
      author_email='huang-ziwei@outlook.com',
      install_requires=required,
     )
