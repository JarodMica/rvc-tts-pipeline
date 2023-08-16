from setuptools import setup, find_packages

# Read the requirements.txt file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='rvc_tts_pipe',
    version='0.1',
    description='tts-pipeline',
    author='Jarod Mica',
    packages=find_packages(),
    install_requires=[
        'rvc @ git+https://github.com/JarodMica/rvc.git#egg=rvc',
    ],
)