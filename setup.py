from setuptools import setup, find_packages

setup(
    name='dexter',
    version='0.0.',
    description='A package for standardised experiment workflows',
    author='Alex Alvarez Perez',
    author_email='fa.alvarezperez@hotmail.com',
    url='https://github.com/aalvarezperez/Dexter',
    license='GPL',
    install_requires=[
        'pandas>=1.3.5',
        'numpy>=1.21.1',
        'scipy>=1.7.1',
        'seaborn>=0.10.1',
        'tabulate>=0.8.9',
        'pingouin=0.5.0',
        'matplotlib=3.5'
    ],
    packages=find_packages(exclude=('tests*'))
)
