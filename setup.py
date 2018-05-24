from pathlib import Path

from setuptools import find_packages, setup

requirements = Path(__file__).parent / 'requirements.txt'

with requirements.open(mode='rt', encoding='utf-8') as fp:
    install_requires = [line.strip() for line in fp]

readme = Path(__file__).parent / 'README.rst'

with readme.open(mode='rt', encoding='utf-8') as fp:
    readme_text = fp.read()

VERSION = '0.0.1'

setup(
    name='numerical_algorithms',
    maintainer='Luka Shostenko',
    maintainer_email='luka.shostenko@gmail.com',
    version='{version}'.format(
        version=VERSION,
    ),
    download_url='https://github.com/wikibusiness/lexrank/archive/{version}.tar.gz'.format(  # noqa
        version=VERSION,
    ),
    description='Optimized numerical routines.',
    long_description=readme_text,
    license='Apa',
    author='Luka Shostenko',
    author_email='luka.shostenko@gmail.com',
    packages=find_packages(include=['numerical_algorithms.*']),
    python_requires='>=3.5.0',
    install_requires=install_requires,
    include_package_data=True,
)
