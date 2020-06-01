import pathlib
import re
import setuptools

__packagename__ = 'pyrff'
ROOT = pathlib.Path(__file__).parent


def get_version():
    VERSIONFILE = pathlib.Path(ROOT, __packagename__, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise Exception('Unable to find version string in %s.' % (VERSIONFILE,))


__version__ = get_version()


setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version=__version__,
    description='Implementation of random fourier feature (RFF) approximations and Thompson sampling.',
    license='AGPLv3',
    long_description=open(pathlib.Path(ROOT, 'README.md')).read(),
    long_description_content_type='text/markdown',
    url='https://github.com/michaelosthege/pyrff',
    author='Michael Osthege',
    author_email='michael.osthege@outlook.com',
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    install_requires=[
        open(pathlib.Path(ROOT, 'requirements.txt')).readlines()
    ],
    python_requires='>=3.6'
)
