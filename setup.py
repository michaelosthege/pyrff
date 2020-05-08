import setuptools

__packagename__ = 'pyrff'

def get_version():
    import pathlib, re
    VERSIONFILE = pathlib.Path(pathlib.Path(__file__).parent, __packagename__, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise Exception('Unable to find version string in %s.' % (VERSIONFILE,))

__version__ = get_version()


setuptools.setup(name = __packagename__,
        packages = setuptools.find_packages(), # this must be the same as the name above
        version=__version__,
        description='Implementation of random fourier feature (RFF) approximations and Thompson sampling.',
        url='https://github.com/michaelosthege/pyrff',
        #download_url = '%s' % __version__,
        author='Michael Osthege',
        author_email='michael.osthege@outlook.com',
        classifiers= [
            'Programming Language :: Python',
            'Operating System :: OS Independent',
            'License :: OSI Approved :: GNU Affero General Public License v3',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Mathematics'
        ],
        install_requires=[
            # via requirements.txt
        ],
        python_requires='>=3.6'
)
