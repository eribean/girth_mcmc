from setuptools import setup, convert_path

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


print(convert_path('/performance'))
if __name__ == '__main__':
    setup(
        name="girth_mcmc", 
        packages=['girth_mcmc', 'girth_mcmc.dichotomous', 'girth_mcmc.polytomous', 
                  'girth_mcmc.utils'],
        package_dir={'girth_mcmc': 'girth_mcmc'},
        version="0.1.0",
        license="MIT",
        description="MCMC Item Response Theory Estimation.",
        long_description=long_description.replace('<ins>','').replace('</ins>',''),
        long_description_content_type='text/markdown',
        author='Ryan C. Sanchez',
        author_email='rsanchez44@gatech.edu',
        url = 'https://eribean.github.io/girth/',
        download_url = 'https://github.com/eribean/girth/archive/v0.3.7.tar.gz',
        keywords = ['IRT', 'Psychometrics', 'Item Response Theory'],
        install_requires = ['numpy', 'scipy', 'pymc3'],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering', 
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8'
        ]
    )