#!/usr/bin/env python

import sys
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()

setup(name='scikit-sos',
      version='0.1.0',
      description='An sklearn-compatible Python implementation of Stochastic Outlier Selection (SOS) for detecting outliers in data.',
      long_description=README,
      author='Jeroen Janssens',
      author_email='jeroen@jeroenjanssens.com',
      url='https://github.com/jeroenjanssens/sos',
      license='BSD',
      package_dir={'sos': 'sksos'},
      packages=['sos'],
      data_files=[('sos', ['sksos/iris.csv'])],
      install_requires=['numpy'],

      include_package_data=True,
      zip_safe=False,
      entry_points={'console_scripts':
          ['sos=sos:main']
    }
)

