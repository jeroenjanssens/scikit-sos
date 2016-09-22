#!/usr/bin/env python

import sys
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()

setup(name='sos',
      version='0.1.0',
      description='Stochastic Outlier Selection',
      long_description=README,
      author='Jeroen Janssens',
      author_email='',
      url='https://github.com/jeroenjanssens/sos',
      license='',
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


