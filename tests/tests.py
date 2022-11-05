"""
.. module:: tests
:synopsis: tests of core borges entitys
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import os
import sys
sys.path.insert(0, os.path.join('..', 'borges'))
sys.path.insert(0, os.path.join('..', '..', 'borges'))

import borges.content as content

algorithm, parameters = content.create(
    configuration = {'general': {'gpu': True, 'seed': 4}},
    package = 'analyst',
    step = 'scale',
    step = 'normalize',
    parameters = {'copy': False})

print(algorithm, parameters)


