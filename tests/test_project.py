"""
.. module:: project test
:synopsis: tests Theory class
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from pathlib import pathlib.Path

import pandas as pd
import pytest

from borges.core.idea import Idea
from borges.core.project import Theory


def test_project():
    idea = Idea(
        configuration = pathlib.Path.cwd().joinpath('tests', 'idea_settings.ini'))
    borges =Theory(idea = idea)
    print('test', project.library)
    return

if __name__ == '__main__':
    test_project()