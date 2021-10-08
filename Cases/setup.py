#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pprint
import os
import numpy as np
import json
import shutil

from os import path
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict
from PyFoam.Basics.DataStructures import Vector
# from PyFoam.Basics.DataStructures import UnparsedList


def create_case(case):
    shutil.copytree("Base", case['name'])
    set_model(case)


def set_model(case):
    turbfile = ParsedParameterFile(path.join(SolutionDirectory(case['name']).constantDir(), "turbulenceProperties"))
    turbfile['RAS']['RASModel'] = case['RASModel']
    # turbfile["actions"].clear()
    # turbfile["actions"].append(haloDict)
    turbfile.writeFile()


# Opening JSON file
j = open(path.join(os.getcwd(), "cases.json"))  # Check this works elsewhere
cases = json.load(j)

for i in range(len(cases)):
    print(f"Setting up {cases[i]['name']}")
    create_case(cases[i])

