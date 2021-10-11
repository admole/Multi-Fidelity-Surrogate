#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pprint
import os
import numpy as np
import json
import shutil
import glob

from os import path
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict
from PyFoam.Basics.DataStructures import Vector
# from PyFoam.Basics.DataStructures import UnparsedList


class Case:

    def __init__(self, settings):
        self.Name = "test1"
        self.RASModel = "kOmegaSST"
        self.Geometry = {
            'Height': 4.0,
            'Width': 9.0,
            'Cube': 1.0,
            'Spacing': 4.0,
            'Length': 14.0,
            'Inlet': 2.0,
            'Offset': 0.0,
        }
        self.FlowAngle = 0.0
        self.Locations = {}
        self.Vertices = []
        self.Ncells = {
            'Nxi': 10,
            'Nxc': 40,
            'Nxb': 25,
            'Nxo': 20,
            'Nyl': 85,
            'Nyu': 15,
            'Nzo': 15,
            'Nzc': 40,
            'Nnc': 75,
            }
        self.Grading = {
            'ywall': 20.0,
            'yfar': 3.0,
            'xin': 0.4,
            'xc': 1,
            'xb': 1,
            'xo': 10,
            'zc': 1,
            'zr': 7.5,
            'zl': 1.0 / 7.5,
            'nc': 20,
            'ncm': 1.0 / 20,
        }

        for k, v in settings.items():
            if k == 'Geometry':
                self.Geometry.update(v)
            elif k == 'Ncells':
                self.Ncells.update(v)
            elif k == 'Grading':
                self.Grading.update(v)
            setattr(self, k, v)
            
        self.calc_locations()
        self.calc_vertices()

    def calc_locations(self):

        self.Locations = dict(
            z=dict(
                z0=0 - self.Geometry['Width'] / 2.0,
                z1=0 - 3.0 / 2.0 * self.Geometry['Cube'],
                z2=0 - self.Geometry['Cube'] / 2.0,
                z3=self.Geometry['Cube'] / 2.0,
                z4=3.0 / 2.0 * self.Geometry['Cube'],
                z5=self.Geometry['Width'] / 2.0,

                z12=0 - 3.0 / 2.0 * self.Geometry['Cube'] + self.Geometry['Offset'],
                z22=0 - self.Geometry['Cube'] / 2.0 + self.Geometry['Offset'],
                z32=self.Geometry['Cube'] / 2.0 + self.Geometry['Offset'],
                z42=3.0 / 2.0 * self.Geometry['Cube'] + self.Geometry['Offset'],
            ),
            x=dict(
                x0=0.0,
                x1=self.Geometry['Inlet'] - self.Geometry['Cube'],
                x2=self.Geometry['Inlet'],
                x3=self.Geometry['Inlet'] + self.Geometry['Cube'],
                x4=self.Geometry['Inlet'] + 2 * self.Geometry['Cube'],
                x5=self.Geometry['Inlet'] + self.Geometry['Spacing'],
                x6=self.Geometry['Inlet'] + self.Geometry['Cube'] + self.Geometry['Spacing'],
                x7=self.Geometry['Inlet'] + self.Geometry['Cube'] + self.Geometry['Spacing'] + self.Geometry['Cube'],
                x8=self.Geometry['Inlet'] + self.Geometry['Spacing'] + 3 * self.Geometry['Cube'],
                x9=self.Geometry['Length'],
            ),
            y=dict(
                y0=0.0,
                y1=self.Geometry['Cube'],
                y2=2 * self.Geometry['Cube'],
                y3=self.Geometry['Height'],
            ),
        )

    def calc_vertices(self):
        self.Vertices = [
            [self.Locations['x']['x0'], self.Locations['y']['y0'], self.Locations['z']['z0']],
            [self.Locations['x']['x1'], self.Locations['y']['y0'], self.Locations['z']['z0']],
            [self.Locations['x']['x4'], self.Locations['y']['y0'], self.Locations['z']['z0']],
            [self.Locations['x']['x5'], self.Locations['y']['y0'], self.Locations['z']['z0']],
            [self.Locations['x']['x8'], self.Locations['y']['y0'], self.Locations['z']['z0']],
            [self.Locations['x']['x9'], self.Locations['y']['y0'], self.Locations['z']['z0']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y2'], self.Locations['z']['z0']],
            [self.Locations['x']['x1'], self.Locations['y']['y2'], self.Locations['z']['z0']],
            [self.Locations['x']['x4'], self.Locations['y']['y2'], self.Locations['z']['z0']],
            [self.Locations['x']['x5'], self.Locations['y']['y2'], self.Locations['z']['z0']],
            [self.Locations['x']['x8'], self.Locations['y']['y2'], self.Locations['z']['z0']],
            [self.Locations['x']['x9'], self.Locations['y']['y2'], self.Locations['z']['z0']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y3'], self.Locations['z']['z0']],
            [self.Locations['x']['x1'], self.Locations['y']['y3'], self.Locations['z']['z0']],
            [self.Locations['x']['x4'], self.Locations['y']['y3'], self.Locations['z']['z0']],
            [self.Locations['x']['x5'], self.Locations['y']['y3'], self.Locations['z']['z0']],
            [self.Locations['x']['x8'], self.Locations['y']['y3'], self.Locations['z']['z0']],
            [self.Locations['x']['x9'], self.Locations['y']['y3'], self.Locations['z']['z0']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y0'], self.Locations['z']['z1']],
            [self.Locations['x']['x1'], self.Locations['y']['y0'], self.Locations['z']['z1']],
            [self.Locations['x']['x4'], self.Locations['y']['y0'], self.Locations['z']['z1']],
            [self.Locations['x']['x5'], self.Locations['y']['y0'], self.Locations['z']['z12']],
            [self.Locations['x']['x8'], self.Locations['y']['y0'], self.Locations['z']['z12']],
            [self.Locations['x']['x9'], self.Locations['y']['y0'], self.Locations['z']['z1']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y2'], self.Locations['z']['z1']],
            [self.Locations['x']['x1'], self.Locations['y']['y2'], self.Locations['z']['z1']],
            [self.Locations['x']['x4'], self.Locations['y']['y2'], self.Locations['z']['z1']],
            [self.Locations['x']['x5'], self.Locations['y']['y2'], self.Locations['z']['z12']],
            [self.Locations['x']['x8'], self.Locations['y']['y2'], self.Locations['z']['z12']],
            [self.Locations['x']['x9'], self.Locations['y']['y2'], self.Locations['z']['z1']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y3'], self.Locations['z']['z1']],
            [self.Locations['x']['x1'], self.Locations['y']['y3'], self.Locations['z']['z1']],
            [self.Locations['x']['x4'], self.Locations['y']['y3'], self.Locations['z']['z1']],
            [self.Locations['x']['x5'], self.Locations['y']['y3'], self.Locations['z']['z12']],
            [self.Locations['x']['x8'], self.Locations['y']['y3'], self.Locations['z']['z12']],
            [self.Locations['x']['x9'], self.Locations['y']['y3'], self.Locations['z']['z1']],
    
            [self.Locations['x']['x2'], self.Locations['y']['y0'], self.Locations['z']['z2']],
            [self.Locations['x']['x3'], self.Locations['y']['y0'], self.Locations['z']['z2']],
            [self.Locations['x']['x6'], self.Locations['y']['y0'], self.Locations['z']['z22']],
            [self.Locations['x']['x7'], self.Locations['y']['y0'], self.Locations['z']['z22']],
    
            [self.Locations['x']['x2'], self.Locations['y']['y1'], self.Locations['z']['z2']],
            [self.Locations['x']['x3'], self.Locations['y']['y1'], self.Locations['z']['z2']],
            [self.Locations['x']['x6'], self.Locations['y']['y1'], self.Locations['z']['z22']],
            [self.Locations['x']['x7'], self.Locations['y']['y1'], self.Locations['z']['z22']],
    
            [self.Locations['x']['x2'], self.Locations['y']['y0'], self.Locations['z']['z3']],
            [self.Locations['x']['x3'], self.Locations['y']['y0'], self.Locations['z']['z3']],
            [self.Locations['x']['x6'], self.Locations['y']['y0'], self.Locations['z']['z32']],
            [self.Locations['x']['x7'], self.Locations['y']['y0'], self.Locations['z']['z32']],
    
            [self.Locations['x']['x2'], self.Locations['y']['y1'], self.Locations['z']['z3']],
            [self.Locations['x']['x3'], self.Locations['y']['y1'], self.Locations['z']['z3']],
            [self.Locations['x']['x6'], self.Locations['y']['y1'], self.Locations['z']['z32']],
            [self.Locations['x']['x7'], self.Locations['y']['y1'], self.Locations['z']['z32']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y0'], self.Locations['z']['z4']],
            [self.Locations['x']['x1'], self.Locations['y']['y0'], self.Locations['z']['z4']],
            [self.Locations['x']['x4'], self.Locations['y']['y0'], self.Locations['z']['z4']],
            [self.Locations['x']['x5'], self.Locations['y']['y0'], self.Locations['z']['z42']],
            [self.Locations['x']['x8'], self.Locations['y']['y0'], self.Locations['z']['z42']],
            [self.Locations['x']['x9'], self.Locations['y']['y0'], self.Locations['z']['z4']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y2'], self.Locations['z']['z4']],
            [self.Locations['x']['x1'], self.Locations['y']['y2'], self.Locations['z']['z4']],
            [self.Locations['x']['x4'], self.Locations['y']['y2'], self.Locations['z']['z4']],
            [self.Locations['x']['x5'], self.Locations['y']['y2'], self.Locations['z']['z42']],
            [self.Locations['x']['x8'], self.Locations['y']['y2'], self.Locations['z']['z42']],
            [self.Locations['x']['x9'], self.Locations['y']['y2'], self.Locations['z']['z4']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y3'], self.Locations['z']['z4']],
            [self.Locations['x']['x1'], self.Locations['y']['y3'], self.Locations['z']['z4']],
            [self.Locations['x']['x4'], self.Locations['y']['y3'], self.Locations['z']['z4']],
            [self.Locations['x']['x5'], self.Locations['y']['y3'], self.Locations['z']['z42']],
            [self.Locations['x']['x8'], self.Locations['y']['y3'], self.Locations['z']['z42']],
            [self.Locations['x']['x9'], self.Locations['y']['y3'], self.Locations['z']['z4']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y0'], self.Locations['z']['z5']],
            [self.Locations['x']['x1'], self.Locations['y']['y0'], self.Locations['z']['z5']],
            [self.Locations['x']['x4'], self.Locations['y']['y0'], self.Locations['z']['z5']],
            [self.Locations['x']['x5'], self.Locations['y']['y0'], self.Locations['z']['z5']],
            [self.Locations['x']['x8'], self.Locations['y']['y0'], self.Locations['z']['z5']],
            [self.Locations['x']['x9'], self.Locations['y']['y0'], self.Locations['z']['z5']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y2'], self.Locations['z']['z5']],
            [self.Locations['x']['x1'], self.Locations['y']['y2'], self.Locations['z']['z5']],
            [self.Locations['x']['x4'], self.Locations['y']['y2'], self.Locations['z']['z5']],
            [self.Locations['x']['x5'], self.Locations['y']['y2'], self.Locations['z']['z5']],
            [self.Locations['x']['x8'], self.Locations['y']['y2'], self.Locations['z']['z5']],
            [self.Locations['x']['x9'], self.Locations['y']['y2'], self.Locations['z']['z5']],
    
            [self.Locations['x']['x0'], self.Locations['y']['y3'], self.Locations['z']['z5']],
            [self.Locations['x']['x1'], self.Locations['y']['y3'], self.Locations['z']['z5']],
            [self.Locations['x']['x4'], self.Locations['y']['y3'], self.Locations['z']['z5']],
            [self.Locations['x']['x5'], self.Locations['y']['y3'], self.Locations['z']['z5']],
            [self.Locations['x']['x8'], self.Locations['y']['y3'], self.Locations['z']['z5']],
            [self.Locations['x']['x9'], self.Locations['y']['y3'], self.Locations['z']['z5']],
        ]

    def setup_blockmesh(self):
        bm = ParsedBlockMeshDict(path.join(SolutionDirectory(self.Name).systemDir(), "blockMeshDict"))
        for key in bm:
            for keys in self.Geometry:
                if key == keys:
                    bm[key] = self.Geometry[keys]
            for keys in self.Locations['x']:
                if key == keys:
                    bm[key] = self.Locations['x'][keys]
            for keys in self.Locations['y']:
                if key == keys:
                    bm[key] = self.Locations['y'][keys]
            for keys in self.Locations['z']:
                if key == keys:
                    bm[key] = self.Locations['z'][keys]
            for keys in self.Ncells:
                if key == keys:
                    bm[key] = self.Ncells[keys]
            for keys in self.Grading:
                if key == keys:
                    bm[key] = self.Grading[keys]
        print(f"Updating blockMeshDict")
        bm.writeFile()

    def set_model(self):
        turbfile = ParsedParameterFile(path.join(SolutionDirectory(self.Name).constantDir(), "turbulenceProperties"))
        print(f"Setting RANS model as {self.RASModel}")
        turbfile['RAS']['RASModel'] = self.RASModel
        turbfile.writeFile()

    def create_case(self):
        print(f"\nSetting up {self.Name}")
        if self.Name not in glob.glob(path.basename(path.normpath(path.join(os.getcwd(), self.Name)))):
            print(f"Copying base case to {self.Name}")
            shutil.copytree("Base", self.Name)
        else:
            print(f"{self.Name} already exists\nModifying current case")
        self.set_model()
        self.setup_blockmesh()


# Opening JSON file
j = open(path.join(os.getcwd(), "cases.json"))  # Check this works elsewhere
case_settings = json.load(j)

cases = []
for i in range(len(case_settings)):
    cases.append(Case(case_settings[i]))
    cases[i].create_case()

