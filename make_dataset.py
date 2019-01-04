# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 19:23:11 2018

@author: Francisco
"""

import wfdb as wf
import numpy as np
from datasets import mitdb as mitdb
from matplotlib import pyplot as plt

def testbench():
    
    mitdb.create_datasets()
    
testbench()