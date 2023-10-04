#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:24:25 2019

@author: thorsteinngj
"""

import os
source1 = "/home/thorsteinngj/Desktop/ims/Images/"
dest11 = "/home/thorsteinngj/Desktop/ims/val"
files = os.listdir(source1)
import shutil
import numpy as np
for f in files:
    if np.random.rand(1) < 0.2:
        shutil.move(source1 + '/'+ f, dest11 + '/'+ f)