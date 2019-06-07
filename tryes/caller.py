#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:42:17 2019

@author: alessandro
"""

import os

for i in range(10):
    os.system('python3 prog.py %03d' % i)
    
print("all's done")