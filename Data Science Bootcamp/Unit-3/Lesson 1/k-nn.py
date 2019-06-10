# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:39:33 2019

@author: clyde
"""

k=7
import numpy as np
d = np.sqrt((music.duration - test.duration)**2 + (music.loudness-test.loudness)**2)
f = d.sort_values()
pred1 = len((np.where(music.jazz[f.index[1:k]]==1))[0])
pred2 = len((np.where(music.jazz[f.index[1:k]]==0))[0])
z = [pred2/k,pred1/k]