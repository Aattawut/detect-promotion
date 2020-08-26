#!/usr/bin/python
#-*-coding: utf-8 -*-
##from __future__ import absolute_import
######

from botnoi import scrape as sc
from botnoi import cv
import os
import glob
import pandas as pd
import pickle
import numpy as np

mymod = pickle.load(open('mymodel.p','rb'))
def predictimg(imgurl):
  a = cv.image(imgurl)
  feat = a.getresnet50()
  probList = mymod.predict_proba([feat])[0]
  maxprobind = np.argmax(probList)
  prob = probList[maxprobind]
  outclass = mymod.classes_[maxprobind]
  result = {}
  result['class'] = outclass
  result['probability'] = prob
  return result