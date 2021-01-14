#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:31:37 2019

@author: aayush
"""

import resnet
import Searcher
import argparse
import cv2


# creating the argument parser and parsing the arguments
#intializing the color descriptor
cd = resnet.ResNetFeat()
#loading the query image and describe it

queryFeatures = cd.getFeatureQuery('queries/test4.jpg')
#performing the search
s1 = Searcher.Searcher('output.csv')
results = s1.search(queryFeatures)
query = cv2.imread('queries/test4.jpg')
query2 = cv2.resize(query,(300,300))
cv2.imshow("query",query2)
#displaying the query
#loop over the results
label = []
for i in range(4):
    label.append('rs'+str(i))
i = 0
for (score, resultID) in results:
    #load the result image and display it
    print(resultID)
    result1 = cv2.imread(resultID)
    result = cv2.resize(result1,(300,300))
    cv2.imshow(label[i],result)
    i = i+1
cv2.waitKey(0)

