# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database
from resnet import ResNetFeat

depth = 5
d_type = 'd1'
query_idx = 0

if __name__ == '__main__':
  db = Database()
  
  # retrieve by resnet
  method = ResNetFeat()
  query = method.getFeatureQuery('queries/image_06736.jpg')
  samples = method.make_samples(db)
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)
