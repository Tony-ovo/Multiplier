import unittest
import numpy as np
import torch
from signed88.hardware import get_design

class ModelEquivalenceTest(unittest.TestCase):
 def test_hard_forward_matches_numpy(self):
  for name in ['aggressive','fast','default','balanced','quality','area']:
   with self.subTest(name=name):
    d=get_design(name);model=d.build_model(d.spec.base_inits,0.999,0.0).cpu()
    with torch.no_grad(): value,_=model.forward_low_grid(c_init=1.0,c_out=1.0,hard_middle=True)
    got=value.cpu().numpy().round().astype(np.int32);expected=d.hard_low_numpy(d.spec.base_inits)
    self.assertTrue(np.array_equal(got,expected),f'{name}: maxdiff={np.max(np.abs(got-expected))}')
if __name__=='__main__':unittest.main()
