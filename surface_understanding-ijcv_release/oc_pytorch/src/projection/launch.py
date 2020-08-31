from __future__ import print_function   
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from modules.stn import STN
from modules.stn import VTN
from modules.gridgen import CylinderGridGen, AffineGridGen, PersGridGen
from PIL import Image
from matplotlib import mlab
import matplotlib.pyplot as plt

g = PersGridGen(4,4,4).cuda()
input = Variable(torch.from_numpy(np.array([[[1, 0, 0,0], [0, 1, 0,0], [0, 0, 1, 0]]], dtype=np.float32)).cuda(), requires_grad = True)
out = g(input)
v = VTN().cuda()

input = Variable(torch.from_numpy(np.array([[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]], dtype=np.float32)).cuda(), requires_grad = True)
out2 = g(input)
print(out2.size())
input = Variable(torch.from_numpy(np.reshape(np.linspace(1,64,64), (1,4,4,4,1)).astype('float32'), 
                                   ),
         requires_grad = True).cuda()
res2 = v(input, out2)
res2, res1 = v.f.backward(res2.data)
print(out2[0,:,:,:,0].squeeze())
print(out2[0,:,:,:,1].squeeze())
print(out2[0,:,:,:,2].squeeze())
print(input[0,:,:,:].squeeze())
print(res2[0,:,:,:,:].squeeze())
