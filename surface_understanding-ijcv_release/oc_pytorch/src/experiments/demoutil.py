# Demo Util : taken from drc : https://github.com/shubhtuls/drc/blob/master/demo/
import torch
import os
import numpy as np

M = {}
cubeV = torch.Tensor([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
cubeV = 2 * cubeV - 1.
cubeF = torch.Tensor([[1,7,5], [1,3,7], [1,4,3], [1,2,4], [3,8,7], [3,4,8], [5,7,8], [5,8,6], [1,5,6], [1,6,2], [2,6,8], [2,8,4]])

def voxelsToMesh(predVol, thresh):
	vcounter = 0
	fcounter = 0
	totPoints = int((predVol > thresh).float().sum().data.cpu().item())
	vAll = cubeV.repeat(totPoints, 1)
	fAll = cubeF.repeat(totPoints, 1)

	fOffset = (torch.linspace(0,12*totPoints-1, 12*totPoints))
	fOffset = fOffset.repeat(3, 1).transpose(0,1)

	fOffset = (fOffset / 12).floor() * 8
	fAll = fAll + fOffset
	for x in range(0, predVol.size(0)):
		print(x, predVol.size(0))
		for y in range(0, predVol.size(1)):
			for z in range(0, predVol.size(2)):
				if predVol[x,y,z] > thresh:
					vAll[vcounter:vcounter+8,0] += x
					vAll[vcounter:vcounter+8,1] += y
					vAll[vcounter:vcounter+8,2] += z
					vcounter = vcounter + 8
					fcounter = fcounter+12

	return vAll / 57. -0.5, fAll

import vtk
from numpy import random

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')



def renderpointcloud(pointcloud, file, suffix):
	if not os.path.exists(file):
		os.makedirs(file)

	np.save(file + suffix, pointcloud.cpu().data.numpy())


def writeObj(meshfile, vertices, faces):
	mtlfilename = meshfile.split('.obj')[0] + '.mtl'
	with open(meshfile, 'w') as meshfilehandle:
		cval = 0.2
		with open(mtlfilename, 'w') as mtlfile:
			mtlfile.write("newmtl m0\n")
			mtlfile.write('Ka %f %f %f\n' % (cval, cval, cval))
			mtlfile.write('Kd %f %f %f\n' % (cval, cval, cval))
			mtlfile.write('Ks 1 1 1\n')
			mtlfile.write('illum %d\n' % 9)

		mtlfile = mtlfilename.split('/')[-1]
		meshfilehandle.write('mtllib %s\n usemtl m0\n' % mtlfile)
		for vx in range(0, vertices.shape[0]):
			meshfilehandle.write('v %f %f %f\n' % (vertices[vx,0], vertices[vx,1], vertices[vx,2]))

		for fx in range(0, faces.shape[0]):
			meshfilehandle.write('f %d %d %d\n' % (faces[fx, 0]+1, faces[fx, 1]+1, faces[fx, 2]+1))



def renderMesh(blenderExec, blendFile, meshFile, pngFile):
	command = 'bash oc_pytorch/src/experiments/renderer/render.sh %s %s %s %s' % (blenderExec, blendFile, meshFile, pngFile)
	print(command)
	os.system(command)
