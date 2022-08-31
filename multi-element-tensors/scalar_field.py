import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time

import tensor_elements as elements
import frames
import fields
import state_bundle as bundles

from typing import List

def scalar_field(
	N = 8,
	animationDuration = 2.,
	simDuration = 3.0,
	dt = 0.1,
	display_dx = 0.1):

	t0 = time.time()

	xData = np.arange(-1., 1. + display_dx / 2, display_dx)
	yData = np.arange(-1., 1. + display_dx / 2, display_dx)
	meshX, meshY = np.meshgrid(xData, yData)
	tData = np.arange(0, simDuration, dt)

	###############################################################
	##################         grid data         ##################
	###############################################################

	globalFrame = frames.CoordinateFrame('global', 2)
	globalToGlobal = globalFrame.getIdentityTransform()

	centralGrid = fields.GridElement((N, N), globalFrame)
	grid = fields.Grid([globalToGlobal], [centralGrid])

	###############################################################
	##################        metric data        ##################
	###############################################################

	def metricFunction(x: frames.CoordinatePos):
		x.inFrame(globalFrame)
		return frames.FramedTensor(np.identity(2), x, 0)

	def inverseMetricFunction(x: frames.CoordinatePos):
		x.inFrame(globalFrame)
		return frames.FramedTensor(np.identity(2), x, 2)

	metric = fields.TensorField.functionInit(grid, metricFunction)
	gradMetric = metric.commaGrad()
	invMetric = fields.TensorField.functionInit(grid, inverseMetricFunction)

	christoffel_lower = fields.TensorField.defaultInit(grid, gradMetric.tensorShape)
	christoffel_lower.addData('smn<-mns', -gradMetric)
	christoffel_lower.addData('smn<-msn', gradMetric)
	christoffel_lower.addData('smn<-snm', gradMetric)
	christoffel = fields.TensorField.tensorProduct(
		'smn,as->amn', christoffel_lower, invMetric)

	###############################################################
	##################         init data         ##################
	###############################################################

	def bump(x: np.ndarray, x0: np.ndarray, width: float):
		r = (x - x0) / width * np.pi / 2
		r = np.sqrt(np.dot(r, r))
		return np.cos(r)**2 if r < (np.pi / 2) else 0

	def initialPhi(x: frames.CoordinatePos):
		x.inFrame(globalFrame)
		# return frames.FramedTensor(0., x, 0)
		return frames.FramedTensor(bump(x.coords,np.zeros(2),0.8), x, 0)
	def initialPi(x: frames.CoordinatePos):
		x.inFrame(globalFrame)
		return frames.FramedTensor(0., x, 0)

	initPhi = fields.TensorField.functionInit(grid, initialPhi)
	initPi = fields.TensorField.functionInit(grid, initialPi)
	initGamma= fields.trueGradient(initPhi, christoffel, 0)

	initState = bundles.StateBundle(
		[metric, invMetric, christoffel, initPhi, initPi, initGamma],
		[0,1,2])

	initStateVector = initState.canonicalPack()

	currentState = initState

	# #phi, pi, gamma_x, gamma_y

	# xLeftHat = np.array([0., 1, 1, 0]) / np.sqrt(2.)
	# xRightHat = np.array([0., 1, -1, 0]) / np.sqrt(2.)
	# yLeftHat = np.array([0., 1, 0, 1]) / np.sqrt(2.)
	# yRightHat = np.array([0., 1, 0, -1]) / np.sqrt(2.)

	def handleBoundary(
		boundaryBundle: bundles.BoundaryBundle):

		# metricField = boundaryBundle.tensorElements[0]
		invMetricField = boundaryBundle.tensorElements[1]
		# christoffelField = boundaryBundle.tensorElements[2]
		# phiField = boundaryBundle.tensorElements[3]
		piField = boundaryBundle.tensorElements[4]
		gammaField = boundaryBundle.tensorElements[5]

		shape = boundaryBundle.gridBoundary.shape
		dim = len(shape)
		normalAxis = boundaryBundle.gridBoundary.axis

		orthoVector = np.array([0] * normalAxis + [1] + [0] * (dim - normalAxis))
		orthoVector *= 1 if boundaryBundle.gridBoundary.isRightBoundary else -1
		def orthoInitFunct(x: np.ndarray):
			return orthoVector
		orthoVectorField = elements.TensorFieldElement.functionInit(
			[count - 1 for count in shape], orthoInitFunct)

		outgoing = -elements.TensorFieldElement.tensorProduct(
			'i,j,ij->', gammaField, orthoVectorField, invMetricField)

		gammaField.addData(elements.TensorFieldElement.tensorProduct(
			',j->j', outgoing, orthoVectorField))
		outgoing.addData(piField)

		gammaField.addData(-elements.TensorFieldElement.tensorProduct(
			',j->j', outgoing, orthoVectorField) / 2)
		piField.setData(outgoing / 2)

	def stateDot(time, stateVector):
		currentState.canonicalUnpack(stateVector)
		
		for boundary in currentState.boundaries:
			handleBoundary(boundary)

		# metricField = currentState.tensorFields[0]
		invMetricField = currentState.tensorFields[1]
		christoffelField = currentState.tensorFields[2]
		# phiField = currentState.tensorFields[3]
		piField = currentState.tensorFields[4]
		gammaField = currentState.tensorFields[5]

		gradGamma = fields.trueGradient(gammaField, christoffelField, 0)

		phiDot = piField.copy()
		piDot = fields.TensorField.tensorProduct('ij,ij->',gradGamma,invMetricField)

		gammaDot = fields.trueGradient(piField, christoffelField, 0)

		stateDot = bundles.StateBundle(
			[phiDot, piDot, gammaDot])
			
		return stateDot.canonicalPack()

	t1 = time.time()

	solutionSet = scipy.integrate.solve_ivp(
		stateDot, [tData[0], tData[-1]], initStateVector, dense_output=True)

	t2 = time.time()
	print(t2 - t1)

	outputDataSet = np.zeros((len(tData), len(initStateVector)))

	for i in range(len(tData)):
		outputDataSet[i] = solutionSet.sol(tData[i])

	currentState.canonicalUnpack(outputDataSet[i])
	phiField = currentState.tensorFields[3]
	piField = currentState.tensorFields[4]
	gammaField = currentState.tensorFields[5]

	print(phiField.elements[0].data)
	print(piField.elements[0].data)
	print(gammaField.elements[0].data[:,:,0])
	print(gammaField.elements[0].data[:,:,1])



scalar_field(
	N=24,
	simDuration=3.)
