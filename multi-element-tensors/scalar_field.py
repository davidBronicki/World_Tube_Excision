import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time

import elements, frames, fields, grids, bundles

def scalar_field(
	N = 8,
	animationDuration = 2.,
	simDuration = 3.0,
	dt = 0.1,
	display_dx = 0.1):

	tData = np.arange(0, simDuration, dt)

	###############################################################
	##################         grid data         ##################
	###############################################################

	globalFrame = frames.CoordinateFrame('global', 2)
	globalToGlobal = globalFrame.getIdentityTransform()

	centralGrid = grids.GridElement((N, N), globalFrame)
	grid = grids.Grid([globalToGlobal], [centralGrid])

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
	print("simulation completed in " + str(t2 - t1) + " seconds")
	t1 = t2

	outputDataSet = np.zeros((len(tData), len(initStateVector)))

	for i in range(len(tData)):
		outputDataSet[i] = solutionSet.sol(tData[i])

	solver_tData = solutionSet.t
	dtList = []
	for i in range(1, len(solver_tData)):
		dtList.append(solver_tData[i] - solver_tData[i - 1])
	print("average deltaT: " + str(sum(dtList) / len(dtList)))

	# tData = solutionSet.t
	# yDataSet = np.transpose(solutionSet.y)
	phiDataSet = []
	# piDataSet = []
	# gamma_xDataSet = []
	# gamma_yDataSet = []

	chebEngine = elements.ChebEngine(N)
	engines = [chebEngine, chebEngine]

	xData = np.arange(-1., 1. + display_dx / 2, display_dx)
	yData = np.arange(-1., 1. + display_dx / 2, display_dx)
	meshX, meshY = np.meshgrid(xData, yData)
	preComputedArray = elements.PreCompute(engines, [xData, yData])

	for outputData in outputDataSet:
		currentState.canonicalUnpack(outputData)

		# metricField = currentState.tensorFields[0]
		# invMetricField = currentState.tensorFields[1]
		# christoffelField = currentState.tensorFields[2]
		phiField = currentState.tensorFields[3]
		# piField = currentState.tensorFields[4]
		# gammaField = currentState.tensorFields[5]

		phiFunct = phiField.elements[0].toFunction()
		phiData = np.zeros((len(xData), len(yData)))
		for i in range(len(xData)):
			for j in range(len(yData)):
				phiData[i][j] = phiFunct(np.array([i, j]), preComputedArray)
		phiDataSet.append(phiData)

	t2 = time.time()
	print("data evaluated in " + str(t2 - t1) + " seconds")
	t1 = t2

	minVal = -1.
	maxVal = 1.

	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	ax = fig.add_subplot(111,
		projection='3d',
		autoscale_on=False,
		xlim=(-1., 1.),
		ylim=(-1., 1.),
		zlim=(minVal-0.01, maxVal+0.01))
	plot = ax.plot_surface(meshX,meshY,phiDataSet[0])

	def animate(frame):
		ax.collections.clear()
		plot = ax.plot_surface(meshX,meshY,phiDataSet[frame], color='blue')
	
	ani = animation.FuncAnimation(
		fig, animate, len(tData),
		interval = animationDuration * 1000 / len(tData)
	)
	plt.show()

if __name__ == '__main__':
	scalar_field(
		N=24,
		simDuration=3.)
