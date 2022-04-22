import numpy as np
import scipy.integrate
import scipy.fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import Callable, List, Union

import time

class ChebEngine:
	def __init__(self, order: int):
		self.order = order
		self.N = order
		self.collocationPoints = []
		for i in range(0, order + 1):
			self.collocationPoints.append(np.cos(np.pi * i / order))
		
		self.polynomialCoefs = [[1],[1]]
		for n in range(2, order + 1):
			self.polynomialCoefs.append([])
			if n%2 == 0:
				self.polynomialCoefs[n].append(-self.polynomialCoefs[n - 2][0])
				for i in range(1, n // 2):
					self.polynomialCoefs[n].append(
						2 * self.polynomialCoefs[n - 1][i - 1] - self.polynomialCoefs[n - 2][i])
				self.polynomialCoefs[n].append(2 * self.polynomialCoefs[n - 1][n // 2 - 1])
			else:
				for i in range(0, n // 2):
					self.polynomialCoefs[n].append(
						2 * self.polynomialCoefs[n - 1][i] - self.polynomialCoefs[n - 2][i])
				self.polynomialCoefs[n].append(2 * self.polynomialCoefs[n - 1][n // 2])
	
	def getCollocationPoint(self, n: int) -> float:
		return self.collocationPoints[n]

	def eval(self, n: int, x: float) -> float:
		monomialValue = 1 if n % 2 == 0 else x
		xSquared = x * x
		output = monomialValue * self.polynomialCoefs[n][0]
		for i in range(1, (n + 2) // 2):
			monomialValue *= xSquared
			output += monomialValue * self.polynomialCoefs[n][i]
		return output
	
	def preCompute(self, xList: np.ndarray) -> np.ndarray:
		output = np.zeros((len(xList), self.N + 1))
		for i in range(len(xList)):
			for j in range(self.N + 1):
				output[i][j] = self.eval(j, xList[i])
		return output

class PreCompute:
	def __init__(self, chebEngines: List[ChebEngine], xLists: List[np.ndarray]):
		self.preComputedValues = []
		for engine, xList in zip(chebEngines, xLists):
			self.preComputedValues.append(engine.preCompute(xList))
	
	def __call__(self,
		orderTuple: Union[tuple, List[int], np.ndarray],
		indexTuple: Union[tuple, List[int], np.ndarray]):

		output = 1
		for order, index, computed in zip(orderTuple, indexTuple, self.preComputedValues):
			output *= computed[index][order]
		return output

class ModalRepresentation:
	def __init__(self, chebEngines: List[ChebEngine], data: Union[np.ndarray, None] = None):
		self.chebEngines = chebEngines
		if data is None:
			shape = []
			for engine in chebEngines:
				shape.append(engine.N + 1)
			self.data = np.zeros(shape)
		else:
			self.data = data

	def getSurface(self, axis: int, index: int) -> 'ModalRepresentation':
		return ModalRepresentation(
			self.chebEngines[:axis]+self.chebEngines[axis+1:],
			np.take(self.data, index, axis))

	def setSurface(self, axis: int, index: int, surfaceValue: 'ModalRepresentation') -> None:
		sliceShape = surfaceValue.data.shape[:axis] + (1,) + surfaceValue.data.shape[axis:]

		np.put_along_axis(
			self.data,
			np.full(sliceShape, index),
			np.reshape(surfaceValue.data, sliceShape),
			axis)

	def getRightBoundary(self, axis: int) -> 'ModalRepresentation':
		return self.getSurface(axis, 0)

	def getLeftBoundary(self, axis: int) -> 'ModalRepresentation':
		return self.getSurface(axis, self.data.shape[axis] - 1)

	def setRightBoundary(self, axis: int, boundaryValue: 'ModalRepresentation') -> None:
		self.setSurface(axis, 0, boundaryValue)

	def setLeftBoundary(self, axis: int, boundaryValue: 'ModalRepresentation') -> None:
		self.setSurface(axis, self.data.shape[axis] - 1, boundaryValue)

	def __mul__(self, other: Union['ModalRepresentation', int, float]):
		if type(other) == ModalRepresentation:
			return ModalRepresentation(self.chebEngines, self.data * other.data)
		else:
			return ModalRepresentation(self.chebEngines, self.data * other)

	__rmul__ = __mul__

	def __truediv__(self, scalar: Union[int, float]):
		return ModalRepresentation(self.chebEngines, self.data / scalar)

	def __add__(self, other: 'ModalRepresentation'):
		return ModalRepresentation(self.chebEngines, self.data + other.data)

	def __sub__(self, other: 'ModalRepresentation'):
		return ModalRepresentation(self.chebEngines, self.data - other.data)

def functionToModal(funct: Callable[[np.ndarray],float], chebEngines: List[ChebEngine]) -> ModalRepresentation:
	N = len(chebEngines)
	shape = []
	for engine in chebEngines:
		shape.append(engine.order + 1)
	totalSize = np.prod(shape)
	indices = np.indices(shape)
	indices = np.transpose(np.reshape(indices, (N, totalSize)))

	data = np.zeros(totalSize)
	for i in range(totalSize):
		regionPosition = np.zeros(N)
		for j in range(N):
			regionPosition[j] = chebEngines[j].getCollocationPoint(indices[i][j])
		data[i] = funct(regionPosition)
	return ModalRepresentation(chebEngines, np.reshape(data, shape))

def gradientComponent(modal: ModalRepresentation, axis: int) -> ModalRepresentation:
	N = modal.chebEngines[axis].N

	#change to spectral along given axis
	specData = ModalRepresentation(
		modal.chebEngines,
		scipy.fft.dct(modal.data, type=1, axis=axis) / N)

	specData.setRightBoundary(axis, specData.getRightBoundary(axis) / 2)
	specData.setLeftBoundary(axis, specData.getLeftBoundary(axis) / 2)

	# specData[0] /= 2
	# specData[N] /= 2

	#perform derivative in spectral basis
	specDerivative = ModalRepresentation(modal.chebEngines)
	# outputData = np.zeros(N + 1)

	specDerivative.setSurface(axis, N - 1, 2 * N * specData.getSurface(axis, N))
	specDerivative.setSurface(axis, N - 2, 2 * (N - 1) * specData.getSurface(axis, N - 1))
	# outputData[N - 1] = 2 * N * specRep.data[N]
	# outputData[N - 2] = 2 * (N - 1) * specRep.data[N - 1]

	for i in reversed(range(0, N - 2)):
		specDerivative.setSurface(axis, i,
			specDerivative.getSurface(axis, i + 2)
			+ 2 * (i + 1) * specData.getSurface(axis, i + 1))
		# outputData[i] = outputData[i + 2] + 2 * (i + 1) * specRep.data[i + 1]
	# outputData[0] /= 2

	#change back to modal representation
	# specRep.data[0] *= 2

	specDerivative.setSurface(axis, N, specDerivative.getSurface(axis, N) * 2)
	# specRep.data[N] *= 2
	output = ModalRepresentation(
		modal.chebEngines,
		scipy.fft.dct(specDerivative.data, type=1, axis=axis) / 2)
	# modalData = scipy.fft.dct(specRep.data, type=1) / 2

	return output

def modalToFunction(modal: ModalRepresentation) -> Callable[[np.ndarray, Union[None, PreCompute]], float]:
	dim = len(modal.chebEngines)
	shape = []
	for engine in modal.chebEngines:
		shape.append(engine.N + 1)
	totalSize = np.prod(shape)
	indices = np.indices(shape)
	indices = np.transpose(np.reshape(indices, (dim, totalSize)))
	
	#change to spectral

	specData = scipy.fft.dct(
		modal.data, type=1, axis=0) / modal.chebEngines[0].N
	for i in range(1, len(modal.chebEngines)):
		specData = scipy.fft.dct(
			specData, type=1, axis=i) / modal.chebEngines[i].N

	for i in range(len(indices)):
		for j in range(dim):
			if indices[i][j] == 0 or indices[i][j] == modal.chebEngines[j].N:
				specData[tuple(indices[i])] /= 2

	def outputFunct(position: np.ndarray, preComputeArray: Union[None, PreCompute] = None) -> float:
		if preComputeArray == None:
			output = 0
			for i in range(len(indices)):
				modeContribution = specData[tuple(indices[i])]
				for j in range(dim):
					modeContribution *= modal.chebEngines[j].eval(
						indices[i][j], position[j])
				output += modeContribution
			return output
		else:
			output = 0
			for indexList in indices:
				output += specData[tuple(indexList)] * preComputeArray(indexList, position)
			return output

	
	return outputFunct

def scalar_2D(
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


	a = 12
	b = 0.05
	A = 100

	def initialPhi(x):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return A*r**2 * np.exp(-a * (r - b)) * mask
		return 0.
		# return np.sin(x[0] * np.pi * 2) * np.sin(x[1] * np.pi * 1)
	def initialPi(x):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return -(2 * A * r * np.exp(-a * (r - b)) * mask - a * A * r**2 * np.exp(-a * (r - b)) * mask)
		return 0.

	def bump(x, x0, width):
		r = (x - x0) / (width * np.pi / 2)
		r = np.dot(r, r)
		return np.cos(r)**2 if r < (np.pi / 2) else 0

	def sourceFunct(x, t):
		omega = 10.
		width = 0.15
		radius = 0.2
		sourceLocation = np.array([np.cos(t * omega), np.sin(t * omega)]) * radius
		amplitude = 0.1
		amplitude /= width**2
		return (bump(x, np.array(sourceLocation), width) - bump(x, np.array(-sourceLocation), width)) * amplitude
		# return 5 * np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi)

	chebEngine = ChebEngine(N)
	engines = [chebEngine, chebEngine]

	initPhi = functionToModal(initialPhi, engines)
	initPi = functionToModal(initialPi, engines)
	initGamma_x = gradientComponent(initPhi, 0)
	initGamma_y = gradientComponent(initPhi, 1)

	initState = np.array([
		initPhi.data,
		initPi.data,
		initGamma_x.data,
		initGamma_y.data
	])
	initState = np.reshape(initState, (4 * (N + 1)**2,))

	#phi, pi, gamma_x, gamma_y

	xLeftHat = np.array([0., 1, 1, 0]) / np.sqrt(2.)
	xRightHat = np.array([0., 1, -1, 0]) / np.sqrt(2.)
	yLeftHat = np.array([0., 1, 0, 1]) / np.sqrt(2.)
	yRightHat = np.array([0., 1, 0, -1]) / np.sqrt(2.)

	xLeftProjection = np.outer(xLeftHat, xLeftHat)
	xRightProjection = np.outer(xRightHat, xRightHat)
	yLeftProjection = np.outer(yLeftHat, yLeftHat)
	yRightProjection = np.outer(yRightHat, yRightHat)

	xLeftToRight = np.outer(xRightHat, xLeftHat)
	xRightToLeft = np.outer(xLeftHat, xRightHat)
	yLeftToRight = np.outer(yRightHat, yLeftHat)
	yRightToLeft = np.outer(yLeftHat, yRightHat)

	def stateDot(time, state):
		state = np.reshape(state, (4, N + 1, N + 1))
		phi = ModalRepresentation(engines, state[0])
		pi = ModalRepresentation(engines, state[1])
		gamma_x = ModalRepresentation(engines, state[2])
		gamma_y = ModalRepresentation(engines, state[3])

		xLeftBoundaryState = np.array([
			phi.getLeftBoundary(0).data,
			pi.getLeftBoundary(0).data,
			gamma_x.getLeftBoundary(0).data,
			gamma_y.getLeftBoundary(0).data
		])
		xLeftBoundaryState = np.tensordot(
			np.identity(4) - xRightProjection - 0.0 * xLeftToRight,
			xLeftBoundaryState,
			(1, 0)
		)
		phi.setLeftBoundary(0, ModalRepresentation(
			[chebEngine], xLeftBoundaryState[0]))
		pi.setLeftBoundary(0, ModalRepresentation(
			[chebEngine], xLeftBoundaryState[1]))
		gamma_x.setLeftBoundary(0, ModalRepresentation(
			[chebEngine], xLeftBoundaryState[2]))
		gamma_y.setLeftBoundary(0, ModalRepresentation(
			[chebEngine], xLeftBoundaryState[3]))

		xRightBoundaryState = np.array([
			phi.getRightBoundary(0).data,
			pi.getRightBoundary(0).data,
			gamma_x.getRightBoundary(0).data,
			gamma_y.getRightBoundary(0).data
		])
		xRightBoundaryState = np.tensordot(
			np.identity(4) - xLeftProjection - 0.0 * xRightToLeft,
			xRightBoundaryState,
			(1, 0)
		)
		phi.setRightBoundary(0, ModalRepresentation(
			[chebEngine], xRightBoundaryState[0]))
		pi.setRightBoundary(0, ModalRepresentation(
			[chebEngine], xRightBoundaryState[1]))
		gamma_x.setRightBoundary(0, ModalRepresentation(
			[chebEngine], xRightBoundaryState[2]))
		gamma_y.setRightBoundary(0, ModalRepresentation(
			[chebEngine], xRightBoundaryState[3]))

		yLeftBoundaryState = np.array([
			phi.getLeftBoundary(1).data,
			pi.getLeftBoundary(1).data,
			gamma_x.getLeftBoundary(1).data,
			gamma_y.getLeftBoundary(1).data
		])
		yLeftBoundaryState = np.tensordot(
			np.identity(4) - yRightProjection - 0.0 * yLeftToRight,
			yLeftBoundaryState,
			(1, 0)
		)
		phi.setLeftBoundary(1, ModalRepresentation(
			[chebEngine], yLeftBoundaryState[0]))
		pi.setLeftBoundary(1, ModalRepresentation(
			[chebEngine], yLeftBoundaryState[1]))
		gamma_x.setLeftBoundary(1, ModalRepresentation(
			[chebEngine], yLeftBoundaryState[2]))
		gamma_y.setLeftBoundary(1, ModalRepresentation(
			[chebEngine], yLeftBoundaryState[3]))

		yRightBoundaryState = np.array([
			phi.getRightBoundary(1).data,
			pi.getRightBoundary(1).data,
			gamma_x.getRightBoundary(1).data,
			gamma_y.getRightBoundary(1).data
		])
		yRightBoundaryState = np.tensordot(
			np.identity(4) - yLeftProjection - 0.0 * yRightToLeft,
			yRightBoundaryState,
			(1, 0)
		)
		phi.setRightBoundary(1, ModalRepresentation(
			[chebEngine], yRightBoundaryState[0]))
		pi.setRightBoundary(1, ModalRepresentation(
			[chebEngine], yRightBoundaryState[1]))
		gamma_x.setRightBoundary(1, ModalRepresentation(
			[chebEngine], yRightBoundaryState[2]))
		gamma_y.setRightBoundary(1, ModalRepresentation(
			[chebEngine], yRightBoundaryState[3]))

		pi_x = gradientComponent(pi, 0)
		pi_y = gradientComponent(pi, 1)

		gamma_xx = gradientComponent(gamma_x, 0)
		gamma_yy = gradientComponent(gamma_y, 1)

		phiDot = pi
		piDot = gamma_xx + gamma_yy + functionToModal(lambda x: sourceFunct(x, time), engines)
		gamma_xDot = pi_x
		gamma_yDot = pi_y

		output = np.array([
			phiDot.data,
			piDot.data,
			gamma_xDot.data,
			gamma_yDot.data])
		
		return np.reshape(output, (4 * (N + 1)**2, ))


	solutionSet = scipy.integrate.solve_ivp(
		stateDot, [tData[0], tData[-1]], initState, dense_output=True)

	t1 = time.time()
	print("simulation completed in " + str(t1 - t0) + " seconds")
	x0 = time.time()

	outputDataSet = np.zeros((len(tData), len(initState)))

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

	preComputedArray = PreCompute(engines, [xData, yData])
	for outputData in outputDataSet:
		unpacked = np.reshape(outputData, (4, N + 1, N + 1))

		phiModal = ModalRepresentation(engines, unpacked[0])
		phiFunct = modalToFunction(phiModal)
		phiData = np.zeros((len(xData), len(yData)))
		for i in range(len(xData)):
			for j in range(len(yData)):
				phiData[i][j] = phiFunct(np.array([i, j]), preComputedArray)
		phiDataSet.append(phiData)
	
	t1 = time.time()
	print("data evaluated in " + str(t1 - t0) + " seconds")

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

def scalar_1D(
	N = 15,
	animationDuration = 2.,
	simDuration = 3.0,
	dt = 0.05,
	display_dx = 0.02):

	t0 = time.time()

	xData = np.arange(-1., 1., display_dx)
	tData = np.arange(0, simDuration, dt)

	def initialPhi(x):
		return np.sin(x[0] * np.pi * 3/2)
	def initialPi(x):
		return 0
	def sourceFunct(x, t):
		return np.sin(np.pi * x)

	chebEngine = ChebEngine(N)

	initPhi = functionToModal(initialPhi, [chebEngine])
	initPi = functionToModal(initialPi, [chebEngine])
	initGamma = gradientComponent(initPhi, 0)

	initState = np.resize(np.array(
		[initPhi.data, initPi.data, initGamma.data]),(3 * N + 3, ))

	leftFlowVector = np.array([0, np.sqrt(0.5), np.sqrt(0.5)])
	rightFlowVector = np.array([0, np.sqrt(0.5), -np.sqrt(0.5)])

	leftFlowProjection = np.outer(leftFlowVector, leftFlowVector)
	rightFlowProjection = np.outer(rightFlowVector, rightFlowVector)

	leftFlowRemovalProjection = np.identity(3) - leftFlowProjection
	rightFlowRemovalProjection = np.identity(3) - rightFlowProjection

	def stateDot(time, state):
		state = np.reshape(state, (3, N + 1))
		phi = ModalRepresentation([chebEngine], state[0])
		pi = ModalRepresentation([chebEngine], state[1])
		gamma = ModalRepresentation([chebEngine], state[2])
		source = functionToModal(lambda x: sourceFunct(x, time),[chebEngine])

		rightBoundaryState = np.array([
			phi.getRightBoundary(0).data,
			pi.getRightBoundary(0).data,
			gamma.getRightBoundary(0).data])
		leftBoundaryState = np.array([
			phi.getLeftBoundary(0).data,
			pi.getLeftBoundary(0).data,
			gamma.getLeftBoundary(0).data])

		correctedRightBoundary = np.tensordot(
			leftFlowRemovalProjection,
			rightBoundaryState,
			(1, 0)
		)
		correctedLeftBoundary = np.tensordot(
			rightFlowRemovalProjection,
			leftBoundaryState,
			(1, 0)
		)

		phi.setRightBoundary(0, ModalRepresentation(
			[], correctedRightBoundary[0]))
		pi.setRightBoundary(0, ModalRepresentation(
			[], correctedRightBoundary[1]))
		gamma.setRightBoundary(0, ModalRepresentation(
			[], correctedRightBoundary[2]))

		phi.setLeftBoundary(0, ModalRepresentation(
			[], correctedLeftBoundary[0]))
		pi.setLeftBoundary(0, ModalRepresentation(
			[], correctedLeftBoundary[1]))
		gamma.setLeftBoundary(0, ModalRepresentation(
			[], correctedLeftBoundary[2]))

		phiDot = pi
		piDot = gradientComponent(gamma, 0) + source
		gammaDot = gradientComponent(pi, 0)

		return np.resize(np.array(
			[phiDot.data, piDot.data, gammaDot.data]), (3 * N + 3, ))

	solutionSet = scipy.integrate.solve_ivp(
		stateDot, [tData[0], tData[-1]], initState, dense_output=True)
	t1 = time.time()
	print("simulation completed in " + str(t1 - t0) + " seconds")
	x0 = time.time()

	yDataSet = np.zeros((len(tData), len(initState)))

	for i in range(len(tData)):
		yDataSet[i] = solutionSet.sol(tData[i])

	solver_tData = solutionSet.t
	dtList = []
	for i in range(1, len(solver_tData)):
		dtList.append(solver_tData[i] - solver_tData[i - 1])
	
	print("average deltaT: " + str(sum(dtList) / len(dtList)))
	

	# yDataSet = np.transpose(solutionSet.y)
	phiDataSet = []
	piDataSet = []
	gammaDataSet = []

	for yData in yDataSet:
		unpacked = np.reshape(yData, (3, N + 1))
		phiDataSet.append(np.zeros(len(xData)))
		piDataSet.append(np.zeros(len(xData)))
		gammaDataSet.append(np.zeros(len(xData)))
		phiFunct = modalToFunction(
			ModalRepresentation([chebEngine], unpacked[0]))
		piFunct = modalToFunction(
			ModalRepresentation([chebEngine], unpacked[1]))
		gammaFunct = modalToFunction(
			ModalRepresentation([chebEngine], unpacked[2]))

		for j in range(len(xData)):
			phiDataSet[-1][j] = phiFunct(np.array([xData[j]]))
			piDataSet[-1][j] = piFunct(np.array([xData[j]]))
			gammaDataSet[-1][j] = gammaFunct(np.array([xData[j]]))
	
	t1 = time.time()
	print("data evaluated in " + str(t1 - t0) + " seconds")

	maxVal = -1000.
	minVal = 1000.

	for phiData, piData, gammaData in zip(phiDataSet, piDataSet, gammaDataSet):
		maxVal = max(maxVal, max(phiData), max(piData), max(gammaData))
		minVal = min(minVal, min(phiData), min(piData), min(gammaData))
		maxVal = max(maxVal, max(phiData))
		minVal = min(minVal, min(phiData))

	fig = plt.figure(figsize=(5, 4))
	ax = fig.add_subplot(
		autoscale_on=False,
		xlim=(-1., 1.),
		ylim=(minVal-0.01, maxVal+0.01))
	# ax.set_aspect('equal')
	ax.grid()
	phiGraph, = ax.plot([],[], label='phi')
	piGraph, = ax.plot([],[], label='pi')
	gammaGraph, = ax.plot([],[], label='gamma')
	ax.legend(loc='upper right')

	def animate(i):
		phiGraph.set_data(xData, phiDataSet[i])
		piGraph.set_data(xData, piDataSet[i])
		gammaGraph.set_data(xData, gammaDataSet[i])
		return phiGraph, piGraph, gammaGraph
		# return phiGraph
	
	print(functionToModal(lambda x: sourceFunct(x, 0), [chebEngine]).data)

	ani = animation.FuncAnimation(
		fig, animate, len(tData), interval = int(animationDuration * 20)#(tData[-1] - tData[0])*30
	)
	plt.show()

# scalar_2D(animationDuration=5., N=8)
scalar_2D(animationDuration=10., N=16, simDuration=8.0, dt=0.03)
# scalar_2D(animationDuration=10., N=24, simDuration=1.5, display_dx=0.05, dt=0.03)
# scalar_1D()

