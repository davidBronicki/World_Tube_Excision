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

	def __neg__(self):
		return ModalRepresentation(self.chebEngines, -self.data)

class BoundaryVector:
	def __init__(self, components: List[ModalRepresentation], axis: int):
		self.originalComponents = components
		self.leftComponents: List[ModalRepresentation] = []
		self.rightComponents: List[ModalRepresentation] = []
		for component in components:
			self.leftComponents.append(component.getLeftBoundary(axis))
			self.rightComponents.append(component.getRightBoundary(axis))
		self.leftSliceData = np.zeros((len(components),) + self.leftComponents[0].data.shape)
		self.rightSliceData = np.zeros((len(components),) + self.rightComponents[0].data.shape)
		for i in range(len(components)):
			self.leftSliceData[i] = self.leftComponents[i].data
			self.rightSliceData[i] = self.rightComponents[i].data
		self.axis = axis

	def enforceRightProjectionCondition(self, projectionMatrix: np.ndarray):
		self.rightSliceData = np.tensordot(
			projectionMatrix,
			self.rightSliceData,
			(1, 0)
		)
		for i in range(len(self.originalComponents)):
			self.originalComponents[i].setRightBoundary(
				self.axis,
				ModalRepresentation(self.originalComponents[i].chebEngines,
					self.rightSliceData[i])
			)

	def enforceLeftProjectionCondition(self, projectionMatrix: np.ndarray):
		self.leftSliceData = np.tensordot(
			projectionMatrix,
			self.leftSliceData,
			(1, 0)
		)
		for i in range(len(self.originalComponents)):
			self.originalComponents[i].setLeftBoundary(
				self.axis,
				ModalRepresentation(self.originalComponents[i].chebEngines,
					self.leftSliceData[i])
			)

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

def em_2D(
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

	def initialE(x):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return A*r**2 * np.exp(-a * (r - b)) * mask
		return np.array([0., 0.])
		# return np.sin(x[0] * np.pi * 2) * np.sin(x[1] * np.pi * 1)
	def initialA(x):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return -(2 * A * r * np.exp(-a * (r - b)) * mask - a * A * r**2 * np.exp(-a * (r - b)) * mask)
		return np.array([0., 0.])

	def bump(x, x0, width):
		r = (x - x0) / width * np.pi / 2
		r = np.sqrt(np.dot(r, r))
		return np.cos(r)**2 if r < (np.pi / 2) else 0

	def chargeDensity(x, t):
		width = 0.3
		amplitude = 0.4
		amplitude /= width**2
		sourceLocation = np.array([0., 0.])
		return bump(x, sourceLocation, width) * amplitude

		# return 5 * np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi)

	chebEngine = ChebEngine(N)
	engines = [chebEngine, chebEngine]

	initE_x = functionToModal(lambda x: initialE(x)[0], engines)
	initE_y = functionToModal(lambda x: initialE(x)[1], engines)
	initA_x = functionToModal(lambda x: initialA(x)[0], engines)
	initA_y = functionToModal(lambda x: initialA(x)[1], engines)
	initGamma_xx = gradientComponent(initA_x, 0)
	initGamma_yx = gradientComponent(initA_x, 1)
	initGamma_xy = gradientComponent(initA_y, 0)
	initGamma_yy = gradientComponent(initA_y, 1)

	initState = np.array([
		initE_x.data,
		initE_y.data,
		initA_x.data,
		initA_y.data,
		initGamma_xx.data,
		initGamma_yy.data,
		initGamma_xy.data,
		initGamma_yx.data
	])
	initState = np.reshape(initState, (8 * (N + 1)**2,))

	#e_x, e_y, gamma_yy, gamma_xy
	xAxis_xLeft = np.array([1., 0., 1., 0.]) / np.sqrt(2.)
	xAxis_xRight = np.array([1., 0., -1., 0.]) / np.sqrt(2.)
	xAxis_yRight = np.array([0., 1., 0., 1.]) / np.sqrt(2.)
	xAxis_yLeft = np.array([0., 1., 0., -1.]) / np.sqrt(2.)

	#e_x, e_y, gamma_xx, gamma_yx
	yAxis_xRight = np.array([1., 0., 0., 1.]) / np.sqrt(2.)
	yAxis_xLeft = np.array([1., 0., 0., -1.]) / np.sqrt(2.)
	yAxis_yLeft = np.array([0., 1., 1., 0.]) / np.sqrt(2.)
	yAxis_yRight = np.array([0., 1., -1., 0.]) / np.sqrt(2.)

	def stateDot(time, state):
		state = np.reshape(state, (8, N + 1, N + 1))
		e_x = ModalRepresentation(engines, state[0])
		e_y = ModalRepresentation(engines, state[1])
		gamma_xx = ModalRepresentation(engines, state[2])
		gamma_yy = ModalRepresentation(engines, state[3])
		gamma_xy = ModalRepresentation(engines, state[4])
		gamma_yx = ModalRepresentation(engines, state[5])
		# a_x = ModalRepresentation(engines, state[6])
		# a_y = ModalRepresentation(engines, state[7])

		#horizontal boundary conditions
		xRadiationVector = [e_x, e_y, gamma_yy, gamma_xy]
		boundaryStateVector = BoundaryVector(
			xRadiationVector,
			0
		)
		boundaryStateVector.enforceLeftProjectionCondition(
			np.identity(4)
				- np.outer(xAxis_xRight, xAxis_xRight)
				- np.outer(xAxis_yRight, xAxis_yRight)
		)
		boundaryStateVector.enforceRightProjectionCondition(
			np.identity(4)
				- np.outer(xAxis_xLeft, xAxis_xLeft)
				- np.outer(xAxis_yLeft, xAxis_yLeft)
		)
		#vertical boundary conditions
		yRadiationVector = [e_x, e_y, gamma_xx, gamma_yx]
		boundaryStateVector = BoundaryVector(
			yRadiationVector,
			1
		)
		boundaryStateVector.enforceLeftProjectionCondition(
			np.identity(4)
				- np.outer(yAxis_xRight, yAxis_xRight)
				- np.outer(yAxis_yRight, yAxis_yRight)
		)
		boundaryStateVector.enforceRightProjectionCondition(
			np.identity(4)
				- np.outer(yAxis_xLeft, yAxis_xLeft)
				- np.outer(yAxis_yLeft, yAxis_yLeft)
		)

		###------------------- state dot --------------------###

		gamma_yy_x = gradientComponent(gamma_yy, 0)
		gamma_yx_y = gradientComponent(gamma_yx, 1)
		gamma_xx_y = gradientComponent(gamma_xx, 1)
		gamma_xy_x = gradientComponent(gamma_xy, 0)

		e_x_x = gradientComponent(e_x, 0)
		e_x_y = gradientComponent(e_x, 1)
		e_y_x = gradientComponent(e_y, 0)
		e_y_y = gradientComponent(e_y, 1)

		e_x_dot = gamma_yy_x - gamma_yx_y
		e_y_dot = gamma_xx_y - gamma_xy_x
		# a_y_dot = -e_x
		# a_x_dot = -e_y

		chargeDensityModal = functionToModal(
			lambda x: chargeDensity(x, time), engines)
		gamma_xx_dot = e_y_y - chargeDensityModal
		gamma_yy_dot = e_x_x - chargeDensityModal
		gamma_xy_dot = -e_y_x
		gamma_yx_dot = -e_x_y

		output = np.array([
			e_x_dot.data,
			e_y_dot.data,
			gamma_xx_dot.data,
			gamma_yy_dot.data,
			gamma_xy_dot.data,
			gamma_yx_dot.data])
			# a_x_dot.data,
			# a_y_dot.data])
		
		return np.reshape(output, (8 * (N + 1)**2, ))
	
	solutionSet = scipy.integrate.solve_ivp(
		stateDot, [tData[0], tData[-1]], initState, dense_output=True)

	t1 = time.time()
	print("simulation completed in " + str(t1 - t0) + " seconds")
	t0 = time.time()

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
	e_xDataSet = []
	e_yDataSet = []
	eDataSet = []
	# piDataSet = []
	# gamma_xDataSet = []
	# gamma_yDataSet = []

	preComputedArray = PreCompute(engines, [xData, yData])
	for outputData in outputDataSet:
		unpacked = np.reshape(outputData, (8, N + 1, N + 1))

		e_xModal = ModalRepresentation(engines, unpacked[0])
		e_yModal = ModalRepresentation(engines, unpacked[1])
		e_xFunct = modalToFunction(e_xModal)
		e_yFunct = modalToFunction(e_yModal)
		e_xData = np.zeros((len(xData), len(yData)))
		e_yData = np.zeros((len(xData), len(yData)))
		eData = np.zeros((len(xData), len(yData)))
		for i in range(len(xData)):
			for j in range(len(yData)):
				e_xData[i][j] = e_xFunct(
					np.array([i, j]), preComputedArray)
				e_yData[i][j] = e_yFunct(
					np.array([i, j]), preComputedArray)
				eData[i][j] = np.sqrt(
					e_xData[i][j]**2 + e_yData[i][j]**2)
		e_xDataSet.append(e_xData)
		e_yDataSet.append(e_yData)
		eDataSet.append(eData)
	
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
	# plotx = ax.plot_surface(meshX,meshY,e_xDataSet[0])
	# ploty = ax.plot_surface(meshX,meshY,e_yDataSet[0])
	plotmag = ax.plot_surface(meshX,meshY,eDataSet[0])

	def animate(frame):
		ax.collections.clear()
		# plotx = ax.plot_surface(
		# 	meshX,meshY,e_xDataSet[frame], color='blue')
		# ploty = ax.plot_surface(
		# 	meshX,meshY,e_yDataSet[frame], color='red')
		plotmag = ax.plot_surface(
			meshX,meshY,eDataSet[frame], color='blue')
	
	ani = animation.FuncAnimation(
		fig, animate, len(tData),
		interval = animationDuration * 1000 / len(tData)
	)
	plt.show()

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

	def initialPhi(x):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return A*r**2 * np.exp(-a * (r - b)) * mask
		return 0.
	def initialPi(x):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return -(2 * A * r * np.exp(-a * (r - b)) * mask - a * A * r**2 * np.exp(-a * (r - b)) * mask)
		return 0.

	def bump(x, x0, width):
		r = (x - x0) / width * np.pi / 2
		r = np.sqrt(np.dot(r, r))
		return np.cos(r)**2 if r < (np.pi / 2) else 0

	def sourceFunct(x, t):
		width = 0.25
		amplitude = 0.5
		amplitude /= width**2
		omega = 2.0
		radius = 0.3
		sourceLocation = np.array([np.cos(t * omega), np.sin(t * omega)]) * radius
		return (bump(x, sourceLocation, width) - bump(x, -sourceLocation, width)) * amplitude
		
		# width = 0.15
		# amplitude = 0.1
		# amplitude /= width**2
		# sourceLocation = np.array([0.2, 0.])
		# return bump(x, sourceLocation, width) * amplitude

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

	# xLeftToRight = np.outer(xRightHat, xLeftHat)
	# xRightToLeft = np.outer(xLeftHat, xRightHat)
	# yLeftToRight = np.outer(yRightHat, yLeftHat)
	# yRightToLeft = np.outer(yLeftHat, yRightHat)

	def stateDot(time, state):
		state = np.reshape(state, (4, N + 1, N + 1))
		phi = ModalRepresentation(engines, state[0])
		pi = ModalRepresentation(engines, state[1])
		gamma_x = ModalRepresentation(engines, state[2])
		gamma_y = ModalRepresentation(engines, state[3])
		xRadiationVector = [phi, pi, gamma_x, gamma_y]
		yRadiationVector = [phi, pi, gamma_x, gamma_y]

		#horizontal boundary conditions
		boundaryStateVector = BoundaryVector(
			xRadiationVector,
			0
		)
		boundaryStateVector.enforceLeftProjectionCondition(
			np.identity(4) - xRightProjection
		)
		boundaryStateVector.enforceRightProjectionCondition(
			np.identity(4) - xLeftProjection
		)
		#vertical boundary conditions
		boundaryStateVector = BoundaryVector(
			yRadiationVector,
			1
		)
		boundaryStateVector.enforceLeftProjectionCondition(
			np.identity(4) - yRightProjection
		)
		boundaryStateVector.enforceRightProjectionCondition(
			np.identity(4) - yLeftProjection
		)

		pi_x = gradientComponent(pi, 0)
		pi_y = gradientComponent(pi, 1)

		gamma_xx = gradientComponent(gamma_x, 0)
		gamma_yy = gradientComponent(gamma_y, 1)

		phiDot = pi
		piDot = gamma_xx + gamma_yy + functionToModal(lambda x: sourceFunct(x, time), engines)
		gamma_xDot = pi_x# - functionToModal(lambda x: sourceFunct(x, time), engines)
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
	t0 = time.time()

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
		# phiModal = ModalRepresentation(engines, unpacked[0])
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
	def bump(x, x0, width):
		r = (x - x0) / width * np.sqrt(np.pi / 2)
		r = np.dot(r, r)
		return np.cos(r)**2 if r < (np.pi / 2) else 0
	def sourceFunct(x, t):
		return bump(x, np.array([0.]), 0.15)
		# return np.sin(np.pi * x)

	chebEngine = ChebEngine(N)

	initPhi = functionToModal(initialPhi, [chebEngine])
	initPi = functionToModal(initialPi, [chebEngine])
	initGamma = gradientComponent(initPhi, 0)

	initState = np.resize(np.array(
		[initPhi.data, initPi.data, initGamma.data]),(3 * N + 3, ))

	leftFlowVector = np.array([0, np.sqrt(0.5), np.sqrt(0.5)])
	rightFlowVector = np.array([0, np.sqrt(0.5), -np.sqrt(0.5)])

	leftProjection = np.outer(leftFlowVector, leftFlowVector)
	rightProjection = np.outer(rightFlowVector, rightFlowVector)

	def stateDot(time, state):
		state = np.reshape(state, (3, N + 1))
		phi = ModalRepresentation([chebEngine], state[0])
		pi = ModalRepresentation([chebEngine], state[1])
		gamma = ModalRepresentation([chebEngine], state[2])
		source = functionToModal(lambda x: sourceFunct(x, time),[chebEngine])
		radiationVector = [phi, pi, gamma]

		#boundary conditions
		boundaryStateVector = BoundaryVector(
			radiationVector,
			0
		)
		boundaryStateVector.enforceLeftProjectionCondition(
			np.identity(3) - rightProjection
		)
		boundaryStateVector.enforceRightProjectionCondition(
			np.identity(3) - leftProjection
		)

		phiDot = pi
		piDot = gradientComponent(gamma, 0) + source
		gammaDot = gradientComponent(pi, 0)

		return np.resize(np.array(
			[phiDot.data, piDot.data, gammaDot.data]), (3 * N + 3, ))

	solutionSet = scipy.integrate.solve_ivp(
		stateDot, [tData[0], tData[-1]], initState, dense_output=True)
	t1 = time.time()
	print("simulation completed in " + str(t1 - t0) + " seconds")
	t0 = time.time()

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
		fig, animate, len(tData),
		interval = animationDuration * 1000 / len(tData)
	)
	plt.show()

def chebTest():
	N = 4
	chebEngine = ChebEngine(N)

	def bump(x, x0, width):
		r = (x - x0) / width
		r = np.sqrt(np.dot(r, r))
		return np.exp(- r**2)
		# r = (x - x0) / width * np.pi / 2
		# r = np.sqrt(np.dot(r, r))
		# return np.cos(r)**2 if r < (np.pi / 2) else 0
	def testFunction(x):
		return bump(x, np.array([0.]), 0.15)

	def windowTranslation(x0, x1, x0p, x1p, x):
		return (x - x0) / (x1 - x0) * (x1p - x0p) + x0p

	cutoff = 0.2

	leftFunct = lambda x: testFunction(windowTranslation(-1., 1., -1., -cutoff, x))
	middleFunct = lambda x: testFunction(windowTranslation(-1., 1., -cutoff, cutoff, x))
	rightFunct = lambda x: testFunction(windowTranslation(-1., 1., cutoff, 1., x))

	newLeft = modalToFunction(functionToModal(leftFunct, [chebEngine]))
	newMiddle = modalToFunction(functionToModal(middleFunct, [chebEngine]))
	newRight = modalToFunction(functionToModal(rightFunct, [chebEngine]))

	def newFunction(x):
		if x < -0.15:
			return newLeft(windowTranslation(-1., -cutoff, -1., 1., x))
		elif x < 0.15:
			return newMiddle(windowTranslation(-cutoff, cutoff, -1., 1., x))
		else:
			return newRight(windowTranslation(cutoff, 1., -1., 1., x))

	xData = np.arange(-1.0, 1.005, 0.01)
	trueY = np.zeros(len(xData))
	newY = np.zeros(len(xData))
	for i in range(len(xData)):
		trueY[i] = testFunction(np.array([xData[i]]))
		newY[i] = newFunction(np.array([xData[i]]))

	plt.plot(xData, trueY)
	plt.plot(xData, newY)
	plt.show()



# em_2D(animationDuration=8., N=8, simDuration=4., display_dx=0.1)

# scalar_2D(animationDuration=8., N=8, simDuration=4.)

scalar_2D(animationDuration=5., N=24, simDuration=5.0, dt=0.1)

# scalar_2D(animationDuration=10., N=24, simDuration=1.5, display_dx=0.05, dt=0.03)

# scalar_1D(animationDuration=6., N=8, simDuration=6.)

# chebTest()
