import numpy as np
import scipy.integrate
import scipy.fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable

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
	
	def getCollocationPoint(self, n: int, x0 = -1.0, x1 = 1.0) -> float:
		return (self.collocationPoints[n] + 1) / 2 * (x1 - x0) + x0

	def eval(self, n: int, x: float, x0 = -1.0, x1 = 1.0) -> float:
		x = (x - x0) / (x1 - x0) * 2 - 1
		monomialValue = 1 if n % 2 == 0 else x
		xSquared = x * x
		output = monomialValue * self.polynomialCoefs[n][0]
		for i in range(1, (n + 2) // 2):
			monomialValue *= xSquared
			output += monomialValue * self.polynomialCoefs[n][i]
		return output

class ModalRepresentation:
	def __init__(self, chebEngine: ChebEngine, data: np.ndarray):
		self.chebEngine = chebEngine
		self.data = data
	
	def __mul__(self, other):
		if type(other) == ModalRepresentation:
			return ModalRepresentation(self.chebEngine, self.data * other.data)
		else:
			return ModalRepresentation(self.chebEngine, self.data * other)
	
	__rmul__ = __mul__
	
	def __truediv__(self, scalar):
		return ModalRepresentation(self.chebEngine, self.data / scalar)

	def __add__(self, other):
		return ModalRepresentation(self.chebEngine, self.data + other.data)
	
	def __sub__(self, other):
		return ModalRepresentation(self.chebEngine, self.data - other.data)

class SpectralRepresentation:
	def __init__(self, chebEngine: ChebEngine, data: np.ndarray):
		self.chebEngine = chebEngine
		self.data = data
	
	def __mul__(self, scalar):
		return SpectralRepresentation(self.chebEngine, self.data * scalar)
	
	def __truediv__(self, scalar):
		return SpectralRepresentation(self.chebEngine, self.data / scalar)

	def __add__(self, other):
		return SpectralRepresentation(self.chebEngine, self.data + other.data)
	
	def __sub__(self, other):
		return SpectralRepresentation(self.chebEngine, self.data - other.data)

def functionToModal(funct: Callable[[float],float], chebEngine: ChebEngine, x0 = -1.0, x1 = 1.0) -> ModalRepresentation:
	data = np.zeros(chebEngine.N + 1)
	data[0] = funct(chebEngine.getCollocationPoint(0, x0, x1))
	data[chebEngine.N] = funct(chebEngine.getCollocationPoint(chebEngine.N, x0, x1))
	for i in range(1, chebEngine.N):
		data[i] = funct(chebEngine.getCollocationPoint(i, x0, x1))
	return ModalRepresentation(chebEngine, data)

def modalToSpectral(modalRep: ModalRepresentation) -> SpectralRepresentation:
	N = modalRep.chebEngine.N
	specData = scipy.fft.dct(modalRep.data, type=1) / N
	specData[0] /= 2
	specData[N] /= 2
	return SpectralRepresentation(modalRep.chebEngine, specData)

def spectralToModal(specRep: SpectralRepresentation) -> ModalRepresentation:
	N = specRep.chebEngine.N
	specRep.data[0] *= 2
	specRep.data[N] *= 2
	modalData = scipy.fft.dct(specRep.data, type=1) / 2
	specRep.data[0] /= 2
	specRep.data[N] /= 2
	return ModalRepresentation(specRep.chebEngine, modalData)

def spectralToFunction(specRep: SpectralRepresentation, x0 = -1.0, x1 = 1.0) -> Callable[[float],float]:
	def outputFunction(x):
		output = 0
		for n in range(0, specRep.chebEngine.N + 1):
			output += specRep.data[n].real * specRep.chebEngine.eval(n, x, x0, x1)
		return output
	return outputFunction

def spectralDerivative(specRep: SpectralRepresentation, domainWidth = 2.0) -> SpectralRepresentation:
	N = specRep.chebEngine.N
	widthCorrectionFactor = 2.0 / domainWidth
	outputData = np.zeros(N + 1)
	outputData[N - 1] = 2 * N * specRep.data[N]
	outputData[N - 2] = 2 * (N - 1) * specRep.data[N - 1]
	for i in reversed(range(0, N - 2)):
		outputData[i] = outputData[i + 2] + 2 * (i + 1) * specRep.data[i + 1]
	outputData[0] /= 2
	return SpectralRepresentation(specRep.chebEngine, outputData * widthCorrectionFactor)

def modalBasisFunction(chebEngine: ChebEngine, n: int) -> ModalRepresentation:
	outputData = np.zeros(chebEngine.N + 1)
	outputData[n] = 1
	return ModalRepresentation(chebEngine, outputData)

def spectralBasisFunction(chebEngine: ChebEngine, n: int) -> SpectralRepresentation:
	outputData = np.zeros(chebEngine.N + 1)
	outputData[n] = 1
	return SpectralRepresentation(chebEngine, outputData)

def modalMultiplicationOperator(modalRep: ModalRepresentation) -> np.ndarray:
	return np.diag(modalRep.data)

def modalDifferentialOperator(chebEngine: ChebEngine) -> np.ndarray:
	N = chebEngine.N
	operatorMatrix = np.zeros((N + 1, N + 1))
	for i in range(N+1):
		modalUnit = modalBasisFunction(chebEngine, i)
		modalDeriv = spectralToModal(spectralDerivative(modalToSpectral(modalUnit)))
		operatorMatrix[i] = modalDeriv.data

	operatorMatrix = np.transpose(operatorMatrix)
	return operatorMatrix

###--------------------------tests---------------------------------###

def scalarFieldTest():

	N = 15

	xData = np.arange(-1., 1., 0.05)
	tData = np.arange(0, 3.0, 0.05)
	# tData = np.arange(0, 0.15, 0.005)

	def initialPhi(x):
		# return np.cos(x * np.pi*2)
		return np.sin(x * np.pi * 5 /2)
	def initialPi(x):
		return 0

	chebEngine = ChebEngine(N)

	phiModal = functionToModal(initialPhi, chebEngine)
	phiSpectral = modalToSpectral(phiModal)
	piModal = functionToModal(initialPi, chebEngine)
	piSpectral = modalToSpectral(piModal)
	gammaSpectral = spectralDerivative(phiSpectral)
	gammaModal = spectralToModal(gammaSpectral)

	state = np.resize(np.array(
		[phiModal.data, piModal.data, gammaModal.data]),(3 * N + 3, ))

	leftFlowVector = np.array([0, np.sqrt(0.5), np.sqrt(0.5)])
	rightFlowVector = np.array([0, np.sqrt(0.5), -np.sqrt(0.5)])

	leftFlowProjection = np.outer(leftFlowVector, leftFlowVector)
	rightFlowProjection = np.outer(rightFlowVector, rightFlowVector)

	leftFlowRemovalProjection = np.identity(3) - leftFlowProjection
	rightFlowRemovalProjection = np.identity(3) - rightFlowProjection
	
	def stateDot(time, state):
		state = np.resize(state, (3, N + 1))
		phiModal = ModalRepresentation(chebEngine, state[0])
		piModal = ModalRepresentation(chebEngine, state[1])
		gammaModal = ModalRepresentation(chebEngine, state[2])

		# rightBound = (piModal - gammaModal) / 2
		# leftBound = (piModal + gammaModal) / 2

		# rightBoundaryState = np.array([phiModal[0], piModal[0], gammaModal[0]])
		# leftBoundaryState = np.array([phiModal[-1], piModal[-1], gammaModal[-1]])


		leftBoundaryEntering = (piModal.data[-1] + gammaModal.data[-1]) / 2
		leftBoundaryExiting = (piModal.data[-1] - gammaModal.data[-1]) / 2
		rightBoundaryEntering = (piModal.data[0] - gammaModal.data[0]) / 2
		rightBoundaryExiting = (piModal.data[0] + gammaModal.data[0]) / 2

		phiDot = piModal
		piDot = spectralToModal(spectralDerivative(
			modalToSpectral(gammaModal)))
		gammaDot = spectralToModal(spectralDerivative(
			modalToSpectral(piModal)))

		# rightBoundaryDot = leftFlowRemovalProjection

		piDot.data[-1] = (piDot.data[-1] + gammaDot.data[-1]) / 2
		gammaDot.data[-1] = piDot.data[-1]

		piDot.data[-1] -= leftBoundaryExiting
		gammaDot.data[-1] += leftBoundaryExiting

		piDot.data[0] = (piDot.data[0] - gammaDot.data[0]) / 2
		gammaDot.data[0] = -piDot.data[0]

		piDot.data[0] -= rightBoundaryExiting
		gammaDot.data[0] -= rightBoundaryExiting

		return np.resize(np.array(
			[phiDot.data, piDot.data, gammaDot.data]), (3 * N + 3, ))

	solutionSet = scipy.integrate.solve_ivp(
		stateDot, [tData[0], tData[-1]], state, t_eval=tData)

	tData = solutionSet.t
	yDataSet = np.transpose(solutionSet.y)
	phiDataSet = []
	piDataSet = []
	gammaDataSet = []

	for yData in yDataSet:
		unpacked = np.reshape(yData, (3, N + 1))
		phiDataSet.append(np.zeros(len(xData)))
		piDataSet.append(np.zeros(len(xData)))
		gammaDataSet.append(np.zeros(len(xData)))
		phiFunct = spectralToFunction(modalToSpectral(
			ModalRepresentation(chebEngine, unpacked[0])))
		piFunct = spectralToFunction(modalToSpectral(
			ModalRepresentation(chebEngine, unpacked[1])))
		gammaFunct = spectralToFunction(modalToSpectral(
			ModalRepresentation(chebEngine, unpacked[2])))

		for j in range(len(xData)):
			phiDataSet[-1][j] = phiFunct(xData[j])
			piDataSet[-1][j] = piFunct(xData[j])
			gammaDataSet[-1][j] = gammaFunct(xData[j])

	# for i in range(len(phiDataSet)):
	# 	yData = phiDataSet[i]
	# 	yData = np.reshape(yData, (3, N + 1))
	# 	phiData = ModalRepresentation(chebEngine, yData[0])
	# 	phiFunct = spectralToFunction(modalToSpectral(phiData))
	# 	yData = np.zeros(len(xData))
	# 	for j in range(len(xData)):
	# 		yData[j] = phiFunct(xData[j])
	# 	phiDataSet[i] = yData

	maxVal = -1000.
	minVal = 1000.

	for phiData, piData, gammaData in zip(phiDataSet, piDataSet, gammaDataSet):
		maxVal = max(maxVal, max(phiData), max(piData), max(gammaData))
		minVal = min(minVal, min(phiData), min(piData), min(gammaData))

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
	
	ani = animation.FuncAnimation(
		fig, animate, len(tData), interval = 120#(tData[-1] - tData[0])*30
	)
	plt.show()

def nonLinearSolveTest(N: int, a = 1.0):
	#trying to solve (y')^2=a y, y(0)=1  -->  y(x)=x^2/4+ax+a^2
	chebEngine = ChebEngine(N)

	# estimatedSolution = spectralToModal(spectralBasisFunction(chebEngine, 1))
	estimatedSolution = functionToModal(lambda x: x/2 + 1, chebEngine)

	errors = []
	def calcError():
		estimatedSolutionPrime = spectralToModal(
			spectralDerivative(modalToSpectral(estimatedSolution)))
		errorFunction = estimatedSolutionPrime * estimatedSolutionPrime - estimatedSolution
		errorFunction *= errorFunction
		return sum(errorFunction.data)
	
	errors.append(calcError())
	while(errors[-1] > 1e-12):
		linearMatrix = np.zeros((N+1, N+1))
		boundaryVector = np.zeros(N+1)

		estimatedSolutionPrime = spectralToModal(
			spectralDerivative(modalToSpectral(estimatedSolution)))
		sourceVector = (a * estimatedSolution - estimatedSolutionPrime * estimatedSolutionPrime).data
		sourceVector[N] = 0

		for n in range(N+1):
			modalUnit = modalBasisFunction(chebEngine, n)
			specVector = modalToSpectral(modalUnit)

			columnVector = spectralToModal(spectralDerivative(specVector))
			columnVector *= 2 * estimatedSolutionPrime
			columnVector -= a * modalUnit
			linearMatrix[n] = columnVector.data
			boundaryVector[n] = spectralToFunction(specVector)(0)
		
		linearMatrix = np.transpose(linearMatrix)
		linearMatrix[N] = boundaryVector

		solutionMatrix = np.linalg.inv(linearMatrix)
		estimatedSolution += ModalRepresentation(
			estimatedSolution.chebEngine, solutionMatrix.dot(sourceVector))
		
		errors.append(calcError())
	
	plt.plot(list(range(len(errors))), errors)
	plt.yscale('log')
	plt.show()
	
	estimatedFunction = spectralToFunction(modalToSpectral(estimatedSolution))
	def analyticFunction(x):
		return x**2 / 4 + a * x + a**2

	xList = np.arange(-1.0, 1.0, 0.01)
	yList = np.zeros(len(xList))
	solutionList = np.zeros(len(xList))
	for i in range(len(xList)):
		yList[i] = analyticFunction(xList[i])
		solutionList[i] = estimatedFunction(xList[i])
	
	plt.plot(xList, yList)
	plt.plot(xList, solutionList)
	plt.show()

def linearSolveTest(N, a=5):
	#trying to solve y'=a xy, y(0)=1  -->  y(x)=e^(-a x^2/2)
	chebEngine = ChebEngine(N)
	operatorMatrix = np.zeros((N+1, N+1))
	boundaryVector = np.zeros(N+1)

	sourceVector = np.zeros(N+1)
	sourceVector[N] = 1

	for i in range(N+1):
		modalUnit = modalBasisFunction(chebEngine, i)
		specVector = modalToSpectral(modalUnit)
		specDeriv = spectralDerivative(specVector)
		modalDeriv = spectralToModal(specDeriv)
		unitFunct = spectralToFunction(specVector)
		operatorMatrix[i] = (a * modalUnit * chebEngine.getCollocationPoint(i) + modalDeriv).data

		boundaryVector[i] = unitFunct(0)

	operatorMatrix = np.transpose(operatorMatrix)
	operatorMatrix[N] = boundaryVector

	solutionMatrix = np.linalg.inv(operatorMatrix)

	modalSolution = ModalRepresentation(chebEngine, solutionMatrix.dot(sourceVector))
	spectralSolution = modalToSpectral(modalSolution)
	functionSolution = spectralToFunction(spectralSolution)

	xList = np.arange(-1.0, 1.0, 0.01)
	yList = np.zeros(len(xList))
	solutionList = np.zeros(len(xList))
	for i in range(len(xList)):
		yList[i] = np.exp(-a * xList[i]**2 / 2)
		solutionList[i] = functionSolution(xList[i])
	
	plt.plot(xList, yList)
	plt.plot(xList, solutionList)
	plt.show()

def basicTest(N, a = 5.0):
	def exampleFunction(x: float) -> float:
		return 1 / (1 + np.exp(-a * x))
	def exampleDerivative(x: float) -> float:
		y = exampleFunction(x)
		return a * y * (1 - y)

	chebEngine = ChebEngine(N)

	modalRep = functionToModal(exampleFunction, chebEngine)
	specRep = modalToSpectral(modalRep)
	modalRep = spectralToModal(specRep)
	specRep = modalToSpectral(modalRep)
	approxFunct = spectralToFunction(specRep)

	specD = spectralDerivative(specRep)
	approxDeriv = spectralToFunction(specD)

	xList = np.arange(-1.0, 1.0, 0.01)
	yList = np.zeros(len(xList))
	for i in range(len(xList)):
		yList[i] = exampleFunction(xList[i])

	approxYList = np.zeros(len(xList))
	for i in range(len(xList)):
		approxYList[i] = approxFunct(xList[i])

	plt.plot(xList, yList)
	plt.plot(xList, approxYList)
	plt.show()

	for i in range(len(xList)):
		yList[i] = exampleDerivative(xList[i])
		approxYList[i] = approxDeriv(xList[i])
	plt.plot(xList, yList)
	plt.plot(xList, approxYList)
	plt.show()

scalarFieldTest()
# nonLinearSolveTest(10)
# linearSolveTest(10)
# basicTest(10)
