import numpy as np
import scipy
import matplotlib.pyplot as plt
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
		self.order = len(data) // 2
	
	def __mul__(self, other):
		if type(other) == ModalRepresentation:
			return ModalRepresentation(self.chebEngine, self.data * other.data)
		else:
			return ModalRepresentation(self.chebEngine, self.data * other)
	
	def __div__(self, scalar):
		return ModalRepresentation(self.chebEngine, self.data / scalar)

	def __add__(self, other):
		return ModalRepresentation(self.chebEngine, self.data + other.data)
	
	def __sub__(self, other):
		return ModalRepresentation(self.chebEngine, self.data - other.data)

class SpectralRepresentation:
	def __init__(self, chebEngine: ChebEngine, data: np.ndarray):
		self.chebEngine = chebEngine
		self.data = np.array(data)
		self.order = len(data) // 2
	
	def __mul__(self, scalar):
		return SpectralRepresentation(self.chebEngine, self.data * scalar)
	
	def __div__(self, scalar):
		return SpectralRepresentation(self.chebEngine, self.data / scalar)

	def __add__(self, other):
		return SpectralRepresentation(self.chebEngine, self.data + other.data)
	
	def __sub__(self, other):
		return SpectralRepresentation(self.chebEngine, self.data - other.data)

def functionToModal(funct: Callable[[float],float], chebEngine: ChebEngine, x0 = -1.0, x1 = 1.0) -> ModalRepresentation:
	data = np.zeros(2 * chebEngine.order)
	data[0] = funct(chebEngine.getCollocationPoint(0, x0, x1))
	data[chebEngine.N] = funct(chebEngine.getCollocationPoint(chebEngine.N, x0, x1))
	for i in range(1, chebEngine.N):
		data[i] = funct(chebEngine.getCollocationPoint(i, x0, x1))
		data[2 * chebEngine.N - i] = data[i]
	return ModalRepresentation(chebEngine, data)

def modalToSpectral(modalRep: ModalRepresentation) -> ModalRepresentation:
	N = modalRep.chebEngine.N
	specData = np.fft.rfft(modalRep.data) / N
	specData[0] /= 2
	specData[N] /= 2
	return SpectralRepresentation(modalRep.chebEngine, specData)

def spectralToModal(specRep: SpectralRepresentation) -> SpectralRepresentation:
	N = specRep.chebEngine.N
	specRep.data[0] *= 2
	specRep.data[N] *= 2
	modalData = np.fft.irfft(specRep.data) * N
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
	outputData = np.zeros(N + 1, dtype=complex)
	outputData[N - 1] = 2 * N * specRep.data[N]
	outputData[N - 2] = 2 * (N - 1) * specRep.data[N - 1]
	for i in reversed(range(0, N - 2)):
		outputData[i] = outputData[i + 2] + (i + 1) * specRep.data[i + 1]
	outputData[0] /= 2
	return SpectralRepresentation(specRep.chebEngine, outputData * widthCorrectionFactor)

def modalBasisFunction(chebEngine: ChebEngine, n: int) -> ModalRepresentation:
	outputData = np.zeros(chebEngine.N * 2)
	if n == 0:
		outputData[0] = 1
	else:
		outputData[n] = 1
		outputData[2 * chebEngine.N - n] = 1
	return ModalRepresentation(chebEngine, outputData)

def spectralBasisFunction(chebEngine: ChebEngine, n: int) -> SpectralRepresentation:
	outputData = np.zeros(chebEngine.N * 2)
	outputData[n] = 1
	# if n == 0:
	# 	outputData[0] = 1
	# else:
	# 	outputData[n] = 1
	# 	outputData[2 * chebEngine.N - n] = 1
	return SpectralRepresentation(chebEngine, outputData)

def promoteModalData(chebEngine: ChebEngine, data: np.ndarray) -> ModalRepresentation:
	outputData = np.zeros(chebEngine.N * 2)
	outputData[0] = data[0]
	for i in range(1, chebEngine.N + 1):
		outputData[i] = data[i]
		outputData[2 * chebEngine.N - i] = data[i]
	return ModalRepresentation(chebEngine, outputData)
def promoteSpectralData(chebEngine: ChebEngine, data: np.ndarray) -> SpectralRepresentation:
	outputData = data
	# outputData = np.zeros(chebEngine.N * 2)
	# outputData[0] = data[0]
	# for i in range(1, chebEngine.N + 1):
	# 	outputData[i] = data[i]
	# 	outputData[2 * chebEngine.N - i] = data[i]
	return SpectralRepresentation(chebEngine, outputData)

def demoteSpectralData(spectralRep: SpectralRepresentation) -> np.ndarray:
	return spectralRep.data.real
	# outputData = np.zeros(len(spectralRep.data) // 2 + 1)
	# for i in range(len(outputData)):
	# 	outputData[i] = spectralRep.data[i]
	# return outputData
def demoteModalData(modalRep: ModalRepresentation) -> np.ndarray:
	outputData = np.zeros(len(modalRep.data) // 2 + 1)
	for i in range(len(outputData)):
		outputData[i] = modalRep.data[i]
	return outputData


def linearSolveTest(N):
	chebEngine = ChebEngine(N)
	operatorMatrix = np.zeros((N+1, N+1))
	sourceVector = np.zeros(N+1)
	sourceVector[N] = 1

	for i in range(N+1):
		modalUnit = modalBasisFunction(chebEngine, i)
		print(len(modalUnit.data))
		specVector = modalToSpectral(modalUnit)
		print(len(specVector.data))
		specDeriv = spectralDerivative(specVector)
		print(len(specDeriv.data))
		modalDeriv = spectralToModal(specDeriv)
		print(len(modalDeriv.data))
		unitFunct = spectralToFunction(specVector)
		if i != N:
			operatorMatrix[i] = demoteModalData(modalUnit - modalDeriv)
		operatorMatrix[N][i] = unitFunct(0)

	solutionMatrix = np.linalg.inv(operatorMatrix)

	modalSolution = promoteModalData(chebEngine, solutionMatrix.dot(sourceVector))
	spectralSolution = modalToSpectral(modalSolution)
	functionSolution = spectralToFunction(spectralSolution)

	xList = np.arange(-1.0, 1.0, 0.01)
	yList = np.zeros(len(xList))
	solutionList = np.zeros(len(xList))
	for i in range(len(xList)):
		yList[i] = np.exp(xList[i])
		solutionList[i] = functionSolution(xList[i])
	
	plt.plot(xList, yList)
	plt.plot(xList, solutionList)
	plt.show()

linearSolveTest(10)


def basicTest(N, a = 5.0):
	def exampleFunction(x: float) -> float:
		return 1 / (1 + np.exp(-a * x))
	def exampleDerivative(x: float) -> float:
		y = exampleFunction(x)
		return a * y * (1 - y)


	# def exampleFunction(x: float) -> float:
	# 	return np.sin(0.5 * x) - 0.5*np.cos(1.7 * x)+ 0.2 * np.sin(7.0 * x + 2)
	# def exampleDerivative(x: float) -> float:
	# 	return 0.5 * np.cos(0.5 * x) + 0.5 * 1.7 * np.sin(1.7 * x) + 0.2 * 7.0 * np.cos(7.0 * x + 2)


	# def exampleFunction(x):
	# 	return 1 if x > 0 else 0


	# def exampleFunction(x):
	# 	return np.abs(x)
	# def exampleDerivative(x):
	# 	return 1 if x > 0 else -1

	chebEngine = ChebEngine(N)

	modalRep = functionToModal(exampleFunction, chebEngine)
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
		# for n in range(N + 1):
		# 	approxYList[i] += chebEval(xList[i], n) * coefficients[n]

	plt.plot(xList, yList)
	plt.plot(xList, approxYList)
	# plt.plot(xList, approxYList - yList)
	plt.show()

	for i in range(len(xList)):
		yList[i] = exampleDerivative(xList[i])
		approxYList[i] = approxDeriv(xList[i])
	plt.plot(xList, yList)
	plt.plot(xList, approxYList)
	# plt.plot(xList, approxYList - yList)
	plt.show()

basicTest(10)
