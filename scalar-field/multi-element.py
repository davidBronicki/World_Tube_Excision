import numpy as np
from math import sin, cos
import scipy.integrate
import scipy.fft
from typing import Callable, List, Union, Tuple
import types

import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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

class CoordinateFrame:
	def __init__(self, name: str, dim: int):
		assert(dim > 0), 'non-positive dimension'
		self.name = name
		self.dim = dim

class CoordinatePos:
	def __init__(self, frame: CoordinateFrame, coords: np.ndarray):
		assert(coords.shape == (frame.dim,)), 'incorrect coordinate count'
		self.frame = frame
		self.coords = coords

	def inFrame(self, frame: CoordinateFrame, assertionFailerMessage = 'coordinate not in correct frame'):
		assert(frame is self.frame), assertionFailerMessage

class Geometry:
	def __init__(self, coordFrame: CoordinateFrame):
		self.coordFrame = coordFrame
	def setMetric(self, metricFunction: Callable[[np.ndarray],np.ndarray]):
		def newMetricFunct(_self, position: CoordinatePos) -> np.ndarray:
			position.inFrame(self.coordFrame)
			return metricFunction(position.coords)
		self.metric = types.MethodType(newMetricFunct, self)
	def setChristoffelSymbols(self,
		christoffelFunction: Callable[[np.ndarray],np.ndarray]):

		def newChristoffelFunct(_self, position: CoordinatePos) -> np.ndarray:
			position.inFrame(self.coordFrame)
			return christoffelFunction(position.coords)
		self.christoffelSymbols = types.MethodType(newChristoffelFunct, self)
	def metric(self, position: CoordinatePos) -> np.ndarray:
		raise(NotImplementedError(
			'user left metric undefined: must be set manually'))
	def inverseMetric(self, position: CoordinatePos):
		return np.linalg.inv(self.metric(position))
	def christoffelSymbols(self, position: CoordinatePos) -> np.ndarray:
		raise(NotImplementedError(
			'user left Christoffel symbols undefined: must be set manually'))

class Transformation:
	def __init__(self,
		fromFrame: CoordinateFrame,
		toFrame: CoordinateFrame):

		self.dim = fromFrame.dim
		assert (fromFrame.dim == toFrame.dim), 'different dimensions'
		self.initialFrame = fromFrame
		self.finalFrame = toFrame

	def setInverse(self, inverseTransformation: 'Transformation'):
		def newInverse(_self):
			return inverseTransformation
		self.inverse = types.MethodType(newInverse, self)

	def setMap(self, mapFunct: Callable[[np.ndarray], np.ndarray]):
		def newMap(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.initialFrame, 'incorrect coordinate frame')
			return CoordinatePos(self.finalFrame, mapFunct(position.coords))
		self.map = types.MethodType(newMap, self)

	def setReverseMap(self, mapFunct: Callable[[np.ndarray], np.ndarray]):
		def newMap(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.finalFrame, 'incorrect coordinate frame')
			return CoordinatePos(self.initialFrame, mapFunct(position.coords))
		self.reverseMap = types.MethodType(newMap, self)

	def setJacobian(self,
		jacobianFunct: Callable[[np.ndarray], np.ndarray],
		setInverse = False):

		def newJacobian(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.initialFrame, 'incorrect coordinate frame')
			return jacobianFunct(position.coords)
		self.jacobian = types.MethodType(newJacobian, self)

		if setInverse:
			def newInvJacobian(_self, position: CoordinatePos) -> CoordinatePos:
				position.inFrame(self.initialFrame, 'incorrect coordinate frame')
				return np.linalg.inv(jacobianFunct(position.coords))
			self.inverseJacobian = types.MethodType(newInvJacobian, self)

	def setInverseJacobian(self,
		invJacobianFunct: Callable[[np.ndarray], np.ndarray],
		setJacobian = False):

		def newInvJacobian(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.initialFrame, 'incorrect coordinate frame')
			return invJacobianFunct(position.coords)
		self.inverseJacobian = types.MethodType(newInvJacobian, self)

		if setJacobian:
			def newJacobian(_self, position: CoordinatePos) -> CoordinatePos:
				position.inFrame(self.initialFrame, 'incorrect coordinate frame')
				return np.linalg.inv(invJacobianFunct(position.coords))
			self.jacobian = types.MethodType(newJacobian, self)

	def setGradJacobian(self, gradJacFunct: Callable[[np.ndarray], np.ndarray]):
		def newGradJac(_self, position: CoordinatePos) -> CoordinatePos:
			position.inFrame(self.initialFrame, 'incorrect coordinate frame')
			return gradJacFunct(position.coords)
		self.gradJacobian = types.MethodType(newGradJac, self)

	def inverse(self):
		output = Transformation(
			self.finalFrame,
			self.initialFrame
		)
		output.map = types.MethodType(
			lambda outputSelf, coords: self.reverseMap(coords),
			output)
		output.reverseMap = types.MethodType(
			lambda outputSelf, coords: self.map(coords),
			output)
		output.jacobian = types.MethodType(
			lambda outputSelf, coords: self.inverseJacobian(self.reverseMap(coords)),
			output)
		output.inverseJacobian = types.MethodType(
			lambda outputSelf, coords: self.jacobian(self.reverseMap(coords)),
			output)
		def reversedGradJacobian(outputSelf, coordinates: CoordinatePos) -> np.ndarray:
			coordinates = self.reverseMap(coordinates)
			gradJac = self.gradJacobian(coordinates)
			invJac = self.inverseJacobian(coordinates)
			return np.einsum('bca,ak,cj,ib->ijk',
				gradJac, invJac, invJac, invJac)
		output.gradJacobian = types.MethodType(
			reversedGradJacobian,
			output)
		return output

	def compose(self, other: 'Transformation'):
		assert(self.initialFrame is other.finalFrame), 'incompatable composition'
		output = Transformation(
			other.initialFrame,
			self.finalFrame
		)
		output.map = types.MethodType(
			lambda outputSelf, coords: self.map(other.map(coords)),
			output)
		output.reverseMap = types.MethodType(
			lambda outputSelf, coords: other.reverseMap(self.reverseMap(coords)),
			output)

		def newJacobian(outputSelf, coords: CoordinatePos):
			intermediateCoords = other.map(coords)
			outerJac = self.jacobian(intermediateCoords)
			innerJac = other.jacobian(coords)
			return np.dot(outerJac, innerJac)
		output.jacobian = types.MethodType(newJacobian, output)

		def newInvJacobian(outputSelf, coords: CoordinatePos):
			intermediateCoords = other.map(coords)
			outerJac = self.inverseJacobian(intermediateCoords)
			innerJac = other.inverseJacobian(coords)
			return np.dot(innerJac, outerJac)
		output.inverseJacobian = types.MethodType(newInvJacobian, output)

		def newGradJacobian(outputSelf, coords: CoordinatePos) -> np.ndarray:
			intermediateCoords = other.map(coords)
			innerGradJac = other.gradJacobian(coords)
			innerJac = other.jacobian(coords)
			outerGradJac = self.gradJacobian(intermediateCoords)
			outerJac = self.jacobian(intermediateCoords)

			innerGradJac = np.einsum('ajk,ia->ijk', innerGradJac, outerJac)
			outerGradJac = np.einsum('iab,bk,aj->ijk',
				outerGradJac, innerJac, innerJac)
			
			return innerGradJac + outerGradJac
		output.gradJacobian = types.MethodType(newGradJacobian, output)
		return output

	def map(self, coordinates: CoordinatePos) -> CoordinatePos:
		raise(NotImplementedError(
			'user left map undefined: must be set manually'))

	def reverseMap(self, coordinates: CoordinatePos) -> CoordinatePos:
		raise(NotImplementedError(
			'user left reverse map undefined: must be set manually'))

	def jacobian(self, coordinates: CoordinatePos) -> np.ndarray:
		raise(NotImplementedError(
			'user left jacobian undefined: must be set manually'))

	def inverseJacobian(self, coordinates: CoordinatePos) -> np.ndarray:
		coordinates.inFrame(self.initialFrame, 'incorrect coordinate frame')
		return np.linalg.inv(self.jacobian(coordinates))

	def gradJacobian(self, coordinates: CoordinatePos) -> np.ndarray:
		#TODO: implement numerical gradient?
		raise(NotImplementedError(
			'user left grad jacobian undefined: must be set manually'))

class TensorFieldElement:
	def __init__(self,
		frame: CoordinateFrame,
		chebEngines: List[ChebEngine],
		data: np.ndarray,
		downIndices = 0):

		self.frame = frame
		self.chebEngines = chebEngines
		self.dim = len(chebEngines)
		self.data = data
		self.rank = len(data.shape) - self.dim

		self.downIndices = downIndices
		self.upIndices = self.rank - downIndices

		#validate data dimensions
		assert(self.upIndices >= 0), 'negative up indices'
		assert(self.downIndices >= 0), 'negative down indices'
		for engine, modalCount in zip(chebEngines, data.shape):
			assert(engine.order + 1 == modalCount), 'engine to data mis-match'
		for shouldBeDim in data.shape[self.dim:]:
			assert(shouldBeDim == self.dim), 'data to dimension mis-match'

		self._fullSlice = tuple(
			[slice(0, size) for size in self.data.shape]
		)
	
	@staticmethod
	def tensorProduct(contractionString: str,
		*tensorElements: 'TensorFieldElement'):

		for a, b in zip(tensorElements[:-1], tensorElements[1:]):
			assert(a.frame is b.frame), 'tensor products must be made in same frame'

		dataPack = [element.data for element in tensorElements]

		inputOutputList = contractionString.split('->')

		assert(len(inputOutputList) == 2), 'must specify input and output indices'

		inputIndexLists = inputOutputList[0].split(',')

		assert(len(inputIndexLists) == len(tensorElements)), 'must index each input tensor'

		downIndexCount = 0
		for element, indexString in zip(tensorElements, inputIndexLists):
			for i in range(len(indexString)):
				#if this is a free index (in output index list)
				#and is a down index, then increment the down index counter
				if (indexString[i] in inputOutputList[1] and len(indexString) - i <= element.downIndices):
					downIndexCount += 1

		#setup for broadcasting
		for i in range(len(inputIndexLists)):
			inputIndexLists[i] = '...'+inputIndexLists[i]
		inputIndexString = ','.join(inputIndexLists)
		numpyContractionString = inputIndexString + '->...' + inputOutputList[1]

		outputData = np.einsum(numpyContractionString, *dataPack)
		return TensorFieldElement(
			tensorElements[0].frame,
			tensorElements[0].chebEngines,
			outputData,
			downIndexCount)
			
	@staticmethod
	def defaultInit(
		frame: CoordinateFrame,
		chebEngines: List[ChebEngine],
		upIndices: int,
		downIndices = 0):

		rank = downIndices + upIndices
		modalShape = [engine.order + 1 for engine in chebEngines]
		tensorShape = [len(chebEngines)] * rank
		return TensorFieldElement(
			frame,
			chebEngines,
			np.zeros(modalShape + tensorShape),
			downIndices)

	@staticmethod
	def functionInit(
		frame: CoordinateFrame,
		chebEngines: List[ChebEngine],
		fieldInitializer: Callable[[np.ndarray], np.ndarray],
		upIndices: int,
		downIndices = 0):

		rank = downIndices + upIndices
		modalShape = tuple([engine.order + 1 for engine in chebEngines])
		tensorShape = tuple([len(chebEngines)] * rank)
		data = np.zeros((np.prod(modalShape),) + tensorShape)

		dim = len(chebEngines)
		totalSamplePoints = np.prod(modalShape)
		indices = np.indices(modalShape)
		indices = np.transpose(np.reshape(indices, (dim, totalSamplePoints)))

		for i in range(totalSamplePoints):
			regionPosition = np.zeros(dim)
			for j in range(dim):
				regionPosition[j] = chebEngines[j].getCollocationPoint(indices[i][j])
			data[i] = fieldInitializer(regionPosition)
		data = np.reshape(data, modalShape + tensorShape)

		return TensorFieldElement(
			frame,
			chebEngines,
			data,
			downIndices)

	def toFunction(self) -> Callable[[np.ndarray, Union[None, PreCompute]], float]:

		rank = self.downIndices + self.upIndices
		modalShape = [engine.order + 1 for engine in self.chebEngines]
		tensorShape = tuple([len(self.chebEngines)] * rank)

		totalSamplePoints = np.prod(modalShape)
		indices = np.indices(modalShape)
		indices = np.transpose(np.reshape(indices, (self.dim, totalSamplePoints)))
		
		#change to spectral

		specData = scipy.fft.dct(
			self.data, type=1, axis=0) / self.chebEngines[0].N
		for i in range(1, len(self.chebEngines)):
			specData = scipy.fft.dct(
				specData, type=1, axis=i) / self.chebEngines[i].N

		for indexList in indices:
			for i in range(self.dim):
				if indexList[i] == 0 or indexList[i] == self.chebEngines[i].N:
					specData[tuple(indexList)] /= 2

		def outputFunct(position: np.ndarray, preComputeArray: Union[None, PreCompute] = None) -> float:
			if preComputeArray == None:
				output = np.zeros(tensorShape)
				for indexList in indices:
					modeContribution = np.array(specData[tuple(indexList)])
					for i in range(self.dim):
						modeContribution *= self.chebEngines[i].eval(
							indexList[i], position[i])
					output += modeContribution
				return output
			else:
				output = np.zeros(tensorShape)
				for indexList in indices:
					output += specData[tuple(indexList)] * preComputeArray(indexList, position)
				return output

		return outputFunct

	def __mul__(self, other: Union['TensorFieldElement', int, float]):
		if type(other) == TensorFieldElement:
			#TODO: implement this and add checks
			assert(self.frame is other.frame), 'multiplication must be made in same frame'
			assert(self.rank == 0 or other.rank == 0), 'multiplication must be with scalar. Try tensorProduct'
			return TensorFieldElement(
				self.frame,
				self.chebEngines,
				#to make the broadcast work correctly, we reverse all axes.
				#this makes the broadcast match the first axes going forward
				#instead of the last axes going backwards as is default.
				np.transpose(np.transpose(self.data) * np.transpose(other.data)),
				self.downIndices + other.downIndices#one of these will be zero.
			)
		else:
			return TensorFieldElement(
				self.frame,
				self.chebEngines,
				self.data * other,
				self.downIndices
			)
	__rmul__ = __mul__

	def __truediv__(self, scalar: Union[int, float]):
		if (scalar == 0 or scalar == 0.):
			raise(ZeroDivisionError('tensor divided by zero scalar'))
		return TensorFieldElement(
			self.frame,
			self.chebEngines,
			self.data / scalar,
			self.downIndices)

	def __add__(self, other: 'TensorFieldElement'):
		#require compatible tensors
		assert(self.frame is other.frame), 'addition must be made in same frame'
		assert(self.upIndices == other.upIndices), 'addition with different up indices'
		assert(self.downIndices == other.downIndices), 'addition with different down indices'
		return TensorFieldElement(
			self.frame,
			self.chebEngines,
			self.data + other.data,
			self.downIndices)

	def __sub__(self, other: 'TensorFieldElement'):
		#require compatible tensors
		assert(self.frame is other.frame), 'subtraction must be made in same frame'
		assert(self.upIndices == other.upIndices), 'subtraction with different up indices'
		assert(self.downIndices == other.downIndices), 'subtraction with different down indices'
		return TensorFieldElement(
			self.frame,
			self.chebEngines,
			self.data - other.data,
			self.downIndices)

	def __neg__(self):
		return TensorFieldElement(
			self.frame,
			self.chebEngines,
			-self.data,
			self.downIndices)

	def _partialDerivative(self, axis: int) -> np.ndarray:
		N = self.chebEngines[axis].N

		#for generating advanced slicing sets
		def generateSurfaceSlice(index: int):
			output: list[Union[slice, int]] = list(self._fullSlice)
			output[axis] = index
			return tuple(output)

		#change to spectral along given axis
		specData: np.ndarray = scipy.fft.dct(self.data, type=1, axis=axis) / N
		specData[generateSurfaceSlice(0)] /= 2
		specData[generateSurfaceSlice(N)] /= 2

		#perform derivative in spectral basis
		specDerivativeData: np.ndarray = np.zeros(specData.shape)
		specDerivativeData[generateSurfaceSlice(N - 1)] =\
			2 * N * specData[generateSurfaceSlice(N)]
		specDerivativeData[generateSurfaceSlice(N - 2)] =\
			2 * (N - 1) * specData[generateSurfaceSlice(N - 1)]
		for i in reversed(range(0, N - 2)):
			specDerivativeData[generateSurfaceSlice(i)] =\
				specDerivativeData[generateSurfaceSlice(i + 2)]\
				+ 2 * (i + 1) * specData[generateSurfaceSlice(i + 1)]

		#change back to modal representation
		#(above algorithm computes 0th element off by factor of two,
		# so don't need to adjust here due to cancellation)
		specDerivativeData[generateSurfaceSlice(N)] *= 2
		return scipy.fft.dct(specDerivativeData, type=1, axis=axis) / 2

	def coordinateGradient(self):
		#for generating advanced slicing sets
		fullSlice = tuple(
			[slice(0, size) for size in self.data.shape]
		) + (slice(0, self.dim),)
		def generateGradientSlice(axis: int):
			output: list[Union[slice, int]] = list(fullSlice)
			output[-1] = axis
			return tuple(output)

		outputData: np.ndarray = np.zeros(self.data.shape + (self.dim,))
		for axis in range(self.dim):
			outputData[generateGradientSlice(axis)] = self._partialDerivative(axis)
		return TensorFieldElement(
			self.frame,
			self.chebEngines,
			outputData,
			self.downIndices + 1
		)

	def gradient(self,
		geometry: Geometry,
		transformationFromGlobalCoords: Transformation):
		raise(NotImplementedError('no gradient yet'))

class BoundaryView:
	def __init__(self,
		element: TensorFieldElement,
		axis: int,
		isRightBoundary: bool,
		viewRotationFunction: Callable[[np.ndarray], np.ndarray] = lambda x: x
	):
		self.element = element
		self.axis = axis
		self.isRightBoundary = isRightBoundary
		self.isLeftBoundary = not isRightBoundary

		boundarySlice = list(element._fullSlice)
		boundarySlice[axis] = 0 if isRightBoundary else -1
		boundarySlice = tuple(boundarySlice)

		self.boundaryData = viewRotationFunction(element.data[boundarySlice])

	def copy(self):
		return BoundaryData(self.element.frame,
			self.element.chebEngines[:self.axis]+self.element.chebEngines[self.axis+1:],
			np.array(self.boundaryData),
			self.axis,
			self.isRightBoundary,
			self.element.downIndices)

	def set(self, boundaryData: 'BoundaryData'):
		self.boundaryData[:] = boundaryData.data

class BoundaryData:
	def __init__(self,
		frame: CoordinateFrame,
		chebEngines: List[ChebEngine],
		data: np.ndarray,
		axis: int,
		isRightBoundary: bool,
		downIndices = 0):

		self.frame = frame
		self.chebEngines = chebEngines
		self.dim = len(chebEngines) + 1
		self.data = data
		self.rank = len(data.shape) - self.dim + 1
		self.axis = axis
		self.isRightBoundary = isRightBoundary
		self.isLeftBoundary = not isRightBoundary
		self.downIndices = downIndices
		self.upIndices = self.rank - downIndices

		#validate data dimensions
		assert(self.upIndices >= 0), 'negative up indices'
		assert(self.downIndices >= 0), 'negative down indices'
		for engine, modalCount in zip(chebEngines, data.shape):
			assert(engine.order + 1 == modalCount), 'engine to data mis-match'
		for shouldBeDim in data.shape[self.dim - 1:]:
			assert(shouldBeDim == self.dim), 'data to dimension mis-match'

		self._fullSlice = tuple(
			[slice(0, size) for size in self.data.shape]
		)
	
	@staticmethod
	def tensorProduct(contractionString: str,
		*boundaryElements: 'BoundaryData'):

		for a, b in zip(boundaryElements[:-1], boundaryElements[1:]):
			assert(a.frame is b.frame), 'tensor products must be made in same frame'
			assert(a.isRightBoundary == b.isRightBoundary), 'tensor product only on equivalent boundary'
			assert(a.axis == b.axis), 'tensor product only on equivalent boundary'

		dataPack = [element.data for element in boundaryElements]

		inputOutputList = contractionString.split('->')

		assert(len(inputOutputList) == 2), 'must specify input and output indices'

		inputIndexLists = inputOutputList[0].split(',')

		assert(len(inputIndexLists) == len(boundaryElements)), 'must index each input tensor'

		downIndexCount = 0
		for element, indexString in zip(boundaryElements, inputIndexLists):
			for i in range(len(indexString)):
				#if this is a free index (in output index list)
				#and is a down index, then increment the down index counter
				if (indexString[i] in inputOutputList[1] and len(indexString) - i <= element.downIndices):
					downIndexCount += 1

		#setup for broadcasting
		for i in range(len(inputIndexLists)):
			inputIndexLists[i] = '...'+inputIndexLists[i]
		inputIndexString = ','.join(inputIndexLists)
		numpyContractionString = inputIndexString + '->...' + inputOutputList[1]

		outputData = np.einsum(numpyContractionString, *dataPack)
		return BoundaryData(
			boundaryElements[0].frame,
			boundaryElements[0].chebEngines,
			outputData,
			boundaryElements[0].axis,
			boundaryElements[0].isRightBoundary,
			downIndexCount)
			
	@staticmethod
	def defaultInit(
		frame: CoordinateFrame,
		chebEngines: List[ChebEngine],
		upIndices: int,
		downIndices = 0):

		rank = downIndices + upIndices
		modalShape = [engine.order + 1 for engine in chebEngines]
		tensorShape = [len(chebEngines) + 1] * rank
		return BoundaryData(
			frame,
			chebEngines,
			np.zeros(modalShape + tensorShape),
			downIndices)

	def __mul__(self, other: Union['BoundaryData', int, float]):
		if type(other) == BoundaryData:
			#TODO: implement this and add checks
			assert(self.frame is other.frame), 'multiplication must be made in same frame'
			assert(self.isRightBoundary == other.isRightBoundary), 'multiplication only on equivalent boundary'
			assert(self.axis == other.axis), 'multiplication only on equivalent boundary'
			assert(self.rank == 0 or other.rank == 0), 'multiplication must be with scalar. Try tensorProduct'
			return BoundaryData(
				self.frame,
				self.chebEngines,
				#to make the broadcast work correctly, we reverse all axes.
				#this makes the broadcast match the first axes going forward
				#instead of the last axes going backwards as is default.
				np.transpose(np.transpose(self.data) * np.transpose(other.data)),
				self.axis,
				self.isRightBoundary,
				self.downIndices + other.downIndices#one of these will be zero.
			)
		else:
			return BoundaryData(
				self.frame,
				self.chebEngines,
				self.data * other,
				self.axis,
				self.isRightBoundary,
				self.downIndices
			)
	__rmul__ = __mul__

	def __truediv__(self, scalar: Union[int, float]):
		if (scalar == 0 or scalar == 0.):
			raise(ZeroDivisionError('tensor divided by zero scalar'))
		return BoundaryData(
			self.frame,
			self.chebEngines,
			self.data / scalar,
			self.axis,
			self.isRightBoundary,
			self.downIndices)

	def __add__(self, other: 'TensorFieldElement'):
		#require compatible tensors
		assert(self.frame is other.frame), 'addition must be made in same frame'
		assert(self.isRightBoundary == other.isRightBoundary), 'addition only on equivalent boundary'
		assert(self.axis == other.axis), 'addition only on equivalent boundary'
		assert(self.upIndices == other.upIndices), 'addition with different up indices'
		assert(self.downIndices == other.downIndices), 'addition with different down indices'
		return BoundaryData(
			self.frame,
			self.chebEngines,
			self.data + other.data,
			self.axis,
			self.isRightBoundary,
			self.downIndices)

	def __sub__(self, other: 'TensorFieldElement'):
		#require compatible tensors
		assert(self.frame is other.frame), 'subtraction must be made in same frame'
		assert(self.isRightBoundary == other.isRightBoundary), 'subtraction only on equivalent boundary'
		assert(self.axis == other.axis), 'subtraction only on equivalent boundary'
		assert(self.upIndices == other.upIndices), 'subtraction with different up indices'
		assert(self.downIndices == other.downIndices), 'subtraction with different down indices'
		return BoundaryData(
			self.frame,
			self.chebEngines,
			self.data - other.data,
			self.axis,
			self.isRightBoundary,
			self.downIndices)

	def __neg__(self):
		return BoundaryData(
			self.frame,
			self.chebEngines,
			-self.data,
			self.axis,
			self.isRightBoundary,
			self.downIndices)

class Interface:
	def __init__(self,
		boundaryView1: BoundaryView,
		boundaryView2: BoundaryView,
		normalCovectorField1: BoundaryData,
		transformation: Transformation
		):

		self.boundaryView1 = boundaryView1
		self.boundaryView2 = boundaryView2
		self.transformation_1to2 = transformation
		self.transformation_2to1 = transformation.inverse()
		self.normalCovectorField1 = normalCovectorField1

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

	chebEngine = ChebEngine(N)
	engines = [chebEngine, chebEngine]
	globalFrame = CoordinateFrame('global', 2)

	geometry = Geometry(globalFrame)
	geometry.setMetric(lambda x: np.identity(2))
	
	metric = TensorFieldElement.functionInit(
		globalFrame, engines,
		lambda x: geometry.metric(CoordinatePos(
			globalFrame, x
		)),
		0, 2
	)
	inverseMetric =  TensorFieldElement.functionInit(
		globalFrame, engines,
		lambda x: geometry.inverseMetric(CoordinatePos(
			globalFrame, x
		)),
		0, 2
	)

	xHatField = np.transpose(np.array([np.ones(N+1), np.zeros(N+1)]))
	yHatField = np.transpose(np.array([np.zeros(N+1), np.ones(N+1)]))
	rightBoundaryNormal = BoundaryData(
		globalFrame,
		[chebEngine],
		xHatField,
		0, True, 1)
	leftBoundaryNormal = BoundaryData(
		globalFrame,
		[chebEngine],
		-xHatField,
		0, False, 1)
	topBoundaryNormal = BoundaryData(
		globalFrame,
		[chebEngine],
		yHatField,
		1, True, 1)
	bottomBoundaryNormal = BoundaryData(
		globalFrame,
		[chebEngine],
		-yHatField,
		1, False, 1)

	def initialPhiFunct(x):
		return 0.
	def initialPiFunct(x):
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

	def initialPhi():
		return TensorFieldElement.functionInit(globalFrame, engines, initialPhiFunct, 0)
	def initialPi():
		return TensorFieldElement.functionInit(globalFrame, engines, initialPiFunct, 0)

	def source(t):
		f = lambda x: sourceFunct(x, t)
		return TensorFieldElement.functionInit(globalFrame, engines, f, 0)

	side=(N+1)
	dim = 2
	volume=(N+1)**dim

	def unpackDataVector(dataVector: np.ndarray):
		phi = TensorFieldElement(globalFrame, engines, np.reshape(
			dataVector[0:volume], (side, side)))
		pi = TensorFieldElement(globalFrame, engines, np.reshape(
			dataVector[volume : 2*volume], (side, side)))
		gamma = TensorFieldElement(globalFrame, engines, np.reshape(
			dataVector[2*volume : 4*volume], (side, side, dim)), 1)
		return phi, pi, gamma

	def packDataVector(
		phi: TensorFieldElement,
		pi: TensorFieldElement,
		gamma: TensorFieldElement
	):
		return np.concatenate((
			phi.data.flatten(),
			pi.data.flatten(),
			gamma.data.flatten()
		))

	def handleBoundary(phi, pi, gamma,
		axis,
		rightBoundary: bool,
		normalVector: BoundaryData):

		piBoundary = BoundaryView(pi, axis, rightBoundary)
		gammaBoundary = BoundaryView(gamma, axis, rightBoundary)

		metricBoundaryData = BoundaryView(inverseMetric, axis, rightBoundary).copy()

		inflow = piBoundary.copy() + BoundaryData.tensorProduct(
			'i,j,ij-> ',
			gammaBoundary.copy(), normalVector, metricBoundaryData)

		piBoundary.set(piBoundary.copy() - inflow / 2)
		gammaBoundary.set(gammaBoundary.copy() - normalVector * inflow / 2)

	def stateDot(time, state):
		phi, pi, gamma = unpackDataVector(state)

		handleBoundary(phi, pi, gamma, 0, False, leftBoundaryNormal)
		handleBoundary(phi, pi, gamma, 0, True, rightBoundaryNormal)
		handleBoundary(phi, pi, gamma, 1, False, bottomBoundaryNormal)
		handleBoundary(phi, pi, gamma, 1, True, topBoundaryNormal)

		gradPi = pi.coordinateGradient()
		gradGamma = gamma.coordinateGradient()

		phiDot = pi
		piDot = TensorFieldElement.tensorProduct('ii->', gradGamma) + source(time)
		gammaDot = gradPi
		
		return packDataVector(phiDot, piDot, gammaDot)


	initPhi = initialPhi()
	initPi = initialPi()
	initGamma = initPhi.coordinateGradient()

	initState = np.concatenate((
		initPhi.data.flatten(),
		initPi.data.flatten(),
		initGamma.data.flatten(),
	))

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

	phiDataSet = []

	preComputedArray = PreCompute(engines, [xData, yData])
	for outputData in outputDataSet:
		phi, pi, gamma = unpackDataVector(outputData)

		phiFunct = phi.toFunction()
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

def scalar_2D_polar(
	N = 8,
	animationDuration = 2.,
	simDuration = 3.0,
	dt = 0.1,
	display_dx = 0.1):

	globalFrame = CoordinateFrame('global', 2)
	polarFrame = CoordinateFrame('polar', 2)

	def map_polarToCart(coords):
		x = coords[0] * np.cos(coords[1])
		y = coords[0] * np.sin(coords[1])
		return np.array([x, y])

	def map_cartToPolar(coords):
		r = np.sqrt(np.dot(coords, coords))
		theta = np.arctan2(coords[1], coords[0])
		return np.array([r, theta])

	def jac_polarToCart(coords):
		r = coords[0]
		theta = coords[1]
		return np.array(
			[
				[np.cos(theta),-r * np.sin(theta)],
				[np.sin(theta), r * np.cos(theta)]
			]
		)

	polarToCartTransformation = Transformation(
		polarFrame, globalFrame
	)
	polarToCartTransformation.setMap(map_polarToCart)
	polarToCartTransformation.setReverseMap(map_cartToPolar)
	polarToCartTransformation.setJacobian(jac_polarToCart)

	t0 = time.time()

	rData = np.arange(-1., 1. + display_dx / 2, display_dx)
	thetaData = np.arange(-1., 1. + display_dx / 2, display_dx)
	meshR, meshTheta = np.meshgrid(rData, thetaData)
	tData = np.arange(0, simDuration, dt)

	chebEngine = ChebEngine(N)
	engines = [chebEngine, chebEngine]
	globalFrame = CoordinateFrame('global', 2)

	geometry = Geometry(globalFrame)
	geometry.setMetric(lambda x: np.identity(2))
	
	metric = TensorFieldElement.functionInit(
		globalFrame, engines,
		lambda x: geometry.metric(CoordinatePos(
			globalFrame, x
		)),
		0, 2
	)
	inverseMetric =  TensorFieldElement.functionInit(
		globalFrame, engines,
		lambda x: geometry.inverseMetric(CoordinatePos(
			globalFrame, x
		)),
		0, 2
	)

	xHatField = np.transpose(np.array([np.ones(N+1), np.zeros(N+1)]))
	yHatField = np.transpose(np.array([np.zeros(N+1), np.ones(N+1)]))
	rightBoundaryNormal = BoundaryData(
		globalFrame,
		[chebEngine],
		xHatField,
		0, True, 1)
	leftBoundaryNormal = BoundaryData(
		globalFrame,
		[chebEngine],
		-xHatField,
		0, False, 1)
	topBoundaryNormal = BoundaryData(
		globalFrame,
		[chebEngine],
		yHatField,
		1, True, 1)
	bottomBoundaryNormal = BoundaryData(
		globalFrame,
		[chebEngine],
		-yHatField,
		1, False, 1)

	def initialPhiFunct(x):
		# mask = min((1 - abs(x[0])) * (1 - abs(x[1])), 0.1) * 10
		# r = np.sqrt(x[0]**2 + x[1]**2)
		# return A*r**2 * np.exp(-a * (r - b)) * mask
		return 0.
	def initialPiFunct(x):
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

	def initialPhi():
		return TensorFieldElement.functionInit(globalFrame, engines, initialPhiFunct, 0)
	def initialPi():
		return TensorFieldElement.functionInit(globalFrame, engines, initialPiFunct, 0)

	def source(t):
		f = lambda x: sourceFunct(x, t)
		return TensorFieldElement.functionInit(globalFrame, engines, f, 0)

	side=(N+1)
	dim = 2
	volume=(N+1)**dim

	def unpackDataVector(dataVector: np.ndarray):
		phi = TensorFieldElement(globalFrame, engines, np.reshape(
			dataVector[0:volume], (side, side)))
		pi = TensorFieldElement(globalFrame, engines, np.reshape(
			dataVector[volume : 2*volume], (side, side)))
		gamma = TensorFieldElement(globalFrame, engines, np.reshape(
			dataVector[2*volume : 4*volume], (side, side, dim)), 1)
		return phi, pi, gamma

	def packDataVector(
		phi: TensorFieldElement,
		pi: TensorFieldElement,
		gamma: TensorFieldElement
	):
		return np.concatenate((
			phi.data.flatten(),
			pi.data.flatten(),
			gamma.data.flatten()
		))

	def handleCyclicInterface(
		phi, pi, gamma,
		axis,
		rightNormalVector: BoundaryData,
		leftNormalVector: BoundaryData):

		rightBoundaryPi = BoundaryView(pi, axis, True)
		leftBoundaryPi = BoundaryView(pi, axis, False)
		rightBoundaryGamma = BoundaryView(gamma, axis, True)
		leftBoundaryGamma = BoundaryView(gamma, axis, False)
		rightBoundaryMetricData = BoundaryView(inverseMetric, axis, True).copy()
		leftBoundaryMetricData = BoundaryView(inverseMetric, axis, False).copy()

		outflowRight = rightBoundaryPi.copy() - BoundaryData.tensorProduct(
			'i,j,ij-> ',
			rightBoundaryGamma.copy(), rightNormalVector, rightBoundaryMetricData)
		outflowLeft = leftBoundaryPi.copy() - BoundaryData.tensorProduct(
			'i,j,ij-> ',
			leftBoundaryGamma.copy(), leftNormalVector, leftBoundaryMetricData)

		inflowRight = rightBoundaryPi.copy() + BoundaryData.tensorProduct(
			'i,j,ij-> ',
			rightBoundaryGamma.copy(), rightNormalVector, rightBoundaryMetricData)
		inflowLeft = leftBoundaryPi.copy() + BoundaryData.tensorProduct(
			'i,j,ij-> ',
			leftBoundaryGamma.copy(), leftNormalVector, leftBoundaryMetricData)

		outflowRight.isRightBoundary = inflowLeft.isRightBoundary
		outflowLeft.isRightBoundary = inflowRight.isRightBoundary

		rightBoundaryPi.set(rightBoundaryPi.copy() - inflowRight / 2 + outflowLeft / 2)
		leftBoundaryPi.set(leftBoundaryPi.copy() - inflowLeft / 2 + outflowRight / 2)

		rightBoundaryGamma.set(rightBoundaryGamma.copy() + rightNormalVector * (
			- inflowRight / 2 + outflowLeft / 2
		))
		leftBoundaryGamma.set(leftBoundaryGamma.copy() + leftNormalVector * (
			- inflowLeft / 2 + outflowRight / 2
		))

	def handleBoundary(phi, pi, gamma,
		axis,
		rightBoundary: bool,
		normalVector: BoundaryData):

		piBoundary = BoundaryView(pi, axis, rightBoundary)
		gammaBoundary = BoundaryView(gamma, axis, rightBoundary)

		metricBoundaryData = BoundaryView(inverseMetric, axis, rightBoundary).copy()

		inflow = piBoundary.copy() + BoundaryData.tensorProduct(
			'i,j,ij-> ',
			gammaBoundary.copy(), normalVector, metricBoundaryData)

		piBoundary.set(piBoundary.copy() - inflow / 2)
		gammaBoundary.set(gammaBoundary.copy() - normalVector * inflow / 2)

	def stateDot(time, state):
		phi, pi, gamma = unpackDataVector(state)

		handleCyclicInterface(phi, pi, gamma, 0, rightBoundaryNormal, leftBoundaryNormal)
		handleBoundary(phi, pi, gamma, 1, True, topBoundaryNormal)
		handleBoundary(phi, pi, gamma, 1, False, bottomBoundaryNormal)
		# handleBoundary(phi, pi, gamma, 0, False, leftBoundaryNormal)
		# handleBoundary(phi, pi, gamma, 0, True, rightBoundaryNormal)
		# handleBoundary(phi, pi, gamma, 1, False, bottomBoundaryNormal)
		# handleBoundary(phi, pi, gamma, 1, True, topBoundaryNormal)

		gradPi = pi.coordinateGradient()
		gradGamma = gamma.coordinateGradient()

		phiDot = pi
		piDot = TensorFieldElement.tensorProduct('ii->', gradGamma) + source(time)
		gammaDot = gradPi
		
		return packDataVector(phiDot, piDot, gammaDot)


	initPhi = initialPhi()
	initPi = initialPi()
	initGamma = initPhi.coordinateGradient()

	initState = np.concatenate((
		initPhi.data.flatten(),
		initPi.data.flatten(),
		initGamma.data.flatten(),
	))

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

	preComputedArray = PreCompute(engines, [rData, thetaData])
	for outputData in outputDataSet:
		phi, pi, gamma = unpackDataVector(outputData)

		phiFunct = phi.toFunction()
		phiData = np.zeros((len(rData), len(thetaData)))
		for i in range(len(rData)):
			for j in range(len(thetaData)):
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


def chebTest():
	# N = 10
	N = 6
	chebEngine = ChebEngine(N)

	def bump(x, x0, width):
		r = (x - x0) / width
		r = np.sqrt(np.dot(r, r))
		return np.exp(-r**2)
		# r = (x - x0) / width * np.pi / 2
		# r = np.sqrt(np.dot(r, r))
		# return np.cos(r)**2 if r < (np.pi / 2) else 0
	def testFunction(x):
		return bump(x, np.array([0.]), 0.15)

	globalFrame = CoordinateFrame('global', 1)
	testModal = TensorFieldElement.functionInit(globalFrame, [chebEngine], testFunction, 0)
	newFunction = testModal.toFunction()

	diffModal = testModal.coordinateGradient()
	tempFunction = diffModal.toFunction()
	diffFunction = lambda position: tempFunction(position)[0]

	# def windowTranslation(x0, x1, x0p, x1p, x):
	# 	return (x - x0) / (x1 - x0) * (x1p - x0p) + x0p

	# cutoff = 0.2

	# leftFunct = lambda x: testFunction(windowTranslation(-1., 1., -1., -cutoff, x))
	# middleFunct = lambda x: testFunction(windowTranslation(-1., 1., -cutoff, cutoff, x))
	# rightFunct = lambda x: testFunction(windowTranslation(-1., 1., cutoff, 1., x))

	# newLeft = modalToFunction(functionToModal(leftFunct, [chebEngine]))
	# newMiddle = modalToFunction(functionToModal(middleFunct, [chebEngine]))
	# newRight = modalToFunction(functionToModal(rightFunct, [chebEngine]))

	# def newFunction(x):
	# 	if x < -0.15:
	# 		return newLeft(windowTranslation(-1., -cutoff, -1., 1., x))
	# 	elif x < 0.15:
	# 		return newMiddle(windowTranslation(-cutoff, cutoff, -1., 1., x))
	# 	else:
	# 		return newRight(windowTranslation(cutoff, 1., -1., 1., x))

	xData = np.arange(-1.0, 1.005, 0.01)
	# xData = np.arange(-1.0, 1.05, 0.1)
	trueY = np.zeros(len(xData))
	newY = np.zeros(len(xData))
	DY = np.zeros(len(xData))

	for i in range(len(xData)):
		trueY[i] = testFunction(np.array([xData[i]]))
		newY[i] = newFunction(np.array([xData[i]]))
		DY[i] = diffFunction(np.array([xData[i]]))
	# print(newY)
	# print(DY)

	plt.plot(xData, trueY)
	plt.plot(xData, newY)
	plt.plot(xData, DY)
	plt.show()

chebTest()
# scalar_2D(N=12)
scalar_2D(animationDuration=5., N=24, simDuration=5.0, dt=0.1)
