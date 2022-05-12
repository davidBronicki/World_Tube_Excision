import numpy as np
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

	def inFrame(self, frame: CoordinateFrame, assertionFailerMessage: str):
		assert(frame is not self.frame), assertionFailerMessage

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
			gradJac = np.tensordot(gradJac, invJac, (0, 1))
			gradJac = np.tensordot(gradJac, invJac, (1, 0))
			gradJac = np.tensordot(gradJac, invJac, (2, 0))
			return gradJac
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
			
			outerGradJac = np.tensordot(outerGradJac, innerJac, (1, 0))
			outerGradJac = np.tensordot(outerGradJac, innerJac, (2, 0))
			innerGradJac = np.tensordot(innerGradJac, outerJac, (0, 1))
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
		chebEngines: List[ChebEngine],
		data: np.ndarray,
		downIndices = 0):

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
			assert(engine.order + 1 is modalCount), 'engine to data mis-match'
		for shouldBeDim in data.shape[self.dim:]:
			assert(shouldBeDim is self.dim), 'data to dimension mis-match'

	def defaultInit(
		chebEngines: List[ChebEngine],
		upIndices: int,
		downIndices = 0):

		rank = downIndices + upIndices
		modalShape = [engine.order + 1 for engine in chebEngines]
		tensorShape = [len(chebEngines)] * rank
		return TensorFieldElement(
			chebEngines,
			np.zeros(modalShape + tensorShape),
			downIndices)

	def __mul__(self, other: Union['TensorFieldElement', int, float]):
		if type(other) == TensorFieldElement:
			#TODO: implement this and add checks
			raise(NotImplementedError('scalar field multiply not implemented'))
		else:
			return TensorFieldElement(
				self.chebEngines,
				self.data * other,
				self.downIndices
			)
	__rmul__ = __mul__
	def tensorProduct(self,
		other: 'TensorFieldElement',
		contractions: List[Tuple[int, int]]):

		raise(NotImplementedError('no tensor product yet'))
	def trace(self,
		contractions: List[Tuple[int, int]]):

		raise(NotImplementedError('no trace yet'))

	def __truediv__(self, scalar: Union[int, float]):
		if (scalar == 0 or scalar == 0.):
			raise(ZeroDivisionError('tensor divided by zero scalar'))
		return TensorFieldElement(
			self.chebEngines,
			self.data / scalar,
			self.downIndices)

	def __add__(self, other: 'TensorFieldElement'):
		#require compatible tensors
		assert(self.upIndices is other.upIndices), 'incompatable up indices in addition'
		assert(self.downIndices is other.downIndices), 'incompatable down indices in addition'
		assert(self.dim is other.dim), 'incompatable dimensions in addition'
		return TensorFieldElement(
			self.chebEngines,
			self.data + other.data,
			self.downIndices)

	def __sub__(self, other: 'TensorFieldElement'):
		#require compatible tensors
		assert(self.upIndices is other.upIndices), 'incompatable up indices in subtraction'
		assert(self.downIndices is other.downIndices), 'incompatable down indices in subtraction'
		assert(self.dim is other.dim), 'incompatable dimensions in subtraction'
		return TensorFieldElement(
			self.chebEngines,
			self.data - other.data,
			self.downIndices)

	def __neg__(self):
		return TensorFieldElement(
			self.chebEngines,
			-self.data,
			self.downIndices)
	
	def _partialDerivative(self, axis: int) -> np.ndarray:
		N = self.chebEngines[axis].N

		#for generating advanced slicing sets
		fullSlice = tuple(
			[slice(0, size) for size in self.data.shape]
		)
		def generateSurfaceSlice(index: int):
			output: list[Union[slice, int]] = list(fullSlice)
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
		def generateGradientSlice(index: int):
			output: list[Union[slice, int]] = list(fullSlice)
			output[-1] = index
			return tuple(output)

		outputData: np.ndarray = np.zeros(self.data.shape + (self.dim,))
		for axis in range(self.dim):
			outputData[generateGradientSlice(axis)] = self._partialDerivative(axis)
		return TensorFieldElement(
			self.chebEngines,
			outputData,
			self.downIndices + 1
		)

	def gradient(self,
		geometry: Geometry,
		transformationFromGlobalCoords: Transformation):
		raise(NotImplementedError('no gradient yet'))

def testCoords():
	cartesian = CoordinateFrame('Cartesian', 3)
	cylindrical = CoordinateFrame('Cylindrical', 3)
	spherical = CoordinateFrame('Spherical', 3)

	cartToCyl = Transformation(cartesian, cylindrical)
