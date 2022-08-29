import numpy as np
import scipy
from typing import Callable, List, Union

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


class TensorFieldElement:
	def __init__(self,
		dim: int,
		data: np.ndarray):

		self.spacialDimensions = dim
		self.data = data
		self.rank = len(data.shape) - self.spacialDimensions

		assert(self.rank >= 0), 'negative tensor rank'

		self._fullSlice = tuple(
			[slice(0, size) for size in self.data.shape]
		)

		self.isView = self.data.base is not None

	def copy(self):
		return TensorFieldElement(self.spacialDimensions, self.data.copy())

	def ownYourData(self):
		if self.isView:
			self.data = self.data.copy()
			self.isView = False

	@staticmethod
	def tensorProduct(contractionString: str,
		*tensorElements: 'TensorFieldElement'):

		dataPack = [element.data for element in tensorElements]

		inputOutputList = contractionString.split('->')

		assert(len(inputOutputList) == 2), 'must specify input and output indices'

		inputIndexLists = inputOutputList[0].split(',')

		assert(len(inputIndexLists) == len(tensorElements)
			   ), 'must index each input tensor'

		# downIndexCount = 0
		# for element, indexString in zip(tensorElements, inputIndexLists):
		# 	for i in range(len(indexString)):
		# 		# if this is a free index (in output index list)
		# 		# and is a down index, then increment the down index counter
		# 		if (indexString[i] in inputOutputList[1] and len(indexString) - i <= element.downIndices):
		# 			downIndexCount += 1

		# setup for broadcasting
		for i in range(len(inputIndexLists)):
			inputIndexLists[i] = '...'+inputIndexLists[i]
		inputIndexString = ','.join(inputIndexLists)
		numpyContractionString = inputIndexString + \
			'->...' + inputOutputList[1]

		outputData = np.einsum(numpyContractionString, *dataPack)
		return TensorFieldElement(
			tensorElements[0].spacialDimensions,
			outputData)

	@staticmethod
	def defaultInit(
		gridShape: List[int],
		tensorShape: List[int]):

		if len(gridShape) != 1 and tensorShape == (1,):
			tensorShape = tensorShape[:-1]

		return TensorFieldElement(
			len(gridShape),
			np.zeros(gridShape + tensorShape))

	@staticmethod
	def functionInit(
		chebOrders: List[int],
		fieldInitializer: Callable[[np.ndarray], np.ndarray]):

		chebEngines: List[ChebEngine] = []
		for axis in chebOrders:
			chebEngines.append(ChebEngine(axis))
		gridShape = tuple([engine.order + 1 for engine in chebEngines])
		tensorShape = fieldInitializer(np.zeros(len(gridShape))).shape

		if len(chebOrders) != 1 and tensorShape == (1,):
			tensorShape = tensorShape[:-1]

		data = np.zeros((np.prod(gridShape),) + tensorShape)

		dim = len(chebEngines)
		totalSamplePoints = np.prod(gridShape)
		indices = np.indices(gridShape)
		indices = np.transpose(np.reshape(indices, (dim, totalSamplePoints)))

		for i in range(totalSamplePoints):
			regionPosition = np.zeros(dim)
			for j in range(dim):
				regionPosition[j] = chebEngines[j].getCollocationPoint(
					indices[i][j])
			data[i] = fieldInitializer(regionPosition)
		data = np.reshape(data, gridShape + tensorShape)

		return TensorFieldElement(dim, data)

	def toFunction(self) -> Callable[[np.ndarray, Union[None, PreCompute]], float]:

		gridShape = self.data.shape[self.spacialDimensions:]
		tensorShape = self.data.shape[:self.spacialDimensions]

		totalSamplePoints = np.prod(gridShape)
		indices = np.indices(gridShape)
		indices = np.transpose(np.reshape(
			indices, (self.spacialDimensions, totalSamplePoints)))

		chebEngines = []
		for axis in gridShape:
			chebEngines.append(ChebEngine(axis))

		# change to spectral

		specData = scipy.fft.dct(
			self.data, type=1, axis=0) / chebEngines[0].N
		for i in range(1, len(chebEngines)):
			specData = scipy.fft.dct(
				specData, type=1, axis=i) / chebEngines[i].N

		for indexList in indices:
			for i in range(self.spacialDimensions):
				if indexList[i] == 0 or indexList[i] == chebEngines[i].N:
					specData[tuple(indexList)] /= 2

		def outputFunct(position: np.ndarray, preComputeArray: Union[None, PreCompute] = None) -> float:
			if preComputeArray == None:
				output = np.zeros(tensorShape)
				for indexList in indices:
					modeContribution = np.array(specData[tuple(indexList)])
					for i in range(self.spacialDimensions):
						modeContribution *= chebEngines[i].eval(
							indexList[i], position[i])
					output += modeContribution
				return output
			else:
				output = np.zeros(tensorShape)
				for indexList in indices:
					output += specData[tuple(indexList)] * \
						preComputeArray(indexList, position)
				return output

		return outputFunct

	def __mul__(self, other: Union['TensorFieldElement', int, float]):
		if type(other) == TensorFieldElement:
			assert(self.rank == 0 or other.rank == 0), 'multiplication must be with scalar. Try tensorProduct'
			return TensorFieldElement(
				self.spacialDimensions,
				#to make the broadcast work correctly, we reverse all axes.
				#this makes the broadcast match the first axes going forward
				#instead of the last axes going backwards as is default.
				np.transpose(np.transpose(self.data) * np.transpose(other.data)))
		else:
			return TensorFieldElement(
				self.spacialDimensions,
				self.data * other)
	__rmul__ = __mul__

	def __truediv__(self, other: Union['TensorFieldElement', int, float]):
		if type(other) == TensorFieldElement:
			assert(other.rank == 0), 'division must be with scalar. This is nonsense'
			return TensorFieldElement(
				self.spacialDimensions,
				#to make the broadcast work correctly, we reverse all axes.
				#this makes the broadcast match the first axes going forward
				#instead of the last axes going backwards as is default.
				np.transpose(np.transpose(self.data) / np.transpose(other.data)))
		if (other == 0):
			raise(ZeroDivisionError('tensor divided by zero scalar'))
		return TensorFieldElement(
			self.spacialDimensions,
			self.data / other)

	def __add__(self, other: 'TensorFieldElement'):
		return TensorFieldElement(
			self.spacialDimensions,
			self.data + other.data)

	def __sub__(self, other: 'TensorFieldElement'):
		return TensorFieldElement(
			self.spacialDimensions,
			self.data - other.data)

	def __neg__(self):
		return TensorFieldElement(
			self.spacialDimensions,
			-self.data)

	def _partialDerivative(self, axis: int) -> np.ndarray:
		N = self.data.shape[axis] - 1

		# for generating advanced slicing sets
		def generateSurfaceSlice(index: int):
			output: list[Union[slice, int]] = list(self._fullSlice)
			output[axis] = index
			return tuple(output)

		# change to spectral along given axis
		specData: np.ndarray = scipy.fft.dct(self.data, type=1, axis=axis) / N
		specData[generateSurfaceSlice(0)] /= 2
		specData[generateSurfaceSlice(N)] /= 2

		# perform derivative in spectral basis
		specDerivativeData: np.ndarray = np.zeros(specData.shape)
		specDerivativeData[generateSurfaceSlice(N - 1)] =\
			2 * N * specData[generateSurfaceSlice(N)]
		specDerivativeData[generateSurfaceSlice(N - 2)] =\
			2 * (N - 1) * specData[generateSurfaceSlice(N - 1)]
		for i in reversed(range(0, N - 2)):
			specDerivativeData[generateSurfaceSlice(i)] =\
				specDerivativeData[generateSurfaceSlice(i + 2)]\
				+ 2 * (i + 1) * specData[generateSurfaceSlice(i + 1)]

		# change back to modal representation
		# (above algorithm computes 0th element off by factor of two,
		# so don't need to adjust here due to cancellation)
		specDerivativeData[generateSurfaceSlice(N)] *= 2
		return scipy.fft.dct(specDerivativeData, type=1, axis=axis) / 2

	def coordinateGradient(self):
		# for generating advanced slicing sets
		# fullSlice = tuple(
		# 	[slice(0, size) for size in self.data.shape]
		# ) + (slice(0, self.dim),)

		fullSlice = tuple(
			list(self._fullSlice) +
			[slice(0, self.spacialDimensions)])

		def generateGradientSlice(axis: int):
			output: list[Union[slice, int]] = list(fullSlice)
			output[-1] = axis
			return tuple(output)

		outputData: np.ndarray = np.zeros(self.data.shape + (self.spacialDimensions,))
		for axis in range(self.spacialDimensions):
			outputData[generateGradientSlice(
				axis)] = self._partialDerivative(axis)
		return TensorFieldElement(
			self.spacialDimensions,
			outputData)

	def getBoundaryElement(self, axis: int, rightBoundary: bool):

		slice = list(self._fullSlice)
		slice[axis] = 0 if rightBoundary else self.data.shape[axis] - 1
		dataView = self.data[tuple(slice)]
		return TensorFieldElement(self.spacialDimensions - 1, dataView)

	@staticmethod
	def _createPermutationList(settingRule, requiredLength, initialSlice):
		if settingRule == '':
			return list(range(requiredLength)), initialSlice
		outputLabelling, inputLabelling = tuple(settingRule.split('<-'))
		assert(len(inputLabelling) == requiredLength), 'tensor indexed incorrectly'
		newOutputLabelling = ''
		for i, label in enumerate(outputLabelling):
			if label in inputLabelling:
				assert(label not in newOutputLabelling), "tensor double indexing"
				newOutputLabelling += label
			elif label == '0': # left slice
				initialSlice[i] = -1
			else: # right slice
				assert(label == '1'), "unrecognized token or indices"
				initialSlice[i] = 0

		dataPermutation = list(range(requiredLength))

		for i, label in enumerate(newOutputLabelling):
			for j, label2 in enumerate(inputLabelling):
				if label == label2:
					dataPermutation[i] = j
		
		return dataPermutation, initialSlice

	def addData(self,
		other: 'TensorFieldElement',
		tensorSettingRule = '',
		gridSettingRule = ''):
		"""
		settingRule: empty for no change to dimensions,
			otherwise other's dimensions on left of arrow (->)
			and self's dimensions on right.
			All of other's dimensions must be used, but
			slices can be set on self via specifying dimensions.

			e.g. "" (identical data shapes)
			"abc<-cab" (permute dimensions)
			"i0jk<-kij" (set a left boundary on self)
			"i1jk<-kij" (set a right boundary on self)
		"""

		# assert(False), 'Tensor Field Setter Not Implemented'

		if (tensorSettingRule == '' and gridSettingRule == ''):
			self.data[:] = other.data

		tensorPerm, tensorSlice = TensorFieldElement._createPermutationList(
			tensorSettingRule,
			other.rank,
			self._fullSlice[self.spacialDimensions:])
		gridPerm, gridSlice = TensorFieldElement._createPermutationList(
			gridSettingRule,
			other.spacialDimensions,
			self._fullSlice[:self.spacialDimensions])

		for i in range(len(tensorPerm)):
			tensorPerm[i] += len(gridPerm)

		dataPermutation = tuple(gridPerm + tensorPerm)
		outputSlice = tuple(gridSlice + tensorSlice)

		dataView = self.data[outputSlice]
		assert(dataView.base is not None), 'view not made'

		dataView[:] += np.transpose(other.data, tuple(dataPermutation))

	def setData(self, other: 'TensorFieldElement', settingRule = ''):
		self.data = np.zeros(self.data.shape)
		self.addData(other, settingRule)

	def sqrt_scalar(self):
		assert(self.rank == 0), 'sqrt only allowed for scalar'
		return TensorFieldElement(self.spacialDimensions, np.sqrt(self.data))
