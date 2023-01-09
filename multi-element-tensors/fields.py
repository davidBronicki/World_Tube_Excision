import elements, frames, grids
import numpy as np

from typing import List, Tuple, Union, Callable

class TensorField:
	def __init__(self,
		grid: grids.Grid,
		elements: List[elements.TensorFieldElement]):

		assert(len(elements) == len(grid.gridElements)), 'each grid element requires exactly one tensor element'
		for element, gridElement in zip(elements, grid.gridElements):
			assert(tuple(element.data.shape[:element.spacialDimensions]) == gridElement.shape
				), 'each tensor element must agree in shape with grid element'
			assert(element.data.shape[element.spacialDimensions:] ==
				elements[0].data.shape[elements[0].spacialDimensions:]
				), 'each tensor element must agree in tensor shape'
		
		self.rank = elements[0].rank
		self.spacialDimension = elements[0].spacialDimensions
		self.tensorShape = tuple(elements[0].data.shape[self.spacialDimension:])
		
		self.grid = grid
		
		self.elements = elements

	def copy(self):
		newElements: List[elements.TensorFieldElement] = []
		for element in self.elements:
			newElements.append(element.copy())
		return TensorField(self.grid, newElements)

	@staticmethod
	def tensorProduct(contractionString: str,
		*tensorFields: 'TensorField'):

		for field in tensorFields[1:]:
			assert(tensorFields[0].grid is field.grid), 'must have the same grid for a tensor product'

		outputElements: List[elements.TensorFieldElement] = []

		for tensorElementTuple in zip(*[field.elements for field in tensorFields]):
			outputElements.append(elements.TensorFieldElement.tensorProduct(
				contractionString, *tensorElementTuple))

		return TensorField(tensorFields[0].grid, outputElements)


	@staticmethod
	def defaultInit(grid: grids.Grid,
		tensorShape: Union[List[int], Tuple[int]]):

		outputElements: List[elements.TensorFieldElement] = []
		for gridElement in grid.gridElements:
			outputElements.append(elements.TensorFieldElement.defaultInit(
				list(gridElement.shape), list(tensorShape)))
		return TensorField(grid, outputElements)

	@staticmethod
	def functionInit(
		grid: grids.Grid,
		fieldInitializer: Callable[[frames.CoordinatePos], frames.FramedTensor]):

		outputElements: List[elements.TensorFieldElement] = []
		for gridElement, elementToGlobal, globalToElement in zip(
			grid.gridElements, grid.elementToGlobalTransforms, grid.globalToElementTransforms):

			coordsToPos = lambda x: frames.CoordinatePos(elementToGlobal.initialFrame, x)
			elementToGlobalPos = lambda x: elementToGlobal.map(x)
			globalToElementFrame = lambda x: globalToElement.transformTensor(x)
			deframe = lambda x: x.data
			
			elementInitFunct = lambda x: deframe(globalToElementFrame(
				fieldInitializer(elementToGlobalPos(coordsToPos(x)))))
			outputElements.append(elements.TensorFieldElement.functionInit(
				gridElement.chebOrders, elementInitFunct))
		return TensorField(grid, outputElements)

	def precomputeEval(self, precompute: grids.FieldPrecompute):
		spectralDataTable = [element.spectralData() for element in self.elements]
		return [precompute(i, spectralDataTable[precompute.indexElementID(i)])
			for i in range(len(precompute))]

	def toFunction(self):
		elementFunctions = [element.toFunction() for element in self.elements]
		def outputFunct(x: frames.CoordinatePos):
			elementID = self.grid.locateDomain(x)
			elementPos = self.grid.globalToElementTransforms[elementID].map(x)
			return elementFunctions[elementID](elementPos.coords)
		return outputFunct

	def __mul__(self, other: Union['TensorField', int, float]):
		outputElements: List[elements.TensorFieldElement] = []
		if type(other) == TensorField:
			for element1, element2 in zip(self.elements, other.elements):
				outputElements.append(element1 * element2)
		else:
			for element in self.elements:
				outputElements.append(element * other)
		return TensorField(self.grid, outputElements)

	__rmul__ = __mul__

	def __truediv__(self, other: Union['TensorField', int, float]):
		outputElements: List[elements.TensorFieldElement] = []
		if type(other) == TensorField:
			for element1, element2 in zip(self.elements, other.elements):
				outputElements.append(element1 / element2)
		else:
			for element in self.elements:
				outputElements.append(element / other)
		return TensorField(self.grid, outputElements)

	def __add__(self, other: 'TensorField'):
		outputElements: List[elements.TensorFieldElement] = []
		for element1, element2 in zip(self.elements, other.elements):
			outputElements.append(element1 + element2)
		return TensorField(self.grid, outputElements)

	def __sub__(self, other: 'TensorField'):
		outputElements: List[elements.TensorFieldElement] = []
		for element1, element2 in zip(self.elements, other.elements):
			outputElements.append(element1 - element2)
		return TensorField(self.grid, outputElements)

	def __neg__(self):
		outputElements: List[elements.TensorFieldElement] = []
		for element in self.elements:
			outputElements.append(-element)
		return TensorField(self.grid, outputElements)

	def addData(self, settingRule: str, other: 'TensorField'):
		for element1, element2 in zip(self.elements, other.elements):
			element1.addData(element2, tensorSettingRule=settingRule)

	def setData(self, settingRule: str, other: 'TensorField'):
		for element1, element2 in zip(self.elements, other.elements):
			element1.setData(element2, tensorSettingRule=settingRule)

	def sqrt_scalar(self):
		outputElements: List[elements.TensorFieldElement] = []
		for element in self.elements:
			outputElements.append(element.sqrt_scalar())
		return TensorField(self.grid, outputElements)

	def commaGrad(self):
		outputElements: List[elements.TensorFieldElement] = []
		for element in self.elements:
			outputElements.append(element.coordinateGradient())
		return TensorField(self.grid, outputElements)

	def _canonicalPackSize(self):
		return sum([np.prod(element.data.shape) for element in self.elements])

	def canonicalPack(self):
		totalPoints = self._canonicalPackSize()
		output = np.zeros((totalPoints,))
		currentIndex = 0
		for element in self.elements:
			nextIndex = currentIndex + np.prod(element.data.shape)
			output[currentIndex : nextIndex] = np.ravel(element.data)
			currentIndex = nextIndex
		return output

	def canonicalUnpack(self, newData: np.ndarray):
		assert(len(newData.shape) == 1), 'data cannot be unpacked, must be flat'
		assert(self._canonicalPackSize() == len(newData)), 'data cannot be unpacked, wrong length'
		currentIndex = 0
		for element in self.elements:
			nextIndex = currentIndex + np.prod(element.data.shape)
			element.data[:] = np.reshape(newData[currentIndex : nextIndex], element.data.shape)
			currentIndex = nextIndex

def trueGradient(tensorField: TensorField, christoffel: TensorField, upIndices: int):
	availableChars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
	assert (tensorField.rank <= len(availableChars) - 2
		), 'true gradient with rank>'+str(len(availableChars) - 2)+' is not supported'
	assert (christoffel.rank == 3), "that doesn't look like christoffel symbols..."

	output = tensorField.commaGrad()

	tensorIndexingList = availableChars[:tensorField.rank]
	christoffelUniqueIndexers = availableChars[tensorField.rank: tensorField.rank + 2]

	outputIndexingList = tensorIndexingList + [christoffelUniqueIndexers[1]]
	outputString = '->' + ''.join(outputIndexingList)

	for i in range(upIndices):
		tensorInputIndexingList = tensorIndexingList.copy()
		tensorInputIndexingList[i] = christoffelUniqueIndexers[0]
		christoffelIndexingList = [tensorIndexingList[i]] + christoffelUniqueIndexers
		output += TensorField.tensorProduct(
			''.join(tensorInputIndexingList) + ',' +\
				''.join(christoffelIndexingList) + outputString,
			tensorField, christoffel)

	for i in range(tensorField.rank - upIndices):
		tensorInputIndexingList = tensorIndexingList.copy()
		tensorInputIndexingList[i + upIndices] = christoffelUniqueIndexers[0]
		christoffelIndexingList = christoffelUniqueIndexers + [tensorIndexingList[i + upIndices]]
		output -= TensorField.tensorProduct(
			''.join(tensorInputIndexingList) + ',' +\
				''.join(christoffelIndexingList) + outputString,
			tensorField, christoffel)

	# for i in range(upIndices):
	# 	outputIndexingList = tensorIndexingList.copy()
	# 	outputIndexingList[i] = christoffelUniqueIndexers[0]
	# 	outputIndexingList.append(christoffelUniqueIndexers[1])
	# 	productString = ''.join(tensorIndexingList) + ','
	# 	productString += ''.join(christoffelUniqueIndexers)
	# 	productString += tensorIndexingList[i] + '->'
	# 	productString += ''.join(outputIndexingList)
	# 	output += TensorField.tensorProduct(
	# 		productString, tensorField, christoffel)

	# for i in range(tensorField.rank - upIndices):
	# 	outputIndexingList = tensorIndexingList.copy()
	# 	outputIndexingList[upIndices + i] = christoffelUniqueIndexers[1]
	# 	outputIndexingList.append(christoffelUniqueIndexers[0])
	# 	productString = ''.join(tensorIndexingList) + ','
	# 	productString += tensorIndexingList[upIndices + i]
	# 	productString += ''.join(christoffelUniqueIndexers) + '->'
	# 	productString += ''.join(outputIndexingList)
	# 	output -= TensorField.tensorProduct(
	# 		productString, tensorField, christoffel)

	return output

def tensorTransform(tensorField: TensorField, jac: TensorField, invJac: TensorField, upIndices: int):
	availableChars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
	assert (tensorField.rank <= len(availableChars) // 2
		), 'tensor transform with rank>'+str(len(availableChars) // 2)+' is not supported'
	assert (jac.rank == 2), "that doesn't look like a jacobian..."
	assert (invJac.rank == 2), "that doesn't look like an inverse jacobian..."

	inputIndexingString = availableChars[:tensorField.rank]
	outputIndexingString = availableChars[tensorField.rank : 2 * tensorField.rank]

	contractionString = [inputIndexingString]
	for i in range(upIndices):
		contractionString.append(outputIndexingString[i] + inputIndexingString[i])
	for i in range(upIndices, tensorField.rank):
		contractionString.append(inputIndexingString[i] + outputIndexingString[i])
	contractionString = ','.join(contractionString)
	contractionString += '->' + outputIndexingString

	downIndices = tensorField.rank - upIndices
	return TensorField.tensorProduct(contractionString,
		*tuple([tensorField] + [jac] * upIndices + [invJac] * (downIndices)))
