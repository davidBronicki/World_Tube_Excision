import tensor_elements, frames
import numpy as np

from typing import List, Tuple, Union, Callable

class GridElement:
	def __init__(self, chebOrders: Tuple[int], frame: frames.CoordinateFrame):
		assert(frame.dim == len(chebOrders)), 'grid shape must agree with frame dim'
		self.chebOrders = chebOrders
		self.shape = tuple([order + 1 for order in chebOrders])
		self.frame = frame
		self.spacialDimensions = len(chebOrders)
		self.boundaries: List[GridBoundary] = []
		self.index = -1 # set in grid instantiation
		for axis in range(self.spacialDimensions):
			self.boundaries.append(GridBoundary(self, axis, True))
			self.boundaries.append(GridBoundary(self, axis, False))

	def getBoundary(self, axis: int, rightSide: bool):
		return self.boundaries[2 * axis + (0 if rightSide else 1)]

class GridBoundary:
	def __init__(self, baseElement: GridElement, axis: int, isRightBoundary: bool):
		self.baseElement = baseElement
		self.shape = tuple(baseElement.shape[:axis] + baseElement.shape[axis + 1:])
		self.axis = axis
		self.isRightBoundary = isRightBoundary
		self.index = -1 # set in grid instantiation
		self.interface: Union[GridInterface, None] = None

class GridInterface:
	def __init__(self,
		leftGridBoundary: GridBoundary,
		rightGridBoundary: GridBoundary,
		leftToRightSetterRule = ''):
		
		leftGridBoundary.interface = self
		self.leftGridBoundary = leftGridBoundary

		rightGridBoundary.interface = self
		self.rightGridBoundary = rightGridBoundary

		self.leftToRightSetterRule = leftToRightSetterRule
		if leftToRightSetterRule == '':
			self.rightToLeftSetterRule = ''
		else:
			right, left = leftToRightSetterRule.split('<-')
			self.rightToLeftSetterRule = '<-'.join((left, right))

		self.index = -1 # set in grid instantiation
		self.leftToRightTransform: Union[frames.Transformation, None] = None
		self.leftToRightJacobianField: Union[tensor_elements.TensorFieldElement, None] = None
		self.leftToRightInvJacobianField: Union[tensor_elements.TensorFieldElement, None] = None
		self.rightToLeftTransform: Union[frames.Transformation, None] = None
		self.rightToLeftJacobianField: Union[tensor_elements.TensorFieldElement, None] = None
		self.rightToLeftInvJacobianField: Union[tensor_elements.TensorFieldElement, None] = None

class Grid:
	def __init__(self,
		globalToElementTransforms: List[frames.Transformation],
		gridElements: List[GridElement]):

		assert(len(gridElements) == len(globalToElementTransforms)), 'each element must have a coordinate frame'
		self.globalToElementTransforms = globalToElementTransforms
		self.globalFrame = globalToElementTransforms[0].initialFrame
		self.elementFrames: List[frames.CoordinateFrame] = []
		self.elementToGlobalTransforms: List[frames.Transformation] = []
		for t in globalToElementTransforms:
			assert(t.initialFrame is self.globalFrame), 'only one global frame allowed'
			self.elementFrames.append(t.finalFrame)
			self.elementToGlobalTransforms.append(t.inverse())

		self.gridElements = gridElements
		for i in range(len(gridElements)):
			self.gridElements[i].index = i
		self.boundaries: List[GridBoundary] = []
		self.interfaces: List[GridInterface] = []

		for element in gridElements:
			for boundary in element.boundaries:
				if boundary.interface is None:
					self.boundaries.append(boundary)
				else:
					self._gridLevelInterfaceInit(boundary.interface)

	def _gridLevelInterfaceInit(self, interface: GridInterface):
		if interface.rightToLeftTransform is not None: return

		self.interfaces.append(interface)
		
		leftBoundary = interface.leftGridBoundary
		rightboundary = interface.rightGridBoundary
		leftFrame = leftBoundary.baseElement.frame
		rightFrame = rightboundary.baseElement.frame
		leftIndex = self.elementFrames.index(leftFrame)
		rightIndex = self.elementFrames.index(rightFrame)

		interface.leftToRightTransform =\
			self.globalToElementTransforms[rightIndex].compose(
				self.elementToGlobalTransforms[leftIndex])
		interface.leftToRightJacobianField =\
			tensor_elements.TensorFieldElement.functionInit(leftBoundary.shape,
				lambda x: interface.leftToRightTransform.jacobian(
					frames.CoordinatePos(leftFrame, x)).data)
		interface.leftToRightInvJacobianField =\
			tensor_elements.TensorFieldElement.functionInit(leftBoundary.shape,
				lambda x: interface.leftToRightTransform.inverseJacobian(
					frames.CoordinatePos(leftFrame, x)).data)
		interface.rightToLeftTransform =\
			self.globalToElementTransforms[leftIndex].compose(
				self.elementToGlobalTransforms[rightIndex])
		interface.rightToLeftJacobianField =\
			tensor_elements.TensorFieldElement.functionInit(leftBoundary.shape,
				lambda x: interface.rightToLeftTransform.jacobian(
					frames.CoordinatePos(rightFrame, x)).data)
		interface.rightToLeftInvJacobianField =\
			tensor_elements.TensorFieldElement.functionInit(leftBoundary.shape,
				lambda x: interface.rightToLeftTransform.inverseJacobian(
					frames.CoordinatePos(rightFrame, x)).data)

class TensorField:
	def __init__(self,
		grid: Grid,
		elements: List[tensor_elements.TensorFieldElement]):

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
		newElements: List[tensor_elements.TensorFieldElement] = []
		for element in self.elements:
			newElements.append(element.copy())
		return TensorField(self.grid, newElements)

	@staticmethod
	def tensorProduct(contractionString: str,
		*tensorFields: 'TensorField'):

		for field in tensorFields[1:]:
			assert(tensorFields[0].grid is field.grid), 'must have the same grid for a tensor product'

		outputElements: List[tensor_elements.TensorFieldElement] = []

		for tensorElementTuple in zip(*[field.elements for field in tensorFields]):
			outputElements.append(tensor_elements.TensorFieldElement.tensorProduct(
				contractionString, *tensorElementTuple))

		return TensorField(tensorFields[0].grid, outputElements)


	@staticmethod
	def defaultInit(grid: Grid,
		tensorShape: List[int]):

		outputElements: List[tensor_elements.TensorFieldElement] = []
		for gridElement in grid.gridElements:
			outputElements.append(tensor_elements.TensorFieldElement.defaultInit(
				gridElement.shape, tensorShape))
		return TensorField(grid, outputElements)

	@staticmethod
	def functionInit(
		grid: Grid,
		fieldInitializer: Callable[[frames.CoordinatePos], frames.FramedTensor]):

		outputElements: List[tensor_elements.TensorFieldElement] = []
		for gridElement, transform in zip(grid.gridElements, grid.globalToElementTransforms):
			coordsToPos = lambda x: frames.CoordinatePos(transform.initialFrame, x)
			elementToGlobalPos = lambda x: transform.map(x)
			globalToElementFrame = lambda x: transform.transformTensor(x)
			deframe = lambda x: x.data
			
			elementInitFunct = lambda x: deframe(globalToElementFrame(
				fieldInitializer(elementToGlobalPos(coordsToPos(x)))))
			outputElements.append(tensor_elements.TensorFieldElement.functionInit(
				gridElement.chebOrders, elementInitFunct))
		return TensorField(grid, outputElements)

	def __mul__(self, other: Union['TensorField', int, float]):
		outputElements: List[tensor_elements.TensorFieldElement] = []
		if type(other) == TensorField:
			for element1, element2 in zip(self.elements, other.elements):
				outputElements.append(element1 * element2)
		else:
			for element in self.elements:
				outputElements.append(element * other)
		return TensorField(self.grid, outputElements)

	__rmul__ = __mul__

	def __truediv__(self, other: Union['TensorField', int, float]):
		outputElements: List[tensor_elements.TensorFieldElement] = []
		if type(other) == TensorField:
			for element1, element2 in zip(self.elements, other.elements):
				outputElements.append(element1 / element2)
		else:
			for element in self.elements:
				outputElements.append(element / other)
		return TensorField(self.grid, outputElements)

	def __add__(self, other: 'TensorField'):
		outputElements: List[tensor_elements.TensorFieldElement] = []
		for element1, element2 in zip(self.elements, other.elements):
			outputElements.append(element1 + element2)
		return TensorField(self.grid, outputElements)

	def __sub__(self, other: 'TensorField'):
		outputElements: List[tensor_elements.TensorFieldElement] = []
		for element1, element2 in zip(self.elements, other.elements):
			outputElements.append(element1 - element2)
		return TensorField(self.grid, outputElements)

	def __neg__(self):
		outputElements: List[tensor_elements.TensorFieldElement] = []
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
		outputElements: List[tensor_elements.TensorFieldElement] = []
		for element in self.elements:
			outputElements.append(element.sqrt_scalar())
		return TensorField(self.grid, outputElements)

	def commaGrad(self):
		outputElements: List[tensor_elements.TensorFieldElement] = []
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
			output[currentIndex : nextIndex] =\
				np.ravel(element.data)
			currentIndex = nextIndex
		return output

	def canonicalUnpack(self, newData: np.ndarray):
		assert(len(newData.shape) == 1), 'data cannot be unpacked, must be flat'
		totalPoints = self._canonicalPackSize()
		assert(totalPoints == len(newData)), 'data cannot be unpacked, wrong length'
		currentIndex = 0
		for element in self.elements:
			nextIndex = currentIndex + np.prod(element.data.shape)
			element.data[:] = np.reshape(newData[currentIndex : nextIndex], element.data.shape)

def trueGradient(tensorField: TensorField, christoffel: TensorField, upIndices: int):
	availableChars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
	assert (tensorField.rank <= len(availableChars) - 2
		), 'true gradient with rank>'+str(len(availableChars) - 2)+' is not supported'
	assert (christoffel.rank == 3), "that doesn't look like christoffel symbols..."

	output = tensorField.commaGrad()

	tensorIndexingList = availableChars[:tensorField.rank]
	christoffelUniqueIndexers = availableChars[tensorField.rank: tensorField.rank + 2]

	for i in range(upIndices):
		outputIndexingList = tensorIndexingList.copy()
		outputIndexingList[i] = christoffelUniqueIndexers[0]
		outputIndexingList.append(christoffelUniqueIndexers[1])
		productString = ''.join(tensorIndexingList) + ','
		productString += ''.join(christoffelUniqueIndexers)
		productString += tensorIndexingList[i] + '->'
		productString += ''.join(outputIndexingList)
		productString +=  christoffelUniqueIndexers[0]
		output += TensorField.tensorProduct(
			productString, tensorField, christoffel)

	for i in range(tensorField.rank - upIndices):
		outputIndexingList = tensorIndexingList.copy()
		outputIndexingList[upIndices + i] = christoffelUniqueIndexers[1]
		outputIndexingList.append(christoffelUniqueIndexers[0])
		productString = ''.join(tensorIndexingList) + ','
		productString += tensorIndexingList[upIndices + i]
		productString += ''.join(christoffelUniqueIndexers) + '->'
		productString += ''.join(outputIndexingList)
		output += TensorField.tensorProduct(
			productString, tensorField, christoffel)

	return output

def tensorTransform(tensorField: TensorField, jac: TensorField, invJac: TensorField, upIndices: int):
	availableChars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
	assert (tensorField.rank <= len(availableChars) // 2
		), 'tensor transform with rank>'+str(len(availableChars) // 2)+' is not supported'
	assert (jac.rank == 2), "that doesn't look like a jacobian..."
	assert (invJac.rank == 2), "that doesn't look like a(n inverse) jacobian..."

	inputIndexingString = availableChars[:tensorField.rank]
	outputIndexingString = availableChars[tensorField.rank : 2 * tensorField.rank]

	contractionString = [inputIndexingString]
	for i in range(tensorField.rank):
		contractionString.append(inputIndexingString[i] + outputIndexingString[i])
	contractionString = ','.join(contractionString)
	contractionString += '->' + outputIndexingString

	downIndices = tensorField.rank - upIndices
	return TensorField.tensorProduct(contractionString,
		*tuple([tensorField] + [jac] * upIndices + [invJac] * (downIndices)))
