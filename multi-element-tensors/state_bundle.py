import fields
import tensor_elements as elements
import numpy as np
from typing import List

class InterfaceBundle:
	def __init__(self,
		gridInterface: fields.GridInterface,
		leftTensorElements: elements.TensorFieldElement,
		rightTensorElements: elements.TensorFieldElement):

		self.gridInterface = gridInterface
		self.leftTensorElements = leftTensorElements
		self.rightTensorElements = rightTensorElements

class BoundaryBundle:
	def __init__(self,
		gridBoundary: fields.GridBoundary,
		tensorElements: List[elements.TensorFieldElement]):

		self.gridBoundary = gridBoundary
		self.tensorElements = tensorElements

class StateBundle:
	def __init__(self,
		tensorFields: List[fields.TensorField],
		staticFieldIndices: List[int] = []):

		self.grid = tensorFields[0].grid
		for field in tensorFields:
			assert (field.grid is self.grid), 'bundle must use one grid'
		self.tensorFields = tensorFields
		self.staticFieldIndices = staticFieldIndices

		self.boundaries: List[BoundaryBundle] = []
		self.interfaces: List[InterfaceBundle] = []

		def getBoundaryElements(gridBoundary: fields.GridBoundary):
			elementIndex = gridBoundary.baseElement.index
			axis = gridBoundary.axis
			isRightBoundary = gridBoundary.isRightBoundary
			return [field.elements[elementIndex].getBoundaryElement(
				axis, isRightBoundary) for field in tensorFields]

		for gridBoundary in self.grid.boundaries:
			self.boundaries.append(BoundaryBundle(
				gridBoundary,
				getBoundaryElements(gridBoundary)))
		for gridInterface in self.grid.interfaces:
			leftGridBoundary = gridInterface.leftBoundary
			rightGridBoundary = gridInterface.rightBoundary
			leftBoundaryElements = getBoundaryElements(leftGridBoundary)
			rightBoundaryElements = getBoundaryElements(rightGridBoundary)

			self.interfaces.append(InterfaceBundle(
				gridInterface, leftBoundaryElements, rightBoundaryElements))

	def canonicalPack(self, overrideStaticFieldIndices = None):
		staticFieldIndices: List[int] = self.staticFieldIndices if\
			overrideStaticFieldIndices is None else overrideStaticFieldIndices
		packedDynamicFields: List[np.ndarray] = []
		totalPoints = 0
		for i in range(len(self.tensorFields)):
			if i not in staticFieldIndices:
				packedDynamicFields.append(self.tensorFields[i].canonicalPack())
				totalPoints += len(packedDynamicFields[-1])
		currentIndex = 0
		output = np.zeros((totalPoints,))
		for packedField in packedDynamicFields:
			nextIndex = currentIndex + len(packedField)
			output[currentIndex : nextIndex] = packedField[:]
			currentIndex = nextIndex
		return output

	def canonicalUnpack(self, newData: np.ndarray, overrideStaticFieldIndices = None):
		assert(len(newData.shape) == 1), 'data cannot be unpacked, must be flat'
		staticFieldIndices: List[int] = self.staticFieldIndices if\
			overrideStaticFieldIndices is None else overrideStaticFieldIndices
		dynamicFieldIndices: List[int] = []
		fieldPackSizes: List[int] = []
		for i in range(len(self.tensorFields)):
			if i not in staticFieldIndices:
				dynamicFieldIndices.append(i)
				fieldPackSizes.append(self.tensorFields[i]._canonicalPackSize())
		totalPoints = sum(fieldPackSizes)
		assert(totalPoints == len(newData)), 'data cannot be unpacked, wrong length'
		currentIndex = 0
		for i, packSize in zip(dynamicFieldIndices, fieldPackSizes):
			nextIndex = currentIndex + packSize
			self.tensorFields[i].canonicalUnpack(newData[currentIndex : nextIndex])
			currentIndex = nextIndex
