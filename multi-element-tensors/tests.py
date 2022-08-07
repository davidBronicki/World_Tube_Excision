from tensor_elements import *
import numpy as np

verbose = True

def printTensorState(tensorField):
	print("TENSOR STATE:")
	print("dim:", tensorField.spacialDimensions)
	print("rank:", tensorField.rank)
	print("data:", tensorField.data, sep=' ' if tensorField.rank <= 1 else '\n')

def checkResult(testName, expected, produced = None):
	passed = expected if produced is None else (produced == expected)
	if type(passed) != bool:
		passed = passed.all()
	if passed:
		print(testName, ": PASSED", sep='')
	else:
		print(testName, ": FAILED", sep='')
		if verbose and produced is None:
			print('Truth value(s) given:')
			print(expected)
			print()
		elif verbose:
			print('Expected Result:')
			print(expected)
			print('Produced Result:')
			print(produced)
			print()
	return not passed

def plainTensorTest():
	failedTests = 0

	testData_Tensor = np.resize(np.array(range(9)), (3,3))
	testData_Vector = np.array(range(3))
	expectedData = testData_Tensor.dot(testData_Vector)

	basicTensor = TensorFieldElement(0, testData_Tensor)
	basicVector = TensorFieldElement(0, testData_Vector)
	testResult = TensorFieldElement.tensorProduct('ij,j->i', basicTensor, basicVector)

	failedTests += checkResult(
		'Single Matrix-Vector Mult', expectedData, testResult.data)
	failedTests += checkResult(
		'Single Matrix-Vector Mult -- dim', 0, testResult.spacialDimensions)
	failedTests += checkResult(
		'Single Matrix-Vector Mult -- rank', 1, testResult.rank)

	expectedData = np.reshape(np.outer(testData_Vector, testData_Tensor),(3,3,3))
	testResult = TensorFieldElement.tensorProduct('ij,k->kij', basicTensor, basicVector)

	failedTests += checkResult(
		'Single Matrix-Vector Outer Prod', expectedData, testResult.data)
	failedTests += checkResult(
		'Single Matrix-Vector Outer Prod -- dim', 0, testResult.spacialDimensions)
	failedTests += checkResult(
		'Single Matrix-Vector Outer Prod -- rank', 3, testResult.rank)

	testResult = TensorFieldElement.tensorProduct('ii->', basicTensor)

	failedTests += checkResult(
		'Single Trace', 12, testResult.data)
	failedTests += checkResult(
		'Single Trace -- dim', 0, testResult.spacialDimensions)
	failedTests += checkResult(
		'Single Trace -- rank', 0, testResult.rank)

	expectedData = testData_Tensor.dot(testData_Vector).dot(testData_Vector)
	testResult = TensorFieldElement.tensorProduct(
		'ij,i,j->', basicTensor, basicVector, basicVector)
	
	failedTests += checkResult(
		'Single Basis Component Aquisition', expectedData, testResult.data)
	failedTests += checkResult(
		'Single Basis Component Aquisition -- dim', 0, testResult.spacialDimensions)
	failedTests += checkResult(
		'Single Basis Component Aquisition -- rank', 0, testResult.rank)

	doubledData = testData_Tensor * 2
	zeroData = np.zeros((3,3))
	
	failedTests += checkResult(
		'Single Tensor Negative', -testData_Tensor, (-basicTensor).data)
	failedTests += checkResult(
		'Single Tensor Subtraction', zeroData, (basicTensor - basicTensor).data)
	failedTests += checkResult(
		'Single Tensor Addition', doubledData, (basicTensor + basicTensor).data)
	failedTests += checkResult(
		'Single Tensor Mult', doubledData, (basicTensor * 2).data)

	return failedTests

def smallFieldTest():
	failedTests = 0

	testData_TensorField = np.resize(np.array(range(16)), (2,2,2,2))
	testData_VectorField = np.resize(np.array(range(8)), (2,2,2))
	testData_ScalarField = np.resize(np.array(range(4)), (2,2))
	
	tensorField = TensorFieldElement(2, testData_TensorField)
	vectorField = TensorFieldElement(2, testData_VectorField)
	scalarField = TensorFieldElement(2, testData_ScalarField)

	expectedData = np.einsum('...ij,...j->...i', testData_TensorField, testData_VectorField)
	testResult = TensorFieldElement.tensorProduct('ij,j->i', tensorField, vectorField)

	failedTests += checkResult(
		'Field Tensor-Vector Mult', expectedData, testResult.data)
	failedTests += checkResult(
		'Field Tensor-Vector Mult -- dim', 2, testResult.spacialDimensions)
	failedTests += checkResult(
		'Field Tensor-Vector Mult -- rank', 1, testResult.rank)

	expectedData = np.einsum('...ij,...k->...kij', testData_TensorField, testData_VectorField)
	testResult = TensorFieldElement.tensorProduct('ij,k->kij', tensorField, vectorField)

	failedTests += checkResult(
		'Field Tensor-Vector Outer Prod', expectedData, testResult.data)
	failedTests += checkResult(
		'Field Tensor-Vector Outer Prod -- dim', 2, testResult.spacialDimensions)
	failedTests += checkResult(
		'Field Tensor-Vector Outer Prod -- rank', 3, testResult.rank)

	expectedData = np.einsum('...ii->...', testData_TensorField)
	testResult = TensorFieldElement.tensorProduct('ii->', tensorField)

	failedTests += checkResult(
		'Field Trace', expectedData, testResult.data)
	failedTests += checkResult(
		'Field Trace -- dim', 2, testResult.spacialDimensions)
	failedTests += checkResult(
		'Field Trace -- rank', 0, testResult.rank)

	expectedData = np.einsum('...ij,...i,...j->...',
		testData_TensorField, testData_VectorField, testData_VectorField)
	testResult = TensorFieldElement.tensorProduct(
		'ij,i,j->', tensorField, vectorField, vectorField)
	
	failedTests += checkResult(
		'Field Basis Component Aquisition', expectedData, testResult.data)
	failedTests += checkResult(
		'Field Basis Component Aquisition -- dim', 2, testResult.spacialDimensions)
	failedTests += checkResult(
		'Field Basis Component Aquisition -- rank', 0, testResult.rank)

	expectedData = np.einsum('...ij,...->...ij',
		testData_TensorField, testData_ScalarField)
	testResult = TensorFieldElement.tensorProduct(
		'ij,->ij', tensorField, scalarField)
	
	failedTests += checkResult(
		'Scalar Field Tensor Mult', expectedData, testResult.data)
	failedTests += checkResult(
		'Scalar Field Tensor Mult -- dim', 2, testResult.spacialDimensions)
	failedTests += checkResult(
		'Scalar Field Tensor Mult -- rank', 2, testResult.rank)

	testResult = tensorField * scalarField
	
	failedTests += checkResult(
		'Scalar Field Operator Mult', expectedData, testResult.data)
	failedTests += checkResult(
		'Scalar Field Operator Mult -- dim', 2, testResult.spacialDimensions)
	failedTests += checkResult(
		'Scalar Field Operator Mult -- rank', 2, testResult.rank)

	doubledData = testData_TensorField * 2
	zeroData = np.zeros((2,2,2,2))
	
	failedTests += checkResult(
		'Field Tensor Negative', -testData_TensorField, (-tensorField).data)
	failedTests += checkResult(
		'Field Tensor Subtraction', zeroData, (tensorField - tensorField).data)
	failedTests += checkResult(
		'Field Tensor Addition', doubledData, (tensorField + tensorField).data)
	failedTests += checkResult(
		'Field Tensor Mult', doubledData, (tensorField * 2).data)
	
	return failedTests

def derivativeTests():
	failedTests = 0

	def initialVectorField(coords):
		x = coords[0]
		y = coords[1]
		return np.array([
			np.sin(x) * np.cos(y),
			np.exp(x) * np.cos(y)])

	def gradX(coords):
		x = coords[0]
		y = coords[1]
		return np.array([
			np.cos(x) * np.cos(y),
			np.exp(x) * np.cos(y)])

	def gradY(coords):
		x = coords[0]
		y = coords[1]
		return np.array([
			-np.sin(x) * np.sin(y),
			-np.exp(x) * np.sin(y)])
	
	def grad(coords):
		x = coords[0]
		y = coords[1]
		return np.array([
			[ np.cos(x) * np.cos(y),-np.sin(x) * np.sin(y)],
			[ np.exp(x) * np.cos(y),-np.exp(x) * np.sin(y)]])

	gridShape = (10, 10)
	initTensor = TensorFieldElement.functionInit(gridShape, initialVectorField)

	expectedGradX = TensorFieldElement.functionInit(gridShape, gradX)
	expectedGradY = TensorFieldElement.functionInit(gridShape, gradY)
	expectedGradTotal = TensorFieldElement.functionInit(gridShape, grad)

	calculatedGradX = initTensor._partialDerivative(0)
	calculatedGradY = initTensor._partialDerivative(1)
	calculatedCoordinateGrad = initTensor.coordinateGradient()

	deltaGradX = calculatedGradX - expectedGradX.data
	deltaGradY = calculatedGradY - expectedGradY.data
	deltaGradTotal = calculatedCoordinateGrad.data - expectedGradTotal.data

	gradX_passed = np.abs(deltaGradX) < 1e-8
	gradY_passed = np.abs(deltaGradY) < 1e-8
	gradTotal_passed = np.abs(deltaGradTotal) < 1e-8

	failedTests += checkResult(
		'x gradient', gradX_passed)
	failedTests += checkResult(
		'y gradient', gradY_passed)
	failedTests += checkResult(
		'full coordinate gradient', gradTotal_passed)
	
	return failedTests

def boundaryTests():
	failedTests = 0

	testData_TensorField = np.resize(np.array(range(16)), (2,2,2,2))
	tensorField = TensorFieldElement(2, testData_TensorField.copy())

	leftData = testData_TensorField[:,1,:,:]
	rightData = testData_TensorField[:,0,:,:]

	leftBoundaryField = tensorField.getBoundaryElement(1, False)
	rightBoundaryField = tensorField.getBoundaryElement(1, True)

	failedTests += checkResult(
		'Left Boundary Data', leftData, leftBoundaryField.data)
	failedTests += checkResult(
		'Right Boundary Data', rightData, rightBoundaryField.data)
	failedTests += checkResult(
		'Original Field is not a View', not tensorField.isView)
	failedTests += checkResult(
		'Boundary is a View', rightBoundaryField.isView)

	newDataRightSet = testData_TensorField.copy()
	newDataRightSet[0,:,:,:] = leftData
	newDataLeftSet = testData_TensorField.copy()
	newDataLeftSet[1,:,:,:] = leftData

	newFieldRight = tensorField.copy()
	newFieldRight.setData('1i<-i', leftBoundaryField)

	newFieldLeft = tensorField.copy()
	newFieldLeft.setData('0i<-i', leftBoundaryField)

	failedTests += checkResult(
		'Setting Right Boundary', newDataRightSet, newFieldRight.data)
	failedTests += checkResult(
		'Setting Left Boundary', newDataLeftSet, newFieldLeft.data)

	return failedTests

if __name__ == "__main__":
	failedTests = 0
	print('Single Tensor Tests:\n')
	failedTests += plainTensorTest()
	print('\n\nSmall Tensor Field Tests:\n')
	failedTests += smallFieldTest()
	print('\n\nCoordinate Gradient Tests:\n')
	failedTests += derivativeTests()
	print('\n\nBoundary Data Tests:\n')
	failedTests += boundaryTests()

	print('\n\n')

	if (failedTests == 0):
		print('All tests passed.')
	elif (failedTests == 1):
		print(failedTests, 'failed test.')
	else:
		print(failedTests, 'failed tests.')
