package main

import (
	"fmt"
	"math"
)

const (
	inputNodes   = 2
	hiddenNodes  = 2
	outputNodes  = 1
	learningRate = 0.5
)

func main() {
	weightsInputHidden := createMatrix(hiddenNodes, inputNodes)
	weightsHiddenOutput := createMatrix(outputNodes, hiddenNodes)
	biasesHidden := createVector(hiddenNodes)
	biasesOutput := createVector(outputNodes)

	fmt.Println("Neural Net from Scratch - Now with Training!")
	initializeWeightsAndBiases(weightsInputHidden, weightsHiddenOutput, biasesHidden, biasesOutput)

	// XOR training data
	trainingInputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	trainingTargets := []float64{0, 1, 1, 0}

	// Train the network
	fmt.Println("Training the network...")
	for epoch := 0; epoch < 10000; epoch++ {
		totalLoss := 0.0
		
		for i := range trainingInputs {
			// Forward pass
			hiddenLayer, output := forwardPassWithHidden(trainingInputs[i], weightsInputHidden, biasesHidden, weightsHiddenOutput, biasesOutput)
			
			// Calculate loss
			loss := meanSquaredError(output, trainingTargets[i])
			totalLoss += loss
			
			// Backward pass (backpropagation)
			backpropagate(trainingInputs[i], hiddenLayer, output, trainingTargets[i], weightsInputHidden, weightsHiddenOutput, biasesHidden, biasesOutput)
		}
		
		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d, Average Loss: %.6f\n", epoch, totalLoss/float64(len(trainingInputs)))
		}
	}

	// Test the trained network
	fmt.Println("\nTesting trained network:")
	for i, input := range trainingInputs {
		_, output := forwardPassWithHidden(input, weightsInputHidden, biasesHidden, weightsHiddenOutput, biasesOutput)
		fmt.Printf("Input: %v, Target: %.0f, Output: %.4f\n", input, trainingTargets[i], output)
	}
}

func forwardPass(
	inputs []float64,
	weightsInputHidden [][]float64,
	biasesHidden []float64,
	weightsHiddenOutput [][]float64,
	biasesOutput []float64,
) float64 {
	hiddenLayer := createVector(hiddenNodes)

	for i := range hiddenLayer {
		sum := dotProduct(weightsInputHidden[i], inputs) + biasesHidden[i]
		hiddenLayer[i] = sigmoid(sum)
	}
	outputSum := dotProduct(weightsHiddenOutput[0], hiddenLayer) + biasesOutput[0]
	output := sigmoid(outputSum)

	return output
}

func dotProduct(a, b []float64) float64 {
	sum := 0.0

	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Forward pass that returns both hidden layer and output (needed for backpropagation)
func forwardPassWithHidden(
	inputs []float64,
	weightsInputHidden [][]float64,
	biasesHidden []float64,
	weightsHiddenOutput [][]float64,
	biasesOutput []float64,
) ([]float64, float64) {
	hiddenLayer := createVector(hiddenNodes)

	// Calculate hidden layer
	for i := range hiddenLayer {
		sum := dotProduct(weightsInputHidden[i], inputs) + biasesHidden[i]
		hiddenLayer[i] = sigmoid(sum)
	}

	// Calculate output
	outputSum := dotProduct(weightsHiddenOutput[0], hiddenLayer) + biasesOutput[0]
	output := sigmoid(outputSum)

	return hiddenLayer, output
}

// Mean squared error loss function
func meanSquaredError(predicted, target float64) float64 {
	diff := predicted - target
	return 0.5 * diff * diff
}

// Derivative of sigmoid function
func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// Backpropagation algorithm
func backpropagate(
	inputs []float64,
	hiddenLayer []float64,
	output float64,
	target float64,
	weightsInputHidden [][]float64,
	weightsHiddenOutput [][]float64,
	biasesHidden []float64,
	biasesOutput []float64,
) {
	// Calculate output layer error
	outputError := target - output
	outputDelta := outputError * sigmoidDerivative(output)

	// Update weights and bias for hidden-to-output connections
	for i := range weightsHiddenOutput[0] {
		weightsHiddenOutput[0][i] += learningRate * outputDelta * hiddenLayer[i]
	}
	biasesOutput[0] += learningRate * outputDelta

	// Calculate hidden layer errors
	hiddenErrors := make([]float64, hiddenNodes)
	for i := range hiddenErrors {
		hiddenErrors[i] = outputDelta * weightsHiddenOutput[0][i]
	}

	// Calculate hidden layer deltas
	hiddenDeltas := make([]float64, hiddenNodes)
	for i := range hiddenDeltas {
		hiddenDeltas[i] = hiddenErrors[i] * sigmoidDerivative(hiddenLayer[i])
	}

	// Update weights and biases for input-to-hidden connections
	for i := range weightsInputHidden {
		for j := range weightsInputHidden[i] {
			weightsInputHidden[i][j] += learningRate * hiddenDeltas[i] * inputs[j]
		}
		biasesHidden[i] += learningRate * hiddenDeltas[i]
	}
}
