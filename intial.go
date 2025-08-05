package main

import (
	"math/rand"
)

func createMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

func createVector(size int) []float64 {
	return make([]float64, size)
}

func initializeWeightsAndBiases(
	weightsInputHidden [][]float64,
	weightsHiddenOutput [][]float64,
	biasesHidden []float64,
	biasesOutput []float64,
) {
	for i, row := range weightsInputHidden {
		for j := range row {
			weightsInputHidden[i][j] = rand.Float64()
		}
	}

	for i, row := range weightsHiddenOutput {
		for j := range row {
			weightsHiddenOutput[i][j] = rand.Float64()
		}
	}

	for i := range biasesHidden {
		biasesHidden[i] = rand.Float64()
	}

	for i := range biasesOutput {
		biasesOutput[i] = rand.Float64()
	}
}
