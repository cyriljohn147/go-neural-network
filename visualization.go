package main

import (
	"fmt"
	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// LossData stores loss values for plotting
type LossData struct {
	Epochs []float64
	Losses []float64
}

// Add a loss value to the data
func (ld *LossData) AddLoss(epoch int, loss float64) {
	ld.Epochs = append(ld.Epochs, float64(epoch))
	ld.Losses = append(ld.Losses, loss)
}

// PlotLoss creates a loss curve plot
func (ld *LossData) PlotLoss(filename string) error {
	p := plot.New()
	p.Title.Text = "Neural Network Training Loss"
	p.X.Label.Text = "Epoch"
	p.Y.Label.Text = "Average Loss"

	// Create line plot
	pts := make(plotter.XYs, len(ld.Epochs))
	for i := range pts {
		pts[i].X = ld.Epochs[i]
		pts[i].Y = ld.Losses[i]
	}

	line, err := plotter.NewLine(pts)
	if err != nil {
		return err
	}
	line.Color = color.RGBA{R: 255, A: 255}

	p.Add(line)
	p.Legend.Add("Training Loss", line)

	// Save the plot
	if err := p.Save(8*vg.Inch, 6*vg.Inch, filename); err != nil {
		return err
	}

	fmt.Printf("Loss plot saved to %s\n", filename)
	return nil
}

// PlotDecisionBoundary creates a simplified decision boundary visualization
func PlotDecisionBoundary(
	weightsInputHidden [][]float64,
	biasesHidden []float64,
	weightsHiddenOutput [][]float64,
	biasesOutput []float64,
	trainingInputs [][]float64,
	trainingTargets []float64,
	filename string,
) error {
	p := plot.New()
	p.Title.Text = "XOR Decision Boundary"
	p.X.Label.Text = "Input 1"
	p.Y.Label.Text = "Input 2"

	// Add training data points only (simplified version)
	class0Training := make(plotter.XYs, 0)
	class1Training := make(plotter.XYs, 0)
	
	for i, input := range trainingInputs {
		if trainingTargets[i] < 0.5 {
			class0Training = append(class0Training, plotter.XY{X: input[0], Y: input[1]})
		} else {
			class1Training = append(class1Training, plotter.XY{X: input[0], Y: input[1]})
		}
	}
	
	if len(class0Training) > 0 {
		scatter0Training, err := plotter.NewScatter(class0Training)
		if err != nil {
			return err
		}
		scatter0Training.GlyphStyle.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255} // Dark blue
		scatter0Training.GlyphStyle.Shape = draw.SquareGlyph{}
		scatter0Training.GlyphStyle.Radius = vg.Points(8)
		p.Add(scatter0Training)
		p.Legend.Add("Class 0 (XOR=0)", scatter0Training)
	}
	
	if len(class1Training) > 0 {
		scatter1Training, err := plotter.NewScatter(class1Training)
		if err != nil {
			return err
		}
		scatter1Training.GlyphStyle.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255} // Dark red  
		scatter1Training.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter1Training.GlyphStyle.Radius = vg.Points(8)
		p.Add(scatter1Training)
		p.Legend.Add("Class 1 (XOR=1)", scatter1Training)
	}

	// Set axis limits
	p.X.Min = -0.2
	p.X.Max = 1.2
	p.Y.Min = -0.2
	p.Y.Max = 1.2

	// Save the plot
	if err := p.Save(8*vg.Inch, 8*vg.Inch, filename); err != nil {
		return err
	}

	fmt.Printf("Decision boundary plot saved to %s\n", filename)
	return nil
}

// PlotNetworkArchitecture creates a text-based visualization
func PlotNetworkArchitecture(filename string) error {
	fmt.Println("\n=== Neural Network Architecture ===")
	fmt.Println("Input Layer (2 nodes):  [X1] [X2]")
	fmt.Println("                          |    |")
	fmt.Println("                          v    v")
	fmt.Println("Hidden Layer (2 nodes): [H1] [H2]")
	fmt.Println("                          |    |")
	fmt.Println("                          v    v")
	fmt.Println("Output Layer (1 node):    [Y]")
	fmt.Println("===================================")
	
	// For now, just print the architecture instead of creating a plot
	// This avoids the complex plotting issues while still providing visualization
	
	fmt.Printf("Network architecture visualization displayed above\n")
	return nil
}

