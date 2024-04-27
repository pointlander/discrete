// Copyright 2024 The Discrete Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/kmeans"
	"github.com/pointlander/matrix"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.9
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.99
	// Eta is the learning rate
	Eta = .1
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// XOR xor mode
func XOR() {
	rng := rand.New(rand.NewSource(1))
	set := tf64.NewSet()
	set.Add("w1", 2, 8)
	set.Add("b1", 8)
	set.Add("w2", 16, 4)
	set.Add("b2", 4)

	input := tf64.NewV(2, 4)
	input.X = append(input.X, -1, -1, 1, -1, -1, 1, 1, 1)

	for i := range set.Weights {
		w := set.Weights[i]
		size := w.S[0] * w.S[1]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:size]
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < size; i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	l1 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l2 := tf64.Softmax(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2")))
	loss := tf64.Avg(tf64.Entropy(l2))

	iterations := 64
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	pow := func(x float64, i int) float64 {
		y := math.Pow(x, float64(i+1))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return y
	}
	for i := 0; i < iterations; i++ {
		set.Zero()
		input.Zero()

		cost := tf64.Gradient(loss).X[0]
		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1, i), pow(B2, i)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(i, cost, time.Now().Sub(start))
		start = time.Now()
		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		if cost < .00001 {
			fmt.Println("stopping...")
			break
		}
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	l2(func(a *tf64.V) bool {
		for i := 0; i < len(a.X); i += a.S[0] {
			fmt.Println(a.X[i : i+a.S[0]])
		}
		return true
	})
}

func meta(rawData [][]float64) []int {
	sample := func(rngSeed int64, x [][]float64) {
		clusters, _, err := kmeans.Kmeans(rngSeed, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := range x {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					x[i][j]++
				}
			}
		}
	}
	length := len(rawData)
	meta := make([][]float64, length)
	for i := range meta {
		meta[i] = make([]float64, length)
	}
	for i := 0; i < 100; i++ {
		sample(int64(i)+1, meta)
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	return clusters
}

// IRIS iris mode
func IRIS() {
	rng := rand.New(rand.NewSource(1))
	set := tf64.NewSet()
	set.Add("w1", 4, 16)
	set.Add("b1", 16)
	set.Add("w2", 32, 4)
	set.Add("b2", 4)

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	max := 0.0
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	input := tf64.NewV(4, 150)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			input.X = append(input.X, measure/max)
		}
	}

	for i := range set.Weights {
		w := set.Weights[i]
		size := w.S[0] * w.S[1]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:size]
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < size; i++ {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float64, len(w.X))
		}
	}

	l1 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w1"), input.Meta()), set.Get("b1")))
	l2 := tf64.Softmax(tf64.Add(tf64.Mul(set.Get("w2"), l1), set.Get("b2")))
	loss := tf64.Avg(tf64.Entropy(l2))

	iterations := 2 * 1024
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	pow := func(x float64, i int) float64 {
		y := math.Pow(x, float64(i+1))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return y
	}
	for i := 0; i < iterations; i++ {
		set.Zero()
		input.Zero()

		cost := tf64.Gradient(loss).X[0] / 150.0
		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1, i), pow(B2, i)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(i, cost, time.Now().Sub(start))
		start = time.Now()
		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		if cost < .01 {
			fmt.Println("stopping...")
			break
		}
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	entropy := func(clusters []int) {
		ab, ba := [3][3]float64{}, [3][3]float64{}
		for i := range datum.Fisher {
			a := int(iris.Labels[datum.Fisher[i].Label])
			b := clusters[i]
			ab[a][b]++
			ba[b][a]++
		}
		entropy := 0.0
		for i := 0; i < 3; i++ {
			entropy += (1.0 / 3.0) * math.Log(1.0/3.0)
		}
		fmt.Println(-entropy, -(1.0/3.0)*math.Log(1.0/3.0))
		for i := range ab {
			entropy := 0.0
			for _, value := range ab[i] {
				if value > 0 {
					p := value / 150
					entropy += p * math.Log(p)
				}
			}
			entropy = -entropy
			fmt.Println("ab", i, entropy)
		}
		for i := range ba {
			entropy := 0.0
			for _, value := range ba[i] {
				if value > 0 {
					p := value / 150
					entropy += p * math.Log(p)
				}
			}
			entropy = -entropy
			fmt.Println("ba", i, entropy)
		}
	}
	l2(func(a *tf64.V) bool {
		rawData := make([][]float64, len(datum.Fisher))
		ii := 0
		for i := 0; i < len(a.X); i += a.S[0] {
			for j := 0; j < a.S[0]; j++ {
				rawData[ii] = append(rawData[ii], a.X[i+j])
			}
			ii++
		}
		clusters := meta(rawData)
		for i, v := range clusters {
			fmt.Printf("%3d %15s %d\n", i, datum.Fisher[i].Label, v)
		}
		entropy(clusters)

		fmt.Println()
		rawData = make([][]float64, len(datum.Fisher))
		for i, data := range datum.Fisher {
			for _, value := range data.Measures {
				rawData[i] = append(rawData[i], value/max)
			}
		}
		clusters = meta(rawData)
		for i, v := range clusters {
			fmt.Printf("%3d %15s %d\n", i, datum.Fisher[i].Label, v)
		}
		entropy(clusters)

		return true
	})
}

func meta2(rawData [][][]float64) []int {
	sample := func(rawData [][]float64, rngSeed int64, x [][]float64) {
		clusters, _, err := kmeans.Kmeans(rngSeed, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := range x {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					x[i][j]++
				}
			}
		}
	}
	length := len(rawData[0])
	meta := make([][]float64, length)
	for i := range meta {
		meta[i] = make([]float64, length)
	}
	for i := 0; i < 100; i++ {
		for r := range rawData {
			sample(rawData[r], int64(i)+1, meta)
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	return clusters
}

// Multi is the multi mode
func Multi() {
	const count = 8
	rng := rand.New(rand.NewSource(1))
	set := [count]tf64.Set{}
	for i := range set {
		set[i] = tf64.NewSet()
		set[i].Add("w1", 4, 16)
		set[i].Add("b1", 16)
		set[i].Add("w2", 32, 4)
		set[i].Add("b2", 4)
	}

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	max := 0.0
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	input := tf64.NewV(4, 150)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			input.X = append(input.X, measure/max)
		}
	}

	for s := range set {
		for i := range set[s].Weights {
			w := set[s].Weights[i]
			size := w.S[0] * w.S[1]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:size]
				w.States = make([][]float64, StateTotal)
				for i := range w.States {
					w.States[i] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for i := 0; i < size; i++ {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float64, len(w.X))
			}
		}
	}

	loss := [count]tf64.Meta{}
	l2 := [count]tf64.Meta{}
	for i := range loss {
		l1 := tf64.Everett(tf64.Add(tf64.Mul(set[i].Get("w1"), input.Meta()), set[i].Get("b1")))
		l2[i] = tf64.Softmax(tf64.Add(tf64.Mul(set[i].Get("w2"), l1), set[i].Get("b2")))
		loss[i] = tf64.Avg(tf64.Entropy(l2[i]))
	}

	iterations := 2 * 1024
	points := make(plotter.XYs, 0, iterations)
	start := time.Now()
	pow := func(x float64, i int) float64 {
		y := math.Pow(x, float64(i+1))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return y
	}
	for s, loss := range loss {
		for i := 0; i < iterations; i++ {
			set[s].Zero()
			input.Zero()

			cost := tf64.Gradient(loss).X[0] / 150.0
			norm := 0.0
			for _, p := range set[s].Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1, i), pow(B2, i)
			scaling := 1.0
			if norm > 1 {
				scaling = 1 / norm
			}
			for _, w := range set[s].Weights {
				for l, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			fmt.Println(i, cost, time.Now().Sub(start))
			start = time.Now()
			points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
			if cost < .01 {
				fmt.Println("stopping...")
				break
			}
		}
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	entropy := func(clusters []int) {
		ab, ba := [3][3]float64{}, [3][3]float64{}
		for i := range datum.Fisher {
			a := int(iris.Labels[datum.Fisher[i].Label])
			b := clusters[i]
			ab[a][b]++
			ba[b][a]++
		}
		entropy := 0.0
		for i := 0; i < 3; i++ {
			entropy += (1.0 / 3.0) * math.Log(1.0/3.0)
		}
		fmt.Println(-entropy, -(1.0/3.0)*math.Log(1.0/3.0))
		for i := range ab {
			entropy := 0.0
			for _, value := range ab[i] {
				if value > 0 {
					p := value / 150
					entropy += p * math.Log(p)
				}
			}
			entropy = -entropy
			fmt.Println("ab", i, entropy)
		}
		for i := range ba {
			entropy := 0.0
			for _, value := range ba[i] {
				if value > 0 {
					p := value / 150
					entropy += p * math.Log(p)
				}
			}
			entropy = -entropy
			fmt.Println("ba", i, entropy)
		}
	}
	rawData := make([][][]float64, count)
	for r := range rawData {
		rawData[r] = make([][]float64, len(datum.Fisher))
		l2[r](func(a *tf64.V) bool {
			ii := 0
			for i := 0; i < len(a.X); i += a.S[0] {
				for j := 0; j < a.S[0]; j++ {
					rawData[r][ii] = append(rawData[r][ii], a.X[i+j])
				}
				ii++
			}
			return true
		})
	}

	clusters := meta2(rawData)
	for i, v := range clusters {
		fmt.Printf("%3d %15s %d\n", i, datum.Fisher[i].Label, v)
	}
	entropy(clusters)

	fmt.Println()
	control := make([][]float64, len(datum.Fisher))
	for i, data := range datum.Fisher {
		for _, value := range data.Measures {
			control[i] = append(control[i], value/max)
		}
	}
	clusters = meta(control)
	for i, v := range clusters {
		fmt.Printf("%3d %15s %d\n", i, datum.Fisher[i].Label, v)
	}
	entropy(clusters)

}

// X is the x mode
func X() {
	rng := matrix.Rand(1)

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	max := 0.0
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			if measure > max {
				max = measure
			}
		}
	}
	input := matrix.NewMatrix(4, 150)
	for _, data := range datum.Fisher {
		for _, measure := range data.Measures {
			input.Data = append(input.Data, float32(measure/max))
		}
	}

	optimizer := matrix.NewOptimizer(&rng, 8, .1, 4, func(samples []matrix.Sample, x ...matrix.Matrix) {
		for index := range samples {
			x1 := samples[index].Vars[0][0].Sample()
			y1 := samples[index].Vars[0][1].Sample()
			z1 := samples[index].Vars[0][2].Sample()
			w1 := x1.Add(y1.H(z1))

			x2 := samples[index].Vars[1][0].Sample()
			y2 := samples[index].Vars[1][1].Sample()
			z2 := samples[index].Vars[1][2].Sample()
			b1 := x2.Add(y2.H(z2))

			x3 := samples[index].Vars[2][0].Sample()
			y3 := samples[index].Vars[2][1].Sample()
			z3 := samples[index].Vars[2][2].Sample()
			w2 := x3.Add(y3.H(z3))

			x4 := samples[index].Vars[3][0].Sample()
			y4 := samples[index].Vars[3][1].Sample()
			z4 := samples[index].Vars[3][2].Sample()
			b2 := x4.Add(y4.H(z4))

			output := w2.MulT(w1.MulT(input).Add(b1).Everett()).Add(b2)

			rawData := make([][]float64, output.Rows)
			for i := 0; i < output.Rows; i++ {
				for j := 0; j < output.Cols; j++ {
					rawData[i] = append(rawData[i], float64(output.Data[i*output.Cols+j]))
				}
			}
			meta := make([][]float64, output.Rows)
			for i := range meta {
				meta[i] = make([]float64, output.Rows)
			}

			for i := 0; i < 100; i++ {
				clusters, _, err := kmeans.Kmeans(int64(i+1), rawData, 3, kmeans.SquaredEuclideanDistance, -1)
				if err != nil {
					panic(err)
				}
				for i := range meta {
					target := clusters[i]
					for j, v := range clusters {
						if v == target {
							meta[i][j]++
						}
					}
				}
			}

			entropy := 0.0
			for i := range meta {
				sum := 0.0
				for _, value := range meta[i] {
					sum += value
				}
				if sum == 0 {
					continue
				}
				for _, value := range meta[i] {
					if value == 0 {
						continue
					}
					p := value / sum
					entropy += p * math.Log(p)
				}
			}
			samples[index].Cost = -entropy / float64(len(meta))
		}
	}, matrix.NewCoord(4, 4), matrix.NewCoord(4, 1), matrix.NewCoord(8, 8), matrix.NewCoord(8, 1))
	var sample matrix.Sample
	last := -1.0
	for {
		s := optimizer.Iterate()
		if last > 0 && math.Abs(last-s.Cost) < 1e-6 {
			sample = s
			break
		}
		fmt.Println(s.Cost)
		last = s.Cost
	}

	x1 := sample.Vars[0][0].Sample()
	y1 := sample.Vars[0][1].Sample()
	z1 := sample.Vars[0][2].Sample()
	w1 := x1.Add(y1.H(z1))

	x2 := sample.Vars[1][0].Sample()
	y2 := sample.Vars[1][1].Sample()
	z2 := sample.Vars[1][2].Sample()
	b1 := x2.Add(y2.H(z2))

	x3 := sample.Vars[2][0].Sample()
	y3 := sample.Vars[2][1].Sample()
	z3 := sample.Vars[2][2].Sample()
	w2 := x3.Add(y3.H(z3))

	x4 := sample.Vars[3][0].Sample()
	y4 := sample.Vars[3][1].Sample()
	z4 := sample.Vars[3][2].Sample()
	b2 := x4.Add(y4.H(z4))

	output := w2.MulT(w1.MulT(input).Add(b1).Everett()).Add(b2)

	rawData := make([][]float64, output.Rows)
	for i := 0; i < output.Rows; i++ {
		for j := 0; j < output.Cols; j++ {
			rawData[i] = append(rawData[i], float64(output.Data[i*output.Cols+j]))
		}
	}
	meta := make([][]float64, output.Rows)
	for i := range meta {
		meta[i] = make([]float64, output.Rows)
	}

	for i := 0; i < 100; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), rawData, 3, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := range meta {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}

	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, v := range clusters {
		fmt.Printf("%3d %15s %d\n", i, datum.Fisher[i].Label, v)
	}
}

var (
	// FlagXOR is the xor mode
	FlagXOR = flag.Bool("xor", false, "xor mode")
	// FlagIRIS is the iris mode
	FlagIRIS = flag.Bool("iris", false, "iris mode")
	// FlagMulti is the multi mode
	FlagMulti = flag.Bool("multi", false, "multi mode")
	// FlagX is x mode
	FlagX = flag.Bool("x", false, "x mode")
)

func main() {
	flag.Parse()

	if *FlagXOR {
		XOR()
		return
	} else if *FlagIRIS {
		IRIS()
		return
	} else if *FlagMulti {
		Multi()
		return
	} else if *FlagX {
		X()
		return
	}
}
