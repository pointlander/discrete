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
	length := len(rawData)
	sample := func(rngSeed int64) (x [150][]int) {
		clusters, _, err := kmeans.Kmeans(rngSeed, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := range x {
			x[i] = make([]int, 150)
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					x[i][j]++
				}
			}
		}
		return x
	}
	var sum [150][]int
	for i := range sum {
		sum[i] = make([]int, 150)
	}
	for i := 0; i < 100; i++ {
		x := sample(int64(i) + 1)
		for i := range sum {
			for j := range sum[i] {
				sum[i][j] += x[i][j]
			}
		}
	}
	rawData = make([][]float64, length)
	for i := 0; i < len(sum); i++ {
		for _, value := range sum[i] {
			rawData[i] = append(rawData[i], float64(value))
		}
	}
	clusters, _, err := kmeans.Kmeans(1, rawData, 3, kmeans.SquaredEuclideanDistance, -1)
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
			fmt.Println(datum.Fisher[i].Label, i, v)
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
			fmt.Println(datum.Fisher[i].Label, i, v)
		}
		entropy(clusters)

		/*for i := 0; i < len(a.X); i += a.S[0] {
			max, index := 0.0, 0
			for key, value := range a.X[i : i+a.S[0]] {
				if value > max {
					max, index = value, key
				}
			}
			println(max, index)
		}

		ii = 0
		for i := 0; i < len(a.X); i += a.S[0] {
			if ii%50 == 0 {
				fmt.Println()
			}
			fmt.Println(ii, a.X[i:i+a.S[0]])
			ii++
		}*/
		return true
	})
}

var (
	// FlagXOR is the xor mode
	FlagXOR = flag.Bool("xor", false, "xor mode")
	// FlagIRIS is the iris mode
	FlagIRIS = flag.Bool("iris", false, "iris mode")
)

func main() {
	flag.Parse()

	if *FlagXOR {
		XOR()
		return
	} else if *FlagIRIS {
		IRIS()
		return
	}
}
