# multilayer-computing-evolution
a program that drives the rate of change of computing power and explain why more's law was a conceptual mistake. 

# Multilayer Computing Evolution

## Description

A Python research simulation investigating the mathematical structure of technological evolution through historical transistor counts.

The project examines whether computing technology can be understood through multiple layers of change rather than a single growth function.

It compares conventional exponential growth with a variable-rate exponential model using regression, differentiation, and logarithmic transformations.

## Research Hypothesis

Technological systems do not only evolve through changes in their measurable properties. The rates at which those properties change can also evolve.

The project investigates four layers:

1. Computing capacity: N(t)
2. Rate of change: N'(t)
3. Acceleration of change: N''(t)
4. Change in acceleration: N'''(t)

## Mathematical Models

Conventional exponential growth:

N(t) = A * e^(kt)

Variable-rate exponential growth:

N(t) = A * e^(bt + ct²)

The second model allows the relative growth rate to evolve over time.

## Features

- Historical transistor-count dataset (1971–2021).
- Exponential regression.
- Variable-rate exponential regression.
- Numerical first and second derivatives.
- Logarithmic transformations.
- Relative growth-rate analysis.
- R-squared model comparison.
- Six analytical visualizations.
- Mathematical scenario projections.

## Installation

    pip install numpy matplotlib

## Execution

    python multilayer_computing_evolution.py

## Important Distinction

Moore's Law is an empirical technological observation, not a mathematical law asserting linear growth in computing power.

This project investigates the limitations of treating a single empirical growth pattern as a complete explanation of technological evolution.

Transistor count is a proxy for computing capacity, not a direct measurement of computing performance.

The models are exploratory and do not establish a causal theory of technological evolution.
