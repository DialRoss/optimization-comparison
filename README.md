# Comparative Analysis of Optimization Algorithms

A Python implementation from scratch of Gradient Descent, L-BFGS, and Adam algorithms tested on classic benchmark functions (Rosenbrock and Himmelblau). This project visually and quantitatively compares their performance in terms of convergence and robustness.

## Features

- **Analytical Gradients**: Implemented exact gradients for precise optimization.
- **Visualization**: 2D contour plots with optimization paths for intuitive understanding.
- **Comparative Analysis**: Quantitative evaluation of convergence speed and accuracy.

## Algorithms Implemented

1.  **Gradient Descent (GD)** with backtracking line search.
2.  **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) quasi-Newton method.
3.  **Adam** (Adaptive Moment Estimation).

## Test Functions

1.  **Rosenbrock Function**: A classic ill-conditioned problem to test efficiency on "long, narrow valleys".
2.  **Himmelblau's Function**: A non-convex function with four equal-valued minima to test robustness to local minima.

## Results

The analysis demonstrates that:
- L-BFGS excels on well-behaved but ill-conditioned problems like Rosenbrock, exhibiting superlinear convergence.
- Adam is robust for non-convex landscapes like Himmelblau, adapting well to different geometries.
- Gradient Descent serves as a baseline but is often inefficient compared to more advanced methods.

![Optimization Paths](images/optimization_paths.png)  <!-- Add an image later -->

## Installation & Usage

1.  Clone the repo:
    ```bash
    git clone https://github.com/yourusername/optimization-comparison.git
    cd optimization-comparison
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Jupyter notebook:
    ```bash
    jupyter notebook notebooks/demo.ipynb
    ```

## Project Structure
