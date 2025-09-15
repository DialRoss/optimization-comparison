import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # <-- AJOUT IMPORTANT !

def plot_optimization_paths(paths, f, title, labels=None, x_lim=(-2, 2), y_lim=(-1, 3)):
    """
    Plot optimization paths on a contour plot of the function.
    """
    # Generate grid points
    xlist = np.linspace(x_lim[0], x_lim[1], 400)
    ylist = np.linspace(y_lim[0], y_lim[1], 400)
    X, Y = np.meshgrid(xlist, ylist)
    
    # Compute function values
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    # Create plot
    plt.figure(figsize=(12, 9))
    levels = np.logspace(-1, 3, 50)
    contour = plt.contourf(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis', alpha=0.8)  # <-- ICI: LogNorm() sans plt.
    plt.colorbar(contour)
    
    # Plot optimization paths
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    if labels is None:
        labels = [f'Method {i+1}' for i in range(len(paths))]
    
    for i, path in enumerate(paths):
        if i < len(colors):
            plt.plot(path[:, 0], path[:, 1], marker=markers[i], color=colors[i], 
                    markersize=4, linewidth=1.5, label=labels[i], markevery=5)
    
    plt.legend(fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

def compare_convergence(paths, f, labels=None):
    """
    Compare convergence of different optimization algorithms.
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    if labels is None:
        labels = [f'Method {i+1}' for i in range(len(paths))]
    
    for i, path in enumerate(paths):
        if i < len(colors):
            # Compute function values along the path
            f_values = [f(point) for point in path]
            plt.semilogy(f_values, color=colors[i], linewidth=2, label=labels[i])
    
    plt.legend(fontsize=12)
    plt.title('Convergence Comparison', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Function Value (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()