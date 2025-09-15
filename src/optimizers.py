import numpy as np

def line_search(f, x, grad, direction, alpha=1.0, beta=0.8, c=1e-4):
    """
    Backtracking line search satisfying Armijo condition.
    """
    fx = f(x)
    while f(x + alpha * direction) > fx + c * alpha * np.dot(grad, direction):
        alpha *= beta
        if alpha < 1e-8:
            break
    return alpha

def gradient_descent(f, grad_f, x0, max_iter=1000, tol=1e-6):
    """
    Gradient Descent with backtracking line search.
    """
    x = x0.copy()
    path = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        direction = -grad
        alpha = line_search(f, x, grad, direction)
        x += alpha * direction
        path.append(x.copy())
        
    return np.array(path)

def l_bfgs(f, grad_f, x0, max_iter=100, tol=1e-6, m=10):
    """
    L-BFGS optimization algorithm (basic implementation).
    """
    x = x0.copy()
    n = len(x)
    s_list = []
    y_list = []
    rho_list = []
    path = [x.copy()]
    
    for k in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        q = grad
        alpha_list = []
        
        # First loop
        for i in reversed(range(len(s_list))):
            s = s_list[i]
            y = y_list[i]
            rho = rho_list[i]
            alpha = rho * np.dot(s, q)
            q = q - alpha * y
            alpha_list.append(alpha)
            
        # Scaling
        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
        else:
            gamma = 1.0
            
        r = gamma * q
        
        # Second loop
        for i in range(len(s_list)):
            s = s_list[i]
            y = y_list[i]
            rho = rho_list[i]
            beta_val = rho * np.dot(y, r)
            alpha = alpha_list[len(s_list) - 1 - i]
            r = r + s * (alpha - beta_val)
            
        direction = -r
        
        # Simple fixed step (you can replace with line search)
        alpha_val = 1e-3
        x_new = x + alpha_val * direction
        s = x_new - x
        y = grad_f(x_new) - grad
        
        if np.linalg.norm(s) < 1e-10:
            break
            
        rho = 1.0 / np.dot(y, s)
        
        # Update memory
        if len(s_list) == m:
            s_list.pop(0)
            y_list.pop(0)
            rho_list.pop(0)
            
        s_list.append(s)
        y_list.append(y)
        rho_list.append(rho)
        x = x_new
        path.append(x.copy())
        
    return np.array(path)

def adam(f, grad_f, x0, max_iter=1000, tol=1e-6, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam optimization algorithm.
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    
    for t in range(1, max_iter + 1):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x.copy())
        
    return np.array(path)