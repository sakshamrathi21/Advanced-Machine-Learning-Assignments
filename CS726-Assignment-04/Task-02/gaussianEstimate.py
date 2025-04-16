import numpy as np
import matplotlib.pyplot as plt

def branin_hoo(x):
    """Calculate the Branin-Hoo function value for given input."""
    x1, x2 = x
    a = 1.0
    f = a * (x2 - (5.1 / (4 * np.pi**2)) * x1**2 + (5 / np.pi) * x1 - 6)**2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10
    return f
    
# Kernel Functions (Students implement)
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF kernel."""
    num = np.sum((x1[:, None, :] - x2[None, :, :])**2, axis=2)
    k = sigma_f**2 * np.exp(- num / (2 * length_scale**2))
    return k

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matérn kernel (nu=1.5)."""
    dist = np.sqrt(np.sum((x1[:, None, :] - x2[None, :, :])**2, axis=2))
    temp = np.sqrt(3) * (dist / length_scale)
    k = sigma_f**2 * (1 + temp)* np.exp(-temp)
    return  k

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel."""
    sqdist = np.sum((x1[:, None, :] - x2[None, :, :])**2, axis=2)
    k = sigma_f**2 * (1 + (sqdist / (2 * alpha * length_scale**2)))**(-alpha)
    return k

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    """Compute the log-marginal likelihood."""
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    lml = -0.5 * y_train.T @ alpha - np.sum(np.log(np.diagonal(L))) - len(y_train) / 2 * np.log(2 * np.pi)
    return lml

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """Optimize hyperparameters using grid search."""
    best_lml = -np.inf
    best_params = (1.0, 1.0)
    for length_scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for sigma_f in [0.1, 1.0, 5.0]:
            lml = log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise)
            if lml > best_lml:
                best_lml = lml
                best_params = (length_scale, sigma_f)
    return best_params[0], best_params[1], noise

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """Perform GP prediction."""
    training_K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    cross_K = kernel_func(x_train, x_test, length_scale, sigma_f)
    test_K = kernel_func(x_test, x_test, length_scale, sigma_f)
    
    L = np.linalg.cholesky(training_K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    mu = cross_K.T @ alpha
    v = np.linalg.solve(L, cross_K)
    cov = test_K - v.T @ v
    std = np.sqrt(np.clip(np.diag(cov), 1e-10, np.inf))
    return mu, std

# Acquisition Functions (Simplified, no erf)
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    z = (mu - y_best - xi) / sigma
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    Phi = 1 / (1 + np.exp(-1.702 * z))
    ei = (mu - y_best - xi) * Phi + sigma * phi
    return ei

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Probability of Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    z = (mu - y_best - xi) / sigma
    pi = 1 / (1 + np.exp(-1.702 * z)) 
    return pi

def plot_graph(x1_grid, x2_grid, z_values, x_train, title, filename):
    """Create and save a contour plot."""
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(x1_grid, x2_grid, z_values, cmap='viridis', levels=50)
    plt.colorbar(cp)
    plt.scatter(x_train[:, 0], x_train[:, 1], c='r', edgecolor='k', s=50, label='Train Points')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    np.random.seed(0)
    n_samples_list = [10, 20, 50, 100]
    kernels = {
        'rbf': (rbf_kernel, 'RBF'),
        'matern': (matern_kernel, 'Matern (nu=1.5)'),
        'rational_quadratic': (rational_quadratic_kernel, 'Rational Quadratic')
    }
    acquisition_strategies = {
        'EI': expected_improvement,
        'PI': probability_of_improvement,
        'random': 'random'
    }
    
    x1_test = np.linspace(-5, 10, 100)
    x2_test = np.linspace(0, 15, 100)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    true_values = np.array([branin_hoo([x1, x2]) for x1, x2 in x_test]).reshape(x1_grid.shape)
    
    for kernel_name, (kernel_func, kernel_label) in kernels.items():
        for n_samples in n_samples_list:
            x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
            y_train = np.array([branin_hoo(x) for x in x_train])
            
            print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
            length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train, kernel_func)
            
            for acq_name, acq_func in acquisition_strategies.items():
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()
                
                y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, 
                                                        kernel_func, length_scale, sigma_f, noise)
                y_mean_grid = y_mean.reshape(x1_grid.shape)
                y_std_grid = y_std.reshape(x1_grid.shape)

                if acq_func == 'random':
                    next_idx = np.random.randint(len(x_test))
                    x_new = x_test[next_idx]
                    y_new = branin_hoo(x_new)
                
                elif acq_func is not None:
                    # Hint: Find y_best, apply acq_func, select new point, update training set, recompute GP
                    y_best = np.min(y_train_current)
                    acq_values = acq_func(y_mean, y_std, y_best)
                    next_idx = np.argmax(acq_values)
                    x_new = x_test[next_idx]
                    y_new = branin_hoo(x_new)

                    x_train_current = np.vstack([x_train_current, x_new])
                    y_train_current = np.append(y_train_current, y_new)

                    y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, kernel_func, length_scale, sigma_f, noise)
                    y_mean_grid = y_mean.reshape(x1_grid.shape)
                    y_std_grid = y_std.reshape(x1_grid.shape)
                
                acq_label = 'random' if acq_name == 'random' else f', Acq={acq_name}'
                plot_graph(x1_grid, x2_grid, true_values, x_train_current,
                          f'True Branin-Hoo Function (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'true_function_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_mean_grid, x_train_current,
                          f'GP Predicted Mean (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_mean_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_std_grid, x_train_current,
                          f'GP Predicted Std Dev (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_std_{kernel_name}_n{n_samples}_{acq_name}.png')

if __name__ == "__main__":
    main()