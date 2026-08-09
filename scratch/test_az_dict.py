import numpy as np
import arviz as az

def test():
    n_zonas = 244
    n_periodos = 3
    n_obs = 732
    
    alpha_samples = np.random.normal(-1.0, 0.05, size=(2, 300))
    phi_samples = np.random.normal(0.0, 0.5, size=(2, 300, n_zonas))
    gamma_samples = np.random.normal(0.0, 0.2, size=(2, 300, n_periodos))
    sigma_phi_samples = np.random.normal(0.5, 0.05, size=(2, 300))
    sigma_t_samples = np.random.normal(0.2, 0.05, size=(2, 300))
    alpha_disp_samples = np.random.normal(1.5, 0.1, size=(2, 300))
    
    posterior = {
        'alpha': alpha_samples,
        'phi': phi_samples,
        'gamma': gamma_samples,
        'sigma_phi': sigma_phi_samples,
        'sigma_t': sigma_t_samples,
        'alpha_disp': alpha_disp_samples
    }
    
    observed_data = {
        'y_obs': np.random.poisson(10, size=n_obs)
    }
    
    posterior_predictive = {
        'y_obs': np.random.poisson(10, size=(2, 300, n_obs))
    }
    
    trace = az.from_dict(
        posterior=posterior,
        observed_data=observed_data,
        posterior_predictive=posterior_predictive
    )
    
    print("Trace attributes:", dir(trace))
    print("Posterior variables:", list(trace.posterior.data_vars))
    print("Posterior predictive variables:", list(trace.posterior_predictive.data_vars))
    print("Observed data variables:", list(trace.observed_data.data_vars))

if __name__ == "__main__":
    test()
