import pymc as pm
import numpy as np

def main():
    print("Testing PyMC Metropolis with blocked=True...")
    with pm.Model() as model:
        # Define a simple multi-dimensional model similar to ours
        alpha = pm.Normal('alpha', mu=0, sigma=3)
        phi_raw = pm.Normal('phi_raw', mu=0, sigma=1, shape=244)
        gamma_raw = pm.Normal('gamma_raw', mu=0, sigma=1, shape=3)
        sigma_phi = pm.HalfNormal('sigma_phi', sigma=1)
        sigma_t = pm.HalfNormal('sigma_t', sigma=1)
        
        phi = phi_raw * sigma_phi
        gamma = gamma_raw * sigma_t
        
        # Simple likelihood
        mu = pm.math.exp(alpha + phi[0] + gamma[0])
        y = pm.Poisson('y', mu=mu, observed=np.array([5]))
        
        # Set step to blocked Metropolis
        step = pm.Metropolis(vars=[alpha, phi_raw, gamma_raw, sigma_phi, sigma_t], blocked=True)
        
        print("Starting sampling test...")
        trace = pm.sample(draws=500, tune=500, chains=2, step=step, random_seed=42, progressbar=False)
        print("Sampling completed successfully!")
        print("Chains:", trace.posterior.dims['chain'])
        print("Draws:", trace.posterior.dims['draw'])

if __name__ == "__main__":
    main()
