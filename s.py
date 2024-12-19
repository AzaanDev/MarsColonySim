import numpy as np
from scipy.stats import beta
from resource_types import ResourceType



results_baseline = []
results_antithetic = []
base_extraction_rate = 5
num_workers = 50

for _ in range(10000):
    # Baseline: Random uniform sample
    u = np.random.uniform(0, 1)
    beta_factor = beta.ppf(u, 2, 5)
    scaled_beta = 0.8 + beta_factor * (1.5 - 0.8)
    extraction = base_extraction_rate * num_workers * scaled_beta
    results_baseline.append(extraction)

    # Antithetic: Use complementary samples
    u1 = np.random.uniform(0, 1)
    u2 = 1 - u1
    beta_1 = 0.8 + beta.ppf(u1, 2, 5) * (1.5 - 0.8)
    beta_2 = 0.8 + beta.ppf(u2, 2, 5) * (1.5 - 0.8)
    extraction_1 = base_extraction_rate * num_workers * beta_1
    extraction_2 = base_extraction_rate * num_workers * beta_2
    results_antithetic.append((extraction_1 + extraction_2) / 2)

# Calculate means and variances
mean_baseline = np.mean(results_baseline)
var_baseline = np.var(results_baseline)

mean_antithetic = np.mean(results_antithetic)
var_antithetic = np.var(results_antithetic)

# Print the results
print("Baseline Mean:", mean_baseline)
print("Baseline Variance:", var_baseline)
print("Antithetic Mean:", mean_antithetic)
print("Antithetic Variance:", var_antithetic)