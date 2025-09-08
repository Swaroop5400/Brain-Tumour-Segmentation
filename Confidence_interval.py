import numpy as np
import scipy.stats as st


""" Approach A :t-distribution-based CI (assumes approximate normality)"""
def ci_t_based(data, alpha=0.05):
    """
    data: 1D array of per-case scores (e.g., Dice per patient)
    returns: (mean, lower, upper)
    """
    arr = np.asarray(data)
    n = len(arr)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n)
    tval = st.t.ppf(1 - alpha/2, df=n-1)
    lower = mean - tval * se
    upper = mean + tval * se
    return mean, lower, upper

"""Approach B: Nonparametric bootstrap CI"""
import numpy as np

def ci_bootstrap(data, alpha=0.05, n_bootstraps=2000, agg=np.mean, random_seed=None):
    """
    data: 1D array of per-case scores
    agg: statistic to compute (np.mean or np.median)
    Returns: (statistic, lower, upper)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    arr = np.asarray(data)
    n = len(arr)
    boots = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(arr, size=n, replace=True)
        boots.append(agg(sample))
    boots = np.array(boots)
    lower = np.percentile(boots, 100 * (alpha/2))
    upper = np.percentile(boots, 100 * (1 - alpha/2))
    stat = agg(arr)
    return stat, lower, upper



""" to use , run the below code 
dice_per_case = np.array([...])   # fill with Dice for each test case
mean, lower, upper = ci_t_based(dice_per_case, alpha=0.05)
print(f"Dice mean={mean:.4f} 95% CI = [{lower:.4f}, {upper:.4f}]")

stat, lboot, uboot = ci_bootstrap(dice_per_case, alpha=0.05, n_bootstraps=2000, random_seed=42)
print(f"Bootstrap mean={stat:.4f} 95% CI = [{lboot:.4f}, {uboot:.4f}]")

"""
