import numpy as np

def generate_mixed_distribution(bathymetry, alpha=0.7):
        valid = bathymetry > 0
        if not np.any(valid):
            raise ValueError("No ocean values found.")
        # sigmoid with preference for shallow areas
        p_sig_coast = generate_sigmoid_probability_distribution(bathymetry, percentile = 25)
        # sigmoid with preference with no preference for neither shallow or deep areas
        p_sig_deep = generate_sigmoid_probability_distribution(bathymetry, percentile = 50)#75 for deep areas
        # mix and normalize
        p_mix = alpha * p_sig_coast + (1 - alpha) * p_sig_deep
        p_mix /= p_mix.sum()
        return p_mix

def generate_basic_probability_distribution(bathymetry):
        epsilon = 1e-3
        # If bathymetry is 0, we set density to 0.0, otherwise we set it to 1.0/(bathymetry + epsilon)
        # with the inverse, we try to give more "importance" to the shallow areas (where bathymetry is small)
        # We add epsilon to avoid division by zero and to ensure that the density is not infinite as a safeguard.
        density = np.where(bathymetry > 0, 1.0 / (np.sqrt(bathymetry) + epsilon), 0)
        p = density / np.sum(density) # Normalize the density to sum to 1
        return p

def generate_sigmoid_probability_distribution(bathymetry, percentile=25):
    """+
    Generates a probability matrix from bathymetry using a sigmoid function,
    automatically computing the inflection point (median) and the slope (IQR)
    based on sea values (bathymetry > 0). Land values (bathymetry == 0) are excluded
    and assigned a probability of 0.
    
    Parameters:
        bathymetry: 2D array with depth values, where 0 represents land and values > 0 represent sea.
        
    Returns:
        p: 2D array with the same shape as 'bathymetry', containing the normalized probability 
        distribution considering only sea values.
    """
    # Create a mask for sea values (bathymetry > 0)
    valid = bathymetry > 0
    if not np.any(valid):
        raise ValueError("No sea values found (bathymetry > 0) to compute probability.")

    # Extract only valid sea values
    valid_values = bathymetry[valid]

    # Compute median and interquartile range (IQR) from valid values
    percentile = np.percentile(valid_values, percentile)
    q75, q25 = np.percentile(valid_values, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        iqr = 1e-3  # Avoid division by zero in case of low variability

    # Apply the sigmoid function: shallower regions (lower bathymetry) 
    # get higher values, deeper regions get lower ones
    p_valid = 1.0 / (1.0 + np.exp((valid_values - percentile) / iqr))

    # Create a probability matrix with the same shape as bathymetry, assigning 0 to land
    p_all = np.zeros(bathymetry.shape)
    p_all[valid] = p_valid

    # Normalize only sea values so that total sum is 1
    p_all /= p_all.sum()

    return p_all
