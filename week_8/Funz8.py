"""
Library of self-made functions needed for the codes implemented for the exercises of the 5th week

@author: david
"""

import numpy as np
import time



def real_time_metropolis_cumulative_variance_check(proposal_std=1.0, check_interval=10, threshold = 0.05):
    current_set = []
    cumulative_set = []
    final_index = None

    acceptance_count = 0

    def target_distribution(x):
        # Replace this with your actual target distribution
        return np.exp(-0.5 * x**2)

    def metropolis_sampler(current_sample):
        proposed_sample = current_sample + np.random.normal(0, 5*proposal_std)
        acceptance_ratio = min(1, target_distribution(proposed_sample) / target_distribution(current_sample))

        if np.random.uniform(0, 1) < acceptance_ratio:
            return proposed_sample, 1
        else:
            return current_sample, 0

    try:
        i = 0
        current_sample = np.random.normal(0, 1)  # Initial sample from the target distribution

        while True:
            current_sample, acceptance = metropolis_sampler(current_sample)
            acceptance_count += acceptance

            current_set.append(current_sample)
            cumulative_set.extend(current_set)

            if i % check_interval == 0 and i > 0:
                # Calculate cumulative variance
                cumulative_variance = np.var(cumulative_set)
                #print(f"Cumulative Variance for samples 1-{i}: {cumulative_variance}")

                # Check threshold
                if abs(cumulative_variance - 1) < threshold:
                    final_index = i
                    #print(f"Stopping at samples 1-{i} due to exceeding the threshold.")
                    break

            i += 1

    except KeyboardInterrupt:
        pass  # Stop the loop if interrupted by the user
    """
    # Report the final index i
    print(f"Final index i: {final_index}")
    acceptance_rate = acceptance_count / i
    print(f"Acceptance rate: {acceptance_rate}")
    """
    return final_index, current_sample

# Call the function to run real-time Metropolis sampling and cumulative variance check
print(real_time_metropolis_cumulative_variance_check())