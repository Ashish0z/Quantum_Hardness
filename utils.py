import numpy as np

def generate_random_bipartition(n):
    # generate a random permutation of the set {1, 2, ..., n}
    perm = np.random.permutation(n)
    # split the set into two parts
    A = perm[:n//2]

    return A

def get_unique_bipartitions(n, num_partitions):
    partitions = []
    while len(partitions) < num_partitions:
        A = generate_random_bipartition(n)
        A = np.sort(A)
        is_unique = True
        for B in partitions:
            if np.array_equal(A, B):
                is_unique = False
                break
        if is_unique:
            partitions.append(A)
    return partitions

def get_n_bit_random_instances(num_instance, n_bit, instances, n_bits):
    n_bit_instances = instances[n_bits == n_bit]
    return np.random.choice(n_bit_instances, num_instance, replace=False)
