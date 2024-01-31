import numpy as np

def swap_bits(binary_str, k, m):
    # Convert the binary string to a list of characters
    binary_list = list(binary_str)

    # Check if the kth and mth positions are within the length of the binary string
    if k < len(binary_list) and m < len(binary_list):
        # Swap the kth and mth bits
        binary_list[k], binary_list[m] = binary_list[m], binary_list[k]

    # Convert the list back to a string
    swapped_binary_str = ''.join(binary_list)

    return swapped_binary_str


def rearrange_state(input_state, pos_qubits, no_qubits):
    pos_qubits.sort()

    coeff_with_bit = []
    bit = 0
    for coeff in input_state:
        #arranging the coefficients (a_i) in increasing order of bit strings (bin(i))
        coeff_with_bit.append([format(bit, f'0{no_qubits}b'), coeff])
        bit += 1
    new_pos = 0
    #swapping pos[0] and 0th bit, pos[1] and 1th bit, and so on
    for pos in pos_qubits:
        for coeff in coeff_with_bit:
            coeff[0] = swap_bits(coeff[0], pos, new_pos)
        new_pos += 1

    #now arranging the coefficients in increasing order of new bit strings
    new_coeff_with_bits = sorted(coeff_with_bit, key = lambda x: int(x[0],2))
    #taking the second element of each element i.e. new coefficient
    new_coeff = [row[1] for row in new_coeff_with_bits]

    return new_coeff


def red_density_matrix(input_state, subspace_qu, space_qu):
    dim_subspace = 2**subspace_qu
    tracedspace_qu = space_qu - subspace_qu
    dim_tracedspace = 2**tracedspace_qu
    # density_matrix = np.outer(np.conj(input_state), input_state)
    input_state_cj = np.conj(input_state)
    red_den = np.zeros((dim_subspace, dim_subspace), dtype = complex)
    indices = np.arange(dim_subspace)

    #search through all the indices, we have [row, column] element
    #for |ab><cd| we are tracing our |a><c| system
    for row in indices:
        for column in indices:
            #taking the inner product 
            for i in range(dim_tracedspace):
                j = i*dim_subspace
                red_den[row][column] += input_state_cj[row+j]*input_state[column+j]
    return red_den


def reduced_density_matrix(state, subspace_qu, space_qu):
    return red_density_matrix(
        rearrange_state(state, subspace_qu, space_qu),
        len(subspace_qu), space_qu)


def calculate_entanglement_entropy(red_den):
    eigenvalues = np.linalg.eigvals(red_den)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-12]

    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return entropy


def calculate_spacing_ratios(red_den):
    eigenvalues = np.linalg.eigvals(red_den)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-12]

    eigenvalues = np.sort(eigenvalues)[::-1]

    differences = -np.diff(eigenvalues)
    differences = differences[np.abs(differences)>1e-12]  # Remove zero differences

    spacing_ratios = differences[1:] / differences[:-1]

    return spacing_ratios