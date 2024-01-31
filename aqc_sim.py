import numpy as np
from scipy.integrate import complex_ode
from scipy import sparse

from schedules import *


def sigma_i_sparse(sigma, i, n):
    """Returns the Pauli matrix 'sigma' acting on the i-th qubit as a sparse
    matrix. Note that the i-th qubit is the i-th qubit from the right, whereas
    sometimes sigma_i is defined on the i-th qubit from the left.

    Args:
        sigma: single-qubit Pauli matrix (i.e. X, Y, Z)
        i: qubit to act on
        n: total number of qubits

    Raises:
        Exception: when i > n - 1

    Returns:
        The Pauli matrix acting on the i-th qubit
    """
    if i > n - 1:
        raise ValueError('Qubit index too high for given n in sigma_i')
    if i < n - 1:
        out = sparse.eye(2**(n-1-i), format='csc', dtype=int)
        out = sparse.kron(out, sigma)
        out = sparse.kron(out, sparse.eye(2**i, format='csc', dtype=int))
    else:
        # when i = n - 1
        out = sparse.kron(sigma, sparse.eye(2**(n-1), format='csc', dtype=int))
    return out


def hypercube_sparse(n, X):
    """Returns the hypercube (transverse field) Hamiltonian A
    as a sparse matrix.
    A = - \sum_{i=0}^{n-1} sigma_i^x
    """
    A = sigma_i_sparse(X, 0, n)

    for i in range(1, n):
        A += sigma_i_sparse(X, i, n)
    
    return A


def driver_hamiltonian_transverse_field(n, X):
    """Returns the transverse field driver Hamiltonian."""
    A = -1 * hypercube_sparse(n, X)
    return A


def hamiltonian(t, schedule, H_driver, H_problem):
    """Returns the total Hamiltonian."""
    B, A = schedule._at_t(t)
    return A*H_driver + B*H_problem


def schrodinger(t, psi, schedule, H_driver, H_problem):
    """Returns the dH/dt according to the Schrodinger equation."""
    return -1j * hamiltonian(t, schedule, H_driver, H_problem).dot(psi)


def aqc_success_prob(n, T, H_driver, H_problem, sched = None, integrator_steps=10000000, psi0=None):
    """Simulates AQC and returns the final success probability.

    Args:
        n: number of qubits
        T: total time of sweep
        H_driver: driver Hamiltonian
        H_problem: problem Hamiltonian
        integrator_steps (optional): max number of integrator timesteps
            (defaults to 10,000,000)
        psi0 (optional): initial state

    Returns:
        float: success probability
        bool: whether the integrator was successful
    """
    N = 2**n
    if psi0 is None:
        psi0 = np.ones(N) * (1 / np.sqrt(N))

    if (sched == None):
        sched = linear_schedule(T)

    schro = lambda t, y: schrodinger(t, y, sched, H_driver, H_problem)
    r = complex_ode(schro)
    r.set_integrator("dop853", nsteps=integrator_steps)
    r.set_initial_value(psi0, 0)
    # r.set_f_params(T, H_driver, H_problem)
    psiN = r.integrate(T)

    # this only works if |00...0> is the optimal solution
    success_prob = np.real(np.conj(psiN[0]) * psiN[0])

    return success_prob, r.successful()


def aqc_find_T(n, Hd, Hp, T_start=10.0, target_prob=0.99, \
tolfrac=0.01, Tmin=0.1, Tmax=10000.0, verbose=False):
    '''Finds the time required to achieve a success probability of
    target_prob.'''
    old_T = 0.0
    T = T_start
    p, success = aqc_success_prob(n, T, Hd, Hp)
    old_p = -float('inf')
    if verbose: print(f'Ran with T={T}, found p={p}')
    if p > target_prob:
        while p > target_prob:
            old_T = T
            T = T*0.5
            if T < Tmin or T > Tmax:
                return None, False
            old_p = p
            p, s = aqc_success_prob(n, T, Hd, Hp)
            if success: success = s
            if verbose: print(f'Ran with T={T}, found p={p}')
        Tlower = T
        Tupper = old_T
        plower = p
        pupper = old_p
        T = Tupper
    else:
        while p < target_prob:
            old_T = T
            T = T*2.0
            if T < Tmin or T > Tmax:
                return None, False
            old_p = p
            p, s = aqc_success_prob(n, T, Hd, Hp)
            if success: success = s
            if verbose: print(f'Ran with T={T}, found p={p}')
        Tlower = old_T
        Tupper = T
        plower = old_p
        pupper = p
    if verbose: print(f'Found range Tlower={Tlower}, Tupper={Tupper}')
    while Tupper - Tlower > 0.5*(Tupper + Tlower)*tolfrac:
        T = 0.5*(Tupper + Tlower)
        if T < Tmin or T > Tmax:
            return None, False
        p, s = aqc_success_prob(n, T, Hd, Hp)
        if success: success = s
        if verbose: print(f'Ran with T={T}, found p={p}')
        if p < target_prob:
            plower, pupper = p, pupper
            Tlower, Tupper = T, Tupper
        elif p >= target_prob:
            plower, pupper = plower, p
            Tlower, Tupper = Tlower, T
    if verbose: print('Done')
    return T, success


def get_2sat_formula(instance_name):
    """loads an instance"""
    out = np.loadtxt("./instances_typical/" + instance_name + ".m2s")
    return out.astype(int)


def get_instances(file='./instances_typical.csv'):
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt(file, delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


def hamiltonian_2sat_sparse(n, formula, sigma_z):
    N = 2 ** n
    out = sparse.csc_matrix((N, N))
    sigma_identity = sparse.eye(N, format='csc')
    for clause in formula:
        v_1 = clause[1]
        v_2 = clause[3]
        sign_1 = -1 * clause[0]                 # -1 because signs should be opposite in Hamiltonian
        sign_2 = -1 * clause[2]
        '''below we use .multiply(), which is elementwise multiplication, rather than .dot(), which is matrix
        multiplication, becasue the matrices in the problem Hamiltonian are diagonal so the end result is the same for
        both types of multiplication, even though .dot() is technically the correct type.'''
        out += (1/4) * (sign_1*sign_2*sigma_i_sparse(sigma_z, v_1, n).multiply(sigma_i_sparse(sigma_z, v_2, n))
                       + sign_1*sigma_i_sparse(sigma_z, v_1, n) + sign_2*sigma_i_sparse(sigma_z, v_2, n) + sigma_identity)
    
    return out


def run(instance_num):
    X_dense = np.array([[0, 1],
                        [1, 0]])
    Z_dense = np.array([[1, 0],
                        [0, -1]])
    X_sparse = sparse.csc_matrix(X_dense)
    Z_sparse = sparse.csc_matrix(Z_dense)

    instance_names, instance_n_bits = get_instances()

    instance_name = instance_names[instance_num]
    sat_formula = get_2sat_formula(instance_name)
    n = instance_n_bits[instance_num]
    print("n:", n)

    N = 2 ** n
    H_driver = driver_hamiltonian_transverse_field(n, X_sparse)
    H_problem = hamiltonian_2sat_sparse(n, sat_formula, Z_sparse)

    out = aqc_find_T(n, H_driver, H_problem, T_start=10.0, target_prob=0.99, tolfrac=0.01, Tmin=0.1, Tmax=10000.0, verbose=True)
    return instance_num, instance_name, out[0], out[1]


# function to find success probability for T=x
def p_Tx(instance, n_bits, x, schedule):
    X_dense = np.array([[0, 1],
                        [1, 0]])
    Z_dense = np.array([[1, 0],
                        [0, -1]])
    X_sparse = sparse.csc_matrix(X_dense)
    Z_sparse = sparse.csc_matrix(Z_dense)

    sat_formula = get_2sat_formula(instance)
    n = n_bits

    N = 2 ** n
    H_driver = driver_hamiltonian_transverse_field(n, X_sparse)
    H_problem = hamiltonian_2sat_sparse(n, sat_formula, Z_sparse)

    p, success = aqc_success_prob(n, x, H_driver, H_problem, schedule)
    return instance, p, success


# def p_Tx_wrapper(x):
#     instance_names, instance_n_bits = get_instances()
#     for i in range(len(instance_names)):
#         if instance_n_bits[i] < 20:
#             continue
#         else:
#             instance_num, instance_name, p, success = p_Tx(i, x)
#             with open("test.csv", "a") as output:
#                 output.write(str(instance_name)+','+str(p)+','+str(success))

    
# function to store states while running
def aqc_success_withStates(n, T, H_driver, H_problem, integrator_steps=10000000, psi0=None):
    N = 2**n
    if psi0 is None:
        psi0 = np.ones(N) * (1 / np.sqrt(N))

    schro = lambda t, y: schrodinger(t, y, T, H_driver, H_problem)
    r = complex_ode(schro)
    r.set_integrator("dop853", nsteps=integrator_steps)
    r.set_initial_value(psi0, 0)

    psi = [psi0]
    dt = 1
    while r.successful() and r.t < T:
        # r.integrate(T, step=True)
        
        r.t+dt
        r.integrate(r.t+dt)
        psi.append(r.y)
    psi = np.array(psi)
    return psi

def create_anim_state(instance_num, T):
    X_dense = np.array([[0, 1],
                        [1, 0]])
    Z_dense = np.array([[1, 0],
                        [0, -1]])
    X_sparse = sparse.csc_matrix(X_dense)
    Z_sparse = sparse.csc_matrix(Z_dense)

    instance_names, instance_n_bits = get_instances()

    instance_name = instance_names[instance_num]
    sat_formula = get_2sat_formula(instance_name)
    n = instance_n_bits[instance_num]
    print("n:", n)

    N = 2 ** n
    H_driver = driver_hamiltonian_transverse_field(n, X_sparse)
    H_problem = hamiltonian_2sat_sparse(n, sat_formula, Z_sparse)

    psi = aqc_success_withStates(n, T, H_driver, H_problem)

    return instance_num, instance_name, psi

import concurrent.futures

def parallel_run(instance_num):
    try:
        result = run(instance_num)
        return result
    except Exception as e:
        print(f"Error in instance {instance_num}: {e}")
        return None

if __name__ == '__main__':
    instance_names, _ = get_instances()
    num_instances = len(instance_names)

    num_cores = 8

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(parallel_run, range(num_instances)))

    # Save the results to a CSV file
    with open("results.csv", "a") as output:
        output.write("Instance Name, T_99, Success\n")
        for result in results:
            if result is not None:
                instance_name, t_99, success = result
                output.write(f"{instance_name},{t_99},{success}\n")

# if __name__ == '__main__':
#     # test run with first instance
#     instance_num, instance_name, t_99, success = run(0)

#     with open("test.csv", "w") as output:
#         output.write(str(instance_name)+','+str(t_99)+','+str(success))
