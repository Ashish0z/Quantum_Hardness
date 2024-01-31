import os
from argparse import ArgumentParser
import numpy as np

from qutip import (Qobj, tensor, qeye, sigmax, sigmaz, mesolve, Options, qsave)


def sigma_j(j, N, sigmaxyz):
    """
    Return a Qobj for the sigmax, sigmay, or sigmaz operator for the jth qubit
    in a N-qubit system.
    """
    if j == 0:
        out = tensor(tensor([qeye(2)]*(N-1)), sigmaxyz)
    elif j == N-1:
        out = tensor(sigmaxyz, tensor([qeye(2)]*(N-1)))
    else:
        out = tensor(tensor([qeye(2)]*(N-1-j)), sigmaxyz)
        out = tensor(out, tensor([qeye(2)]*(j)))
    return out


def load_formula(instancce_name, base_path = '../MAX2SATQuantumData-r2x346d419f-version1/MAX2SATQuantumData/Crosson_DifferentStrategies/instances_crosson'):
    out = np.loadtxt(os.path.join(base_path, instancce_name)).astype(int)
    return out


def get_problem_Hamiltonian(sat_formula, N):
    H_p = 0
    for clause in sat_formula:
        v_1 = clause[1]
        v_2 = clause[3]
        sign_1 = -1 * clause[0]
        sign_2 = -1 * clause[2]
        k = (1/4) * (sign_1*sign_2*sigma_j(v_1, N, sigmaz())*sigma_j(v_2, N, sigmaz()) + sign_1*sigma_j(v_1, N, sigmaz()) + sign_2*sigma_j(v_2, N, sigmaz()) + sigma_j(0, N, qeye(2)))
        H_p += k
    return H_p

def get_hamiltonina(H0, H_p, schedule):
    if schedule == 'linear':
        h_t = [
            [H0, lambda t, args: (args["t_max"] - t) / args["t_max"]],
            [H_p, lambda t, args: t / args["t_max"]],
        ]
    elif schedule == 'quadratic':
        h_t = [
            [H0, lambda t, args: 1 - (t / args["t_max"])**2 * (3 - 2 * t / args["t_max"])],
            [H_p, lambda t, args: (t / args["t_max"])**2 * (3 - 2 * t / args["t_max"])],
        ]
    elif schedule == 'cubic':
        h_t = [
            [H0, lambda t, args: 1 - (t / args["t_max"])**3 * (10 - 15 * t / args["t_max"] + 6 * (t / args["t_max"])**2)],
            [H_p, lambda t, args: (t / args["t_max"])**3 * (10 - 15 * t / args["t_max"] + 6 * (t / args["t_max"])**2)],
        ]

    elif schedule == 'biquadratic':
        h_t = [
            [H0, lambda t, args: 1 - (t / args["t_max"])**4 * (35 - 84 * t / args["t_max"] + 70 * (t / args["t_max"])**2 - 20 * (t / args["t_max"])**3)],
            [H_p, lambda t, args: (t / args["t_max"])**4 * (35 - 84 * t / args["t_max"] + 70 * (t / args["t_max"])**2 - 20 * (t / args["t_max"])**3)],
        ]

    elif schedule == 'cosine':
        h_t = [
            [H0, lambda t, args: 1 - (0.5*(1 - np.cos(np.pi * (t / args["t_max"])**2)))],
            [H_p, lambda t, args: 0.5*(1 - np.cos(np.pi * (t / args["t_max"])**2))],
        ]
    
    return h_t

def run(instance, instace_base, N, taulist, schedule = 'linear', save_path = 'results'):
    psi0 = np.ones(2 ** N) / np.sqrt(2 ** N)
    psi0 = Qobj(psi0.reshape(2**N, 1), dims=[[2]*N, [1]*N])

    H0 = 0
    for n in range(N):
        H0 += sigma_j(n, N, sigmax())
    H0 = -H0
    
    H_p = get_problem_Hamiltonian(load_formula(instance, instace_base), N)
    args = {"t_max": max(taulist)}
    h_t = get_hamiltonina(H0, H_p, schedule)
    result = mesolve(h_t, psi0, taulist, args=args, options = Options(num_cpus=4), progress_bar=True)

    qsave(data=result, name=f'{save_path}/{instance[:-4]}_{schedule}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--instance_list', choices=['crosson', 'typical'], default='typical')
    parser.add_argument('--instance_index', type=int, default=3)
    parser.add_argument('--N_bits', type=int, default=5)
    parser.add_argument('--schedule', choices=['linear', 'quadratic', 'cubic', 'biquadratic', 'cosine'], default='linear')
    parser.add_argument('--taumax', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='results')
    args = parser.parse_args()


    if args.instance_list == 'crosson':
        base_path = '../MAX2SATQuantumData-r2x346d419f-version1/MAX2SATQuantumData/Crosson_DifferentStrategies/instances_crosson'
        instance_list = os.listdir(base_path)

    elif args.instance_list == 'typical':
        base_path = '../MAX2SATQuantumData-r2x346d419f-version1/MAX2SATQuantumData/Mirkarimi_ComparingHardness/instances_typical'
        instance_list = os.listdir(base_path)

    N = args.N_bits

    taumax = args.taumax
    taulist = np.linspace(0, taumax, 100)

    instance = instance_list[args.instance_index]
    print(f'Running {instance} with {N} bits and {args.schedule} schedule')
    run(instance, base_path, N, taulist, args.schedule, args.save_path)


if __name__ == '__main__':
    main()
