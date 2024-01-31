import os
import time
from argparse import ArgumentParser
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from concurrent.futures import ProcessPoolExecutor

from qutip import entropy_vn, qload


def entropy_vs_time(states, n):
    en_list = []
    for state in states:
        rho = state.ptrace([i for i in range(n//2)])
        en_list.append(entropy_vn(rho, base=2))

    return en_list


def save_entropy_plot(en_list, n, schedule, out_path):
    fig, ax = plt.subplots()
    ax.plot(en_list)
    ax.set_ylim(bottom=0, top=max(1.4, max(en_list)))
    ax.set_xlabel('Time')
    ax.set_ylabel('Entropy')
    ax.set_title(f'Entropy vs. time for {n} qubits, {schedule} schedule')
    fig.savefig(out_path)
    plt.close(fig)


def generate_random_bipartition(n):
    perm = np.random.permutation(n)
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


def calculate_spacing_ratios(state, partitions):
    spacing_ratios = []
    for A in partitions:
        rho_A = state.ptrace(A)

        eigs = rho_A.eigenenergies()

        eigs = eigs[np.abs(eigs) > 1e-12]
        eigs = np.sort(eigs)[::-1]

        diffs = -np.diff(eigs)
        diffs = diffs[np.abs(diffs)>1e-12]

        spacing_ratios.extend(diffs[:-1] / diffs[1:])

    return spacing_ratios


def plot_histogram(spacing_ratios, n, schedule, out_path, width = 0.1):
    bins = np.arange(np.floor(min(spacing_ratios).real), np.ceil(max(spacing_ratios).real), width)
    hist, bins = np.histogram(spacing_ratios, bins=bins, density=True)

    fig, ax = plt.subplots()
    ax.plot(bins[:-1], hist, 'o', markersize=3)
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=4)
    ax.set_xlabel('r')
    ax.set_ylabel('P(r)')

    ax.set_title(f'Spacing ratios for {n} qubits, {schedule} schedule')
    fig.savefig(out_path)
    plt.close(fig)


def calc_instance(instance, results_dir, out_dir, n, p):
    logger = logging.getLogger('qutip_process_res')
    
    partitions = get_unique_bipartitions(n, p)

    en = []
    rt = []
    succ = []

    for schedule in ['linear', 'quadratic', 'cubic', 'biquadratic', 'cosine']:
        states = qload(f'{results_dir}/{instance}_{schedule}').states
        logger.info(f'Loaded results for {instance}_{schedule}')

        en.append(entropy_vs_time(states, n))
        logger.info(f'Calculated entropy for {instance}_{schedule}')

        spacing_ratios = []
        for t in range(len(en[-1])):
            spacing_ratios.append(calculate_spacing_ratios(states[t], partitions))
        rt.append(spacing_ratios)
        logger.info(f'Calculated spacing ratios for {instance}_{schedule}')

        success_prob = states[-1][0][0][0]
        success_prob = success_prob * np.conj(success_prob)
        success_prob = success_prob.real
        succ.append(success_prob)
        del states
    
    rt_combined = []
    for t in range(len(rt[0])):
        rt_combined.append(np.concatenate([rt[i][t] for i in range(len(rt))]))

    logger.info(f'Plotting {instance}')

    if os.path.exists(f'{out_dir}/{instance}'):
        os.rmdir(f'{out_dir}/{instance}')
        
    os.mkdir(f'{out_dir}/{instance}')
    os.mkdir(f'{out_dir}/{instance}/data')
    os.mkdir(f'{out_dir}/{instance}/plots')

    np.savez(f'{out_dir}/{instance}/data/entropy.npz', linear=en[0], quadratic=en[1], cubic=en[2], biquadratic=en[3], cosine=en[4])
    np.savez(f'{out_dir}/{instance}/data/spacing_ratios.npz', linear=rt[0], quadratic=rt[1], cubic=rt[2], biquadratic=rt[3], cosine=rt[4], combined=rt_combined)

    with open(f'{out_dir}/{instance}/data/partitions.txt', 'w') as f:
        for A in partitions:
            f.write(f'{A}\n')

    with open(f'{out_dir}/final_results.csv', 'a') as f:
        f.write(f'{instance},{succ[0]},{succ[1]},{succ[2]},{succ[3]},{succ[4]}\n')

    logger.info(f'Saved data for {instance}')


def plot_instance(instance, out_dir, n, hist_width):
    logger = logging.getLogger('qutip_process_res')

    logger.info(f'Plotting {instance}')
    entropies = np.load(f'{out_dir}/{instance}/data/entropy.npz')

    fig, ax = plt.subplots()
    for schedule in ['linear', 'quadratic', 'cubic', 'biquadratic', 'cosine']:
        ax.plot(entropies[schedule], label=schedule)
    ax.set_ylim(bottom=0, top=max(1.4, max(entropies['linear'])))
    ax.set_xlabel('Time')
    ax.set_ylabel('Entropy')
    ax.set_title(f'Entropy vs. time for {n} qubits instance {instance}')
    ax.legend()
    fig.savefig(f'{out_dir}/{instance}/plots/entropy.png')
    plt.close(fig)

    spacing_ratios = np.load(f'{out_dir}/{instance}/data/spacing_ratios.npz')

    for schedule in ['linear', 'quadratic', 'cubic', 'biquadratic', 'cosine']:
        rt = spacing_ratios[schedule]

        # bins = np.arange(np.floor(min(rt).real), np.ceil(max(rt).real), hist_width)
        # hist, bins = np.histogram(rt, bins=bins, density=True)

        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'o', markersize=3)
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=0, right=4)
        ax.set_xlabel('r')
        ax.set_ylabel('P(r)')

        def init():
            line.set_data([], [])
            return line,

        def update(i):
            bins = np.arange(np.floor(min(rt[i]).real), np.ceil(max(rt[i]).real), hist_width)
            hist, bins = np.histogram(rt[i], bins=bins, density=True)
            line.set_data(bins[:-1], hist)
            ax.set_title(f'Spacing ratios for {n} qubits instance {instance}, {schedule} schedule, t={i}')
            return line,

        anim = FuncAnimation(fig, update, init_func=init, frames=len(rt), blit=True)
        anim.save(f'{out_dir}/{instance}/plots/rt_{schedule}.mp4', writer='ffmpeg', fps=3)
        plt.close(fig)

    rt = spacing_ratios['combined']

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o', markersize=3)
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=4)
    ax.set_xlabel('r')
    ax.set_ylabel('P(r)')
    def init():
        line.set_data([], [])
        return line,
    def update(i):
        bins = np.arange(np.floor(min(rt[i]).real), np.ceil(max(rt[i]).real), hist_width)
        hist, bins = np.histogram(rt[i], bins=bins, density=True)
        line.set_data(bins[:-1], hist)
        ax.set_title(f'Spacing ratios for {n} qubits instance {instance}, combined, t={i}')
        return line,
    anim = FuncAnimation(fig, update, init_func=init, frames=len(rt), blit=True)
    anim.save(f'{out_dir}/{instance}/plots/rt_combined.mp4', writer='ffmpeg', fps=3)
    plt.close(fig)


def plot_instance_old(instance, results_dir, out_dir, n, t, p, hist_width):
    logger = logging.getLogger('qutip_process_res')

    partitions = get_unique_bipartitions(n, p)

    ens = {}
    rts = {}
    p_succ = {}

    logger.info(f'Plotting {instance}')
    for schedule in ['linear', 'quadratic', 'cubic', 'biquadratic', 'cosine']:
        states = qload(f'{results_dir}/{instance}_{schedule}').states
        logger.info(f'Loaded results for {instance}_{schedule}')

        en = entropy_vs_time(states, n)
        ens[schedule] = en
        # save_entropy_plot(en, n, schedule, f'{out_dir}/{instance}_{schedule}_entropy.png')
        logger.info(f'Calculated entropy plot for {instance}_{schedule}')

        if t == -1:
            t = np.argmax(en)

        spacing_ratios = calculate_spacing_ratios(states[t], partitions)
        rts[schedule] = spacing_ratios
        # plot_histogram(spacing_ratios, n, schedule, f'{out_dir}/{instance}_{schedule}_hist.png', width=hist_width)
        logger.info(f'Calculated spacing ratios for {instance}_{schedule}')

        success_prob = states[-1][0][0][0]
        success_prob = success_prob * np.conj(success_prob)
        success_prob = success_prob.real
        p_succ[schedule] = success_prob

        del states

    fig, ax = plt.subplots()
    for schedule in ['linear', 'quadratic', 'cubic', 'biquadratic', 'cosine']:
        ax.plot(ens[schedule], label=schedule)
    ax.set_ylim(bottom=0, top=max(1.4, max(ens['linear'])))
    ax.set_xlabel('Time')
    ax.set_ylabel('Entropy')
    ax.set_title(f'Entropy vs. time for {n} qubits instance {instance}')
    ax.legend()
    fig.savefig(f'{out_dir}/{instance}_entropy.png')
    plt.close(fig)

    # for schedule in ['linear', 'quadratic', 'cubic', 'biquadratic', 'cosine']:
    #     rt = rts[schedule]

    #     bins = np.arange(np.floor(min(rt).real), np.ceil(max(rt).real), hist_width)
    #     hist, bins = np.histogram(rt, bins=bins, density=True)

    #     fig, ax = plt.subplots()
    #     ax.plot(bins[:-1], hist, 'o', markersize=3)
    #     ax.set_ylim(bottom=0, top=1)
    #     ax.set_xlim(left=0, right=4)
    #     ax.set_xlabel('r')
    #     ax.set_ylabel('P(r)')
    #     ax.set_title(f'Spacing ratios for {n} qubits instance {instance}, {schedule} schedule, t={t}')
    #     fig.savefig(f'{out_dir}/{instance}_{schedule}_hist.png')
    #     plt.close(fig)

    rt = rts['linear'] + rts['quadratic'] + rts['cubic'] + rts['biquadratic'] + rts['cosine']
    bins = np.arange(np.floor(min(rt).real), np.ceil(max(rt).real), hist_width)
    hist, bins = np.histogram(rt, bins=bins, density=True)

    fig, ax = plt.subplots()
    ax.plot(bins[:-1], hist, 'o', markersize=3)
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=4)
    ax.set_xlabel('r')
    ax.set_ylabel('P(r)')
    ax.set_title(f'Spacing ratios for {n} qubits instance {instance}, combined, t={t}')
    fig.savefig(f'{out_dir}/{instance}_hist.png')
    plt.close(fig)

    with open(f'{out_dir}/final_results.csv', 'a') as f:
        f.write(f'{instance},{p_succ["linear"]},{p_succ["quadratic"]},{p_succ["cubic"]},{p_succ["biquadratic"]},{p_succ["cosine"]}\n')

    logger.info(f'Finished {instance}_{schedule}')

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger('qutip_process_res')

    parser = ArgumentParser()
    parser.add_argument('--results_dir', '-r', help='Directory containing results')
    parser.add_argument('--out_dir', '-o', help='Directory to save plots to')
    parser.add_argument('-n', type=int, help='Number of qubits', default=20)
    parser.add_argument('-t', type=int, help='Time at which to plot ratio', default=50)
    parser.add_argument('-p', type=int, help='Number of partitions to use', default=100)
    parser.add_argument('--hist_width', type=float, help='Width of histogram bins', default=0.1)
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir
    n = args.n
    t = args.t
    p = args.p
    hist_width = args.hist_width
    
    results_list = os.listdir(results_dir)
    instances = [result.split('_')[0] for result in results_list]
    instances = list(set(instances))


    logger.info(f'Found {len(instances)} instances')

    for instance in instances:
        plot_instance_old(instance, results_dir, out_dir, n, t, p, hist_width)
    #     calc_instance(instance, results_dir, out_dir, n, p)
    #     plot_instance(instance, out_dir, n, hist_width)

    # plot_instance_old(instances[0], results_dir, out_dir, n, t, p, hist_width)

    # with ProcessPoolExecutor(max_workers=2) as executor:
    #     # executor.map(plot_instance, instances, [results_dir]*len(instances), [out_dir]*len(instances), [n]*len(instances), [t]*len(instances), [p]*len(instances), [hist_width]*len(instances))
    #     executor.map(plot_instance_old, instances, [results_dir]*len(instances), [out_dir]*len(instances), [n]*len(instances), [t]*len(instances), [p]*len(instances), [hist_width]*len(instances))
    # for instance in instances:
    #     for schedule in ['linear', 'quadratic', 'cubic', 'biquadratic', 'cosine']:
    #         results = qload(f'{results_dir}/{instance}_{schedule}')
    #         logging.info(f'Loaded results for {instance}_{schedule}')

    #         states = results.states

    #         en = entropy_vs_time(states, n)
    #         save_entropy_plot(en, n, schedule, f'{out_dir}/{instance}_{schedule}_entropy.png')
    #         logging.info(f'Saved entropy plot for {instance}_{schedule}')

    #         if t == -1:
    #             t = np.argmax(en)

    #         spacing_ratios = calculate_spacing_ratios(states[t], n, num_partitions=p)
    #         plot_histogram(spacing_ratios, n, schedule, f'{out_dir}/{instance}_{schedule}_hist.png', width=hist_width)

    #         logging.info(f'Finished {instance}_{schedule}')


if __name__ == '__main__':
    main()
