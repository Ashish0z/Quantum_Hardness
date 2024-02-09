import concurrent.futures
import logging
import os
import threading
# import multiprocessing
import time
import numpy as np

from aqc_sim import *
from reduced_rho import *
from utils import *
from schedules import *

# manager = multiprocessing.Manager()
# process_step_counts = {} # manager.dict()
# process_lock = threading.Lock()

# stop_progress_thread = threading.Event()

# total_steps = 0

# def update_overall_progress():
#     global process_step_counts
#     while not stop_progress_thread.is_set():
#         try:
#             total_completed_steps = sum(process_step_counts.values())
#             logging.info("Overall Progress: %s/%s steps completed", total_completed_steps, total_steps)
#             time.sleep(3)  # Update the overall progress every 5 min
#         except Exception as e: 
#             print("Error while updating overall progress: " + str(e))

def bit20_sim(instance, n, schedules):
    process_id = os.getpid()
    process_log_file = f'./Data_20bit/logs/instance_{instance}_progress.log'
    
    # Configure the individual process logging
    process_logger = logging.getLogger(f'process_{process_id}')
    process_logger.setLevel(logging.INFO)
    process_handler = logging.FileHandler(process_log_file, mode = 'w')
    process_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    process_logger.addHandler(process_handler)
    process_logger.propagate = False

    process_logger.info("Initializing process %s with instance %s", process_id, instance)
    
    N = 2**n

    psi0 = np.ones(N)/np.sqrt(N)
    X_dense = np.array([[0, 1],
                        [1, 0]])
    Z_dense = np.array([[1, 0],
                        [0, -1]])
    X_sparse = sparse.csc_matrix(X_dense)
    Z_sparse = sparse.csc_matrix(Z_dense)
    sat_formula = get_2sat_formula(instance)

    H_driver = driver_hamiltonian_transverse_field(n, X_sparse)
    H_problem = hamiltonian_2sat_sparse(n, sat_formula, Z_sparse)

    # with process_lock:
    #     process_step_counts[process_id] = 0

    for sch in schedules:
        process_logger.info("Starting simulation for schedule %s", str(sch))
        psi = [psi0]
        schro = lambda t, y: schrodinger(t, y, sch, H_driver, H_problem)
        r = complex_ode(schro)
        r.set_integrator("dop853", nsteps=10000000)
        # if psi file exists, load it
        if os.path.isfile(f'./Data_20bit/data/instance_{instance}_schedule_{str(sch)}_psi.npz'):
            process_logger.info("Loading psi from file")
            psi = list(np.load(f'./Data_20bit/data/instance_{instance}_schedule_{str(sch)}_psi.npz')['psi'])
            r.set_initial_value(psi[-1], len(psi)-1)
        else:
            r.set_initial_value(psi0, 0)
        while r.successful() and r.t < sch.T:
            r.integrate(r.t+1)
            psi.append(r.y)
            
            np.savez(f'./Data_20bit/data/instance_{instance}_schedule_{str(sch)}_psi.npz', psi=np.array(psi))
            process_logger.info("Simulation for schedule %s is at time %s", str(sch), r.t)
            # with process_lock:
            #     process_step_counts[process_id] = r.t + 100*schedules.index(sch)
        process_logger.info("Simulation for schedule %s completed", str(sch))

        # save psi to file
        np.savez(f'./Data_20bit/data/instance_{instance}_schedule_{str(sch)}_psi.npz', psi=np.array(psi))

    process_logger.info("Simulation for instance %s completed", instance)
    process_handler.close()

def sim_run(instance, n, schedules):
    try:
        bit20_sim(instance, n, schedules)
    except Exception as e:
        print("Error running instance " + instance + ": " + str(e))

if __name__ == "__main__":
    # Set up the problem parameters
    n = 20
    T = 100
    schedules = [linear_schedule(T), quadratic_schedule(T), cubic_schedule(T), biquadratic_schedule(T), cosine_schedule(T)]
    # if os.isfile('selected_instances.csv'):
    #     problems = np.loadtxt('selected_instances.csv', delimiter=',', dtype=str)
    
    # else :
    #     instances, n_bit = get_instances()
    #     problems = get_n_bit_random_instances(instances, n_bit, 1000)
    #     np.savetxt('selected_instances.csv', problems, delimiter=',', fmt='%s')
    
    # problems = [problems[1], problems[-2], problems[2], problems[-3], problems[3], problems[-4], problems[4], problems[-5]]
    problems = ['yjavqxgcrtqnretvdkqdimcoepgysd', 'soqchljihtjmrxxqneiaiocpkdutbm']

    total_steps = len(problems) * len(schedules) * T

    # Configure the overall logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler('./Data_20bit/logs/overall_progress.log', mode='w'), logging.StreamHandler()])

    # Start the overall progress update thread
    # overall_progress_thread = threading.Thread(target=update_overall_progress)
    # overall_progress_thread.daemon = True
    # overall_progress_thread.start()

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = []

        for problem in problems:
            future = executor.submit(sim_run, problem, n, schedules)
            futures.append(future)

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
        # stop_progress_thread.set()

