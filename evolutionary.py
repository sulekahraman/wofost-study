import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import multiprocessing

from optimization_utils import DataLoader, SimulationModel
from optimization_utils import init_data, init_simulation, init_data_and_simulation

import argparse
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def init_toolbox(stepsize=5):
	creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimize the fitness function value
	creator.create("Individual", list, fitness=creator.FitnessMin) # Individual is a list of parameter values
	toolbox = base.Toolbox()
	lb_tsum1, ub_tsum1 = 400, 1050 #150, 1050
	lb_tsum2, ub_tsum2 = 600, 1505
	# Define how each gene will be generated (e.g. tsum1 is a random int in the range).
	toolbox.register('attr_tsum1', random.randrange, lb_tsum1, ub_tsum1, stepsize)
	toolbox.register('attr_tsum2', random.randrange, lb_tsum2, ub_tsum2, stepsize)
	N_CYCLES = 1
	toolbox.register('individual',
	                 tools.initCycle,
	                 creator.Individual,
	                 (toolbox.attr_tsum1, toolbox.attr_tsum2),
	                 n=N_CYCLES)
	toolbox.register('population', tools.initRepeat, list, toolbox.individual)
	return toolbox, creator

def evaluate(individual, sim, dl, df, save=False):
    tsum1, tsum2 = individual[0], individual[1]
    pred_yield = np.zeros(len(df))
    true_yield = np.zeros(len(df))
    for i, (state, county, year) in enumerate(dl.get_data_points(df)):
        true_yield[i] = dl.get_true_yield(df, state, county, year)
        pred_yield[i] = sim.run_model(state, county, year, tsum1, tsum2, true_yield[i])
    if save:
        sim.save_results(f'calibration_output/genetic_algo/{tsum1}_{tsum2}.csv')
    sim.reset_results_df()
    mse = mean_squared_error(pred_yield, true_yield)
    print(f'MSE: {mse} | TSUM1: {tsum1} TSUM2: {tsum2}')
    return mse,

def evaluate_test(individual, sim, dl, df):
    tsum1, tsum2 = individual[0], individual[1]
    mse =tsum1 + tsum2
    print(f'MSE: {mse} | TSUM1: {tsum1} TSUM2: {tsum2}')
    return mse,

def update_toolbox(toolbox, sim, dl, df, test=False):
	toolbox.register("mate", tools.cxOnePoint)
	toolbox.register("mutate", tools.mutGaussian, mu=[0, 0], sigma=[10, 10], indpb=.5)
	toolbox.register("select", tools.selTournament, tournsize=2)
	if not test:
		toolbox.register("evaluate", evaluate, sim=sim, dl=dl, df=df)
	else:
		toolbox.register("evaluate", evaluate_test, sim=sim, dl=dl, df=df)
	return toolbox

def main(toolbox):
    population_size = 20
    crossover_probability = 0.7
    mutation_probability = 0.2
    number_of_generations = 13

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, stats = stats, 
                                   mutpb = mutation_probability, ngen=number_of_generations, halloffame=hof, 
                                   verbose=True) 
    print('HERE 3')
    best_parameters = hof[0] # save the optimal set of parameters
    return pop, log, best_parameters, hof, stats

parser = argparse.ArgumentParser(description='')
parser.add_argument('--parallel', dest='parallel', action='store_true', help='Activate parallel mode.')
parser.add_argument('--test', dest='test', action='store_true', help='Activate test mode.')
args = parser.parse_args()

toolbox, creator = init_toolbox(stepsize=5)
sim, dl, df = init_data_and_simulation(year=2011) # + 5 years
toolbox = update_toolbox(toolbox, sim, dl, df, test=args.test)

if __name__ == '__main__':
	s = time.time()
	if args.parallel:
		CPU_COUNT = multiprocessing.cpu_count() - 1
		print(f'Running {CPU_COUNT} processes in parallel...')
		pool = multiprocessing.Pool(processes=CPU_COUNT)
		toolbox.register("map", pool.map)
		pop, log, best_parameters, hof, stats = main(toolbox)
		pool.close()

	else:
		pop, log, best_parameters, hof, stats = main(toolbox)

	e = time.time()
	print(f'Runtime: {e-s}')
	print(f'Best parameters: {best_parameters}')
	print(f'Log: {log}')
	print(f'Population: {pop}')
	print(f'Hall of fame: {hof}')
	print(f'stats: {stats}')





