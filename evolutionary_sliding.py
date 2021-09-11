from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import multiprocessing
import json 

from optimization_utils import DataLoader, SimulationModel, compute_mse
from optimization_utils import init_data, init_simulation, init_data_and_simulation, init_train_test_data

from pathlib import Path
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def init_toolbox(stepsize=5):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimize the fitness function value
    creator.create("Individual", list, fitness=creator.FitnessMin) # Individual is a list of parameter values
    toolbox = base.Toolbox()
    lb_tsum1, ub_tsum1 = 150, 1050
    lb_tsum2, ub_tsum2 = 600, 1550
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
    toolbox.register('population_guess',
	                 init_seed_population,
	                 list,
	                 creator.Individual,
	                 'seed_population.json')
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
    # print(f'MSE: {mse} | TSUM1: {tsum1} TSUM2: {tsum2}')
    return mse,

def evaluate_test(individual, sim, dl, df):
    tsum1, tsum2 = individual[0], individual[1]
    mse =tsum1 + tsum2
    print(f'MSE: {mse} | TSUM1: {tsum1} TSUM2: {tsum2}')
    return mse,

def update_toolbox(toolbox, sim, dl, df, test=False):
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=[0, 0], sigma=[10, 10], indpb=.5)
    # toolbox.register("select", tools.selBest) #k=len(population)
    toolbox.register("select", tools.selTournament, tournsize=2) # k=len(population)
    if not test:
        toolbox.register("evaluate", evaluate, sim=sim, dl=dl, df=df)
    else:
        toolbox.register("evaluate", evaluate_test, sim=sim, dl=dl, df=df)
    return toolbox

def init_seed_population(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)


def eaSimpleEarlyStopping(population, toolbox, cxpb, mutpb, ngen, stats=None,
                          halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring
    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    .. note::
        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # survivors = toolbox.select(population, len(population)-2)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(survivors, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        # population = offspring + survivors

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}

        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if record['max'] - record['min'] < 0.1:
            print(f'Early stopping!')
            break

        if record['max'] < 0.5:
            print(f'Early stopping!')
            break

        if record['min'] < 0.2:
            print(f'Early stopping!')
            break

    return population, logbook

def run_ea_algorithm(toolbox, verbose=False):
    population_size = 4 # + 6 individuals from population_guess
    crossover_probability = 0.7
    mutation_probability = 0.2
    number_of_generations = 8
    if verbose:
	    print('-----------------------------------')
	    print(f'EA Initialization:\n'
	          f'{population_size=}\n' 
	          f'{crossover_probability=}\n'
	          f'{mutation_probability=}\n'
	          f'{number_of_generations=}')
	    print('-----------------------------------')
          # f'population_size = {population_size = }\n', 

    pop = toolbox.population(n=population_size) + toolbox.population_guess()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, stats = stats, 
    #                                mutpb = mutation_probability, ngen=number_of_generations, halloffame=hof, 
    #                                verbose=True) 
    # Use the custom function below for early stopping
    pop, log = eaSimpleEarlyStopping(pop, 
    								 toolbox, 
    								 cxpb=crossover_probability,
    								 stats=stats,
    								 mutpb=mutation_probability,
    								 ngen=number_of_generations,
    								 halloffame=hof, 
                               		 verbose=True) 
    best_parameters = hof[0] # Save the optimal set of parameters
    return pop, log, best_parameters


parser = argparse.ArgumentParser(description='')
parser.add_argument('--parallel', dest='parallel', action='store_true', help='Activate parallel mode.')
parser.add_argument('--test', dest='test', action='store_true', help='Activate test mode.')
parser.add_argument('--interval', type=int, default=4, help='Interval of years to use for training')

args = parser.parse_args()

interval = args.interval
crop_name = 'maize'
model_name = 'pp'
exp_date = 'apr28'

toolbox, creator = init_toolbox(stepsize=5)
tmp_dl = DataLoader()
years = sorted(tmp_dl.df['Year'].unique())
if args.test:
    years = years[-1:]
sim = init_simulation(crop_name, model_name)
# print(f'Yield data available for years: {years}')

if __name__ == '__main__':
    big_s = time.time()
    results = [] # List of results to create DataFrame later
    for year in years[interval:-interval]:
        dl = init_train_test_data(tmp_dl, year, interval)
        if not dl:
            print(f'Not running EA for year: {year} due to missing data!')
            continue
        toolbox = update_toolbox(toolbox, sim, dl, dl.train_df, test=args.test)
        
        print(f'Running EA for year {year}...')
        s = time.time()
        if args.parallel:
            CPU_COUNT = multiprocessing.cpu_count() - 2
            print(f'Running {CPU_COUNT} processes in parallel.')
            pool = multiprocessing.Pool(processes=CPU_COUNT)
            toolbox.register("map", pool.map)
            pop, log, best_parameters = run_ea_algorithm(toolbox)
            pool.close()
        else:
            pop, log, best_parameters = run_ea_algorithm(toolbox)
        e = time.time()
        print(f'EA completed for year {year}. Runtime: {e-s}. ',
              f'Best parameters: {best_parameters}.')
        
        tsum1, tsum2 = round(best_parameters[0]), round(best_parameters[1])
        x = [tsum1, tsum2]

        fname = f'calibrated_sliding/{exp_date}/{crop_name}/{model_name}/interval{interval}/train/{tsum1}_{tsum2}_{year-interval}-{year-1}.csv'
        Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
        mse = compute_mse(x, sim, dl, dl.train_df, fname=fname, save=True)
        train_years = sorted(dl.train_df['Year'].unique())
        print(f'Years: {train_years} | TSUM1: {tsum1} | TSUM2: {tsum2}| Train MSE: {mse}')
        year_results = x + [train_years] + [mse]

        year_results.append(year)
        fname = f'calibrated_sliding/{exp_date}/{crop_name}/{model_name}/interval{interval}/test/{tsum1}_{tsum2}_{year}.csv'
        Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
        mse = compute_mse(x, sim, dl, dl.test_df, fname=fname, save=True)
        print(f'Year: {year} | TSUM1: {tsum1} | TSUM2: {tsum2}| Test MSE: {mse}')
        year_results.append(mse)

        results.append(year_results)
        print('Results so far: \n', results)
        print('---------------------------------------------------------------------------------')
    # Write results to a csv
    columns = [ 'TSUM1', 'TSUM2', 'Train Years', 'Train MSE', 'Test Year','Test MSE']
    results_summary = pd.DataFrame(results, columns=columns)
    results_fname = f'calibrated_sliding/{exp_date}/results_summary_interval{interval}.csv'
    Path(os.path.dirname(results_fname)).mkdir(parents=True, exist_ok=True)
    results_summary.to_csv(results_fname)
    print(f'Saved results summary at {results_fname}.')
    big_e = time.time()
    print(f'Overall Runtime: {big_e-big_s}')






