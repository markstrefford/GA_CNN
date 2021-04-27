from deap import tools
from deap import algorithms
import random
import pickle
import numpy as np
import os


def checkpoint(log_dir, population, gen, halloffame, logbook):
    print('*** Checkpoint ***')
    # Fill the dictionary using the dict(key=value[, ...]) constructor
    cp = dict(population=population, generation=gen, halloffame=halloffame,
              logbook=logbook, rndstate=random.getstate())

    with open(os.path.join(log_dir, f'checkpoint_{gen}.pkl'), 'wb') as cp_file:
        pickle.dump(cp, cp_file)


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, log_dir='', verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    # global N_EVALS, N_GENS

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        if gen % 100 == 0:
            checkpoint(log_dir, population, gen, halloffame, logbook)

        # From https://stackoverflow.com/questions/58990269/deap-make-mutation-probability-depend-on-generation-number
        # N_EVALS += 1
        # if N_EVALS % 100 == 0:
        #     N_GENS += 1
        #     # TODO: Can we just say N_GEN = ngen here?

    return population, logbook
