import numpy as np
from scipy.special import softmax


class GenAlgo:
    """
    GenAlgo

    It is an generic Genetic Algorithm solver class.
    It accepts the problem as an another class.
    Input class must implement the following class methods:

    - get_fitness()
    - crossover()
    - mutate()



    """
    def __init__(self, population_type, n_population=1000, mutation_rate=0.01, total_generation=5000):
        """
        __init__

        It performs the following operations:
            - It constructs the GenAlgo class
        
        Input:
            population_type = Problem class
            n_population = Number of individuals that will exist in population
            mutation_rate = Mutation rate that will be used
            total_generation = Iteration termination criteria

        Output:
            []
    
        Usage:
            gen_algo = GenAlgo(BrachCurve, n_population=1, mutation_rate=0.0001,  total_generation=1)
            

        """
        # Setting up instance variables
        self.population_type = population_type
        self.n_population = n_population
        self.mutation_rate = mutation_rate
        self.total_generation = total_generation
        self.create_population()

    def create_population(self):
        """
        create_population

        It performs the following operations:
            - It creates the initial population, privately.
        
        Input:
            []

        Output:
            []
    
        Usage:
            self.create_population()


        """

        self.population = [self.population_type()
                           for k in range(self.n_population)]

    def create_mating_pool(self):
        """
        create_mating_pool

        It performs the following operations:
            - It constructs the mating pool, privately.
        
        Input:
            []

        Output:
            []
    
        Usage:
            self.create_mating_pool()


        """
        # Calculate fitness scores of each individual in the problem class
        fit_scores = np.array([individual.get_fitness() for individual in self.population])
        # Specifically, for this problem, the inverse and square operation are applied to the scores to represent scores inversely correlated
        # TODO: Convert this into generic scheme, by taking input from constructor
        fit_scores = np.power(fit_scores ** 2, -1, out = np.zeros_like(fit_scores), where = fit_scores != 0.0)
        # Calculation of sampling proabilities by using softmax oprator
        probs = softmax(fit_scores)
        # Create a mating pool based on the sampling likelihood. 
        self.mating_pool = np.random.choice(a=self.population, size=self.n_population, p=probs)

    def reproduction(self):

        """
        reproduction

        It performs the following operations:
            - It "creates" new childs for the new generation, privately.
        
        Input:
            []

        Output:
            []
    
        Usage:
            self.reproduction()

        """
        # For loop for replacing every individual in the current population
        for k in range(self.n_population):
            # Select two individual from mating pool
            parents = np.random.choice(a = self.mating_pool, size = 2)
            # Apply crossover and mutation on them
            self.population[k] = (parents[0].crossover(
                parents[1])).mutate(self.mutation_rate)
        #self.population = np.array(self.population)

    def survive(self):
        """
        survive

        It performs the following operations:
            - It basically loops over mating pool construction and reproduction, publicly.
            - It continues until a predetermined total generation is reached.
        
        Input:
            []

        Output:
            history = History of previously found fittest individual in the generation
    
        Usage:
            GenAlgo_instance.survive()

        """
        history = []
        # Saving the history of previously found candidates

        for k in range(self.n_population):
            # Save the fittest individual
            history.append((self.get_fittest().dna, self.get_fittest().fitness))
            # Verbose
            print("Generation {0} created!\nBest fit score = {1}".format(k, self.get_fittest().fitness))
            # Loop over mating pool construction and reproduction
            self.create_mating_pool()
            self.reproduction()
        
        print("I hope the best survived!")
        return history
        
    def get_fittest(self):
        """
        get_fittest

        It performs the following operations:
            - It calculates the fittest individual in the generation, privately.

        Input:
            []

        Output:
            fittest_individual = Fittest individual in the generation, in the form of problem class
    
        Usage:
            self.get_fittest()

        """
        
        fittest_ix = np.argmin([individual.get_fitness()
                                for individual in self.population])

        return self.population[fittest_ix]
