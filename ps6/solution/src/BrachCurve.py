from typing import MutableSet
import numpy as np
from Utils import numerical_derivative, numerical_integral
from Utils import cycloid

class BrachCurve:
    """
    BrachCurve

    It is an subclass of (not right now, but will be) a generic class which can be
    used in GenAlgo class.

    The main purpose of this class to define necessary variable which will be
    used in Genetic Algorithm. It overrides the following functions:
    - init() [constructor]
    - get_fitness()
    -crossover()
    -mutate()



    """
    def __init__(self, n_dna=10, dna=None):
        """
        __init__

        It performs the following operations:
            - It constructs the BrachCurve class
        
        Input:
            n_dna = Number of chromosomes in the DNA
            dna = Preconstructed dna array

        Output:
            []
    
        Usage:
            BrachCurve(10)
            BrachCurve(predetermined_dna_array)

        """
        self.scale_factor = 1.0
        if dna is None:
        
            self.n_dna = n_dna  # Number of data point to represent curve
            # Initial distribution of y values
            self.dna = np.abs(np.random.randn(self.n_dna)) * self.scale_factor
        
        else:

            self.dna = dna
            self.n_dna = self.dna.shape[0]

        self.dna_type = self.dna.dtype

    def get_fitness(self):

        """
        get_fitness

        It performs the following operations:
            - It calculates the fitness value of the BrachCurve instance
        
        Input:
            []

        Output:
            fitness = Calculated fitness value
    
        Usage:
            BrachCurve.get_fitness()
        """

        # ! TODO: It looks like, right now, the elapsed time is not calculated correctly. Verify it and make it correct.
        g = 9.81  # Gravitational acceleration
       
        # Start point and end point of y is known
        y = np.block([0.0, self.dna, 1.0]) 
        # Take numerical derivative of the function f(x)
        d_y = numerical_derivative(y, 0.0, 1.0, self.n_dna + 1) 

        # Integrand is calculated according to the HW statement
        a = 1 + (d_y ** 2)
        b = 2 * g * y
        # To avoid DivisionByZero, this syntax is used
        integrand = np.divide(a, b, out = np.zeros_like(a), where = b != 0)
        integrand = np.sqrt(integrand)

        I_integrand = numerical_integral(
            integrand, 0.0, 1.0, self.n_dna + 1, method="trapezoidal")  # Take numerical integral
        
        # The last term of the integral equals to the individual integral with respect to end point
        self.fitness = I_integrand[-1]
        
        return self.fitness

    def crossover(self, lover):

        """
        crossover

        It performs the following operations:
            - It performs crossover operation between two BrachCurve instance
        
        Input:
            lover = Second BrachCurve instance other than ´self´

        Output:
            child_instance = Crossover-ed instance of BrachCurve
    
        Usage:
            BrachCurve_instance_1.crossover(BrachCurve_instance_2)

        """
        # Determine a arbitrary point to split DNAs
        midpoint = np.random.randint(0, self.n_dna)
        # Preallocate child's DNA
        child_dna = np.empty(self.dna.shape, dtype=self.dna_type)
        # Reunite child DNA from parents
        child_dna = np.block(
            [self.dna[:(midpoint + 1)], lover.dna[(midpoint + 1):]])


        return self.__class__(dna = child_dna)


    def mutate(self, mutation_rate):
        """
        mutate

        It performs the following operations:
            - It performs mutation operation on the given BrachCurve instance
        
        Input:
            mutation_rate = Likelihood of the mutation

        Output:
            mutated_BrachCurve_instance = Mutated form of the ´self´
    
        Usage:
            BrachCurve_instance.mutate(0.01)

        """
        
        # Determine mutation probabilities for whole DNA array
        mutation_prob = np.random.rand(self.n_dna)
        # Select which one will be mutated
        mutation = mutation_prob < mutation_rate
        n_mutation = mutation[mutation == True].shape[0]
        # And, finally, mutate!
        self.dna[mutation] = np.abs(np.random.randn(n_mutation)) * self.scale_factor

        return self
