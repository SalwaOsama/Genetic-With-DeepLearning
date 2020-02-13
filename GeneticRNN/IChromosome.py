from interface import implements,Interface
from INetwork import INetwork
class IChromosome (Interface):
    def fitness(self):
        pass
    def Crossover(self,iChromosome):
        pass
    def Mutation(self):
        pass
    