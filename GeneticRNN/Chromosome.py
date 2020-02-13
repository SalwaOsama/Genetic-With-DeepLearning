from interface import implements,Interface
from IChromosome import IChromosome
import random
from RNN import RNN
from Genome import Genome
from UnitOfWork import UnitOfWork
from DataSet import DataSet
import numpy as np
import copy


class Chromosome(implements(IChromosome)):
    #List of integers
    _genomes=[]
    _shelveDataFile=[]
    _accuracy=[]
    def __init__(self, genomes):
        self._genomes=copy.deepcopy(genomes)
        for i in range(len(self._genomes)):
            self._genomes[i].InitValue()
       
    def fitness(self):
        uow=UnitOfWork()
        genoWithSegSiz=[geno for geno in self._genomes if geno._genName== 'segment_size']
        if genoWithSegSiz==[]:
            self._shelveDataFile=uow._dataSet().PreparingData()
        else:
            segment_size=genoWithSegSiz[0]._value
            self._shelveDataFile=uow._dataSet.PreparingData(segment_size)
        RNN=RNN(self._shelveDataFile,self._genomes)
        self._accuracy=cnn.RunAndAccuracy()
        return self._accuracy
    def Crossover(self, iChromosome):
        pointsplit=random.randint(0,len(iChromosome._genomes) -1)
        offspring1=Chromosome(copy.deepcopy(self._genomes))
        offspring2=Chromosome(copy.deepcopy(self._genomes))
        parent1=self
        parent2=iChromosome
        #Genes
        offspring1._genomes=np.hstack((parent1._genomes[:pointsplit],parent2._genomes[pointsplit:]))
        offspring2._genomes=np.hstack((parent2._genomes[:pointsplit],parent1._genomes[pointsplit:]))
        return offspring1,offspring2

    def Mutation(self):
        rndIndex=random.randrange(len(self._genomes))
        genome=self._genomes[rndIndex]
        selectedRnVlu=genome.Mutate()
        return self 
    def __str__(self):
        [print(str(gen._genName)+" is "+ str(gen._value))  for gen in self._genomes] 
        return " "
