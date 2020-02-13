from IChromosome import IChromosome
import numpy as np
from Chromosome import Chromosome
from Genome import Genome
from operator import itemgetter
import random
import copy

class Genetic:
    _chromosomes=[]
    _popFitness=[]
    _iteration=[]
    _popSize=[]
    _maxChrome=[]
    _maxFitness=0
    _genomes=[]
    _perMut=[]

    def __init__(self, iteration,popSize,perMut,genomes):
        self._iteration=iteration
        self._popSize=popSize
        self._genomes=genomes
        self._perMut=perMut

    def Run(self):
        self.InitPop()
        i=0
        while(i<self._iteration):
            print("---------- Genetic Iteration No "+str(i+1)+" from "+str(self._iteration)+"--------")
            self.PopFitness()
            indxMaxFit, maxFit = max(enumerate(self._popFitness), key=itemgetter(1))
            maxchrome=self._chromosomes[indxMaxFit]
            print("----The Maximum accuracy in iteraton No"+ str(i+1)+" is "+str(maxFit)+"------------")
            if(self._maxFitness<maxFit):
                self._maxFitness=maxFit
                self._maxChrome=maxchrome
            self.Selection()
            self.CrossOvers()
            self.Mutations()
            i+=1
        print("    Max accuracy is "+str(self._maxFitness)+" with attributes")
        print(self._maxChrome)

    def InitPop(self):
        for i in range(self._popSize):
            new_chrm = Chromosome(copy.deepcopy(self._genomes)) #created random genes values in chromosome
            self._chromosomes.append(new_chrm)
    
    def PopFitness(self):
        self._popFitness= [float(chrom.fitness()) for chrom in self._chromosomes]
    
    def Selection(self):
        fitnessSum=sum(self._popFitness)
        probLst=map((lambda x :x/fitnessSum),self._popFitness)
        cumVlu=0
        cumProbLst=[]
        for prob in probLst:
            cumProbLst.append(cumVlu+prob)
            cumVlu+=prob
        cumProbLst[-1]=1.0
        selectedIndex=[]
        for i in range(self._popSize):
            rn = random.random()
            for j, cum_prob in enumerate(cumProbLst):
                if rn<= cum_prob:
                    selectedIndex.append(j)
                    break 
        self._chromosomes=itemgetter(*selectedIndex)(self._chromosomes)
    def CrossOvers(self):
        offsprings=[]
        chromosomesIter=iter(self._chromosomes)
        for i in range(0,self._popSize,2):
            if i<=self._popSize-2 or (self._popSize-2)%2==0:
                crom=next(chromosomesIter)
                offsp1,offsp2=crom.Crossover(next(chromosomesIter))
                offsprings.append(offsp1)
                offsprings.append(offsp2)
            else:
                offsprings.append(next(chromosomesIter))
        self._chromosomes=offsprings
    
    def Mutations(self):
        offSprings=[]
        noPerMut=int(np.around(self._perMut * self._popSize))
        selectedItems=random.sample(list(np.arange(self._popSize)),noPerMut)
        for x in selectedItems:
            self._chromosomes[x].Mutation()
    