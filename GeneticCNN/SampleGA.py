import random
import copy
import numpy as np
import operator

population_size=5
pop=[]
chromeLen=10
def generate_Initpop():
    sample_chrm = np.arange(0,chromeLen) # a feasible solution
    init_population = [ ] # an empty list
    for i in range( population_size ):
        new_chrm = copy.copy(sample_chrm)
        random.shuffle(new_chrm)
        init_population.append(new_chrm) 
    return init_population

def get_Fitness(chrom):
    sol=[1,2,3,4,5,6,7,8,9,10]
    diff=np.absolute(list(map(operator.sub,chrom,sol)))
    cost=sum(diff)
    return cost
    
def Selection(fitnessLst):
    fitnessSum=sum(fitnessLst)
    probLst=map((lambda x :x/fitnessSum),fitnessLst)
    cumVlu=0
    cumProbLst=[]
    for prob in probLst:
        cumProbLst.append(cumVlu+prob)
        cumVlu+=prob
    cumProbLst[-1]=1.0
    selected=[]
    for i in range(population_size):
        rn = random.random()
        for j, cum_prob in enumerate(cumProbLst):
            if rn<= cum_prob:
                selected.append(j)
                break 
    return selected    

def CrossOver(parents,typeCrossOver="1p"):
    pointsplit=random.randint(0,chromeLen-1)
    offsprings=[]
    for i in range(0,len(parents),2):
        if(i<=len(parents)-2 or (len(parents)-2)%2==0):
            parent1=pop[parents[i]]
            parent2=pop[parents[i+1]]
            offsprings.append(np.hstack((parent1[:pointsplit],parent2[pointsplit:])))
            offsprings.append(np.hstack((parent2[:pointsplit],parent1[pointsplit:])))
        else:
            offsprings.append(pop[parents[i]])


    return offsprings

def Mutation(parents,PerMut,typeMutaion="1M",parameter="Filters"):
    offSprings=[]
    noPerMut=np.around(PerMut*population_size)
    selectedItems=random.sample(np.arange(population_size),noPerMut)
    offSprings=parents
    for x in selectedItems:
         randnumFilters=random.randrange(10,20)  #assume numfilters ia the first paramete
         offSprings[x][0]=randnumFilters
    return offSprings



    return offSprings        


# def crossover(twochroms)
# def mutate(parent)
iteration=100
i=0
pop=generate_Initpop()
while(i<iteration and vlFitest<91):
    fitnessLst=[get_Fitness(p)for p in pop]
    indxFitest, vlFitest = max(enumerate(fitnessLst), key=operator.itemgetter(1))
    selected=Selection(fitnessLst)
    offSprings_CO=CrossOver(selected)
    PerMut=0.6 #percentage of mutation
    offSprings_MT=Mutation(offSprings_CO,PerMut)
    pop=offSprings_MT
    i+=1

print(fitness)