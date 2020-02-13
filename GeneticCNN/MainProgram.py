#import CNN
from DataSet import DataSet
from CNN import CNN
import shelve
from Genome import Genome
from Genetic import Genetic
from UnitOfWork import UnitOfWork

if __name__ == "__main__":
    UOF=UnitOfWork()
    genetic=Genetic(UOF._iteration,UOF._popSize,UOF._perMut,UOF._genomes)
    print(genetic.Run())

    