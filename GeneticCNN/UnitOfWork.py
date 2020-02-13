from DataSet import DataSet
from Genome import Genome
class UnitOfWork:
    
    _dataSet=[]
    _genomes=[Genome('segment_size',150,200),
              Genome('n_filters',150,200),
              Genome('n_layers',1,5),
              Genome('filters_size',3,5),
              Genome('epochs',100,200),
              Genome('batch_size',100,200),
              Genome('IncludeFeat',0,2)
              ]
    _popSize=10
    _perMut=0.5
    _iteration=1000
    
# _test_user_ids=[2, 4, 9, 10, 12, 13, 18, 20, 24] in DataSet

    def __init__(self, pathDataset='datasets/uci_raw_data'):
        _genomes=[]
        self._dataSet=DataSet(pathDataset,'l')  
            
    