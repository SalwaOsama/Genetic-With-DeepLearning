from DataSet import DataSet
from Genome import Genome
class UnitOfWork:
    
    _dataSet=[]
    _genomes=[Genome('Num_units',20,50),
              Genome('learning_rate',0. 0020,0.0030),
              Genome('lambda_loss_amount',0.0010,0.0020),
              Genome('Batch_size',1000,2000),
              Genome('Num_iterations',100,500),
              Genome('Segment_size',100,200),
              ]
    _popSize=10
    _perMut=0.5
    _iteration=22
    
# _test_user_ids=[2, 4, 9, 10, 12, 13, 18, 20, 24] in DataSet

    def __init__(self, pathDataset='datasets/uci_raw_data'):
        _genomes=[]
        self._dataSet=DataSet(pathDataset,'l')  
            
    