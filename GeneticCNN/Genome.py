import random

class Genome:
    _genName=[]
    _start=[]
    _end=[]
    _value=[]
    def __init__(self, genName,start,end,value=[]):
        self._genName= genName
        self._start=   start  
        self._end=     end
        self._value=   value
    
    def Mutate(self):
        self._value=random.randrange(self._start,self._end)
    def InitValue(self):
        #self._value=self._start
        self._value=random.randrange(self._start,self._end)
        return self._value