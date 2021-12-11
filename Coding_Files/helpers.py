from fasta_reader import readFile
import numpy as np

def OnehotEncoding(inpStr):
    _res = []
    for base in inpStr:
        if base == "G":
            base = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "A":
            base = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "V":
            base = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "L":
            base = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "I":
            base = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "P":
            base = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "F":
            base = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "Y":
            base = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "W":
            base = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "S":
            base = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif base == "T":
            base = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif base == "C":
            base = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif base == "M":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif base == "N":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif base == "Q":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif base == "D":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif base == "E":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif base == "K":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif base == "R":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif base == "H":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif base == "X":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        _res.append(base)
    
    return _res


def dictEncoding(inpStr):
    _res = []
    word2int = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}

    for inp in range(len(inpStr)):
        _res.append(word2int.get(inpStr[inp]))

    return _res


def createTrainTestData(posSample,negSample,Encodingtype):
    TrainTest=[]
    seq_len=[]
    num=[]
    pos_label = np.ones((len(posSample),1))
    neg_label = np.zeros((len(negSample),1))
    Label = np.concatenate((pos_label,neg_label),axis=0).flatten()
    TrainTestSample = posSample + negSample

    if Encodingtype == "dict":
       for i in TrainTestSample:
           seq_len=len(i)
           i=np.array(dictEncoding(i)).reshape([1,seq_len])
           TrainTest.append(i)
       TrainTest=np.array(TrainTest).reshape(len(TrainTestSample),seq_len)
       return Label, TrainTest
    else:
       for i in TrainTestSample:
           num = len(i) * 20
           i=np.array(OnehotEncoding(i)).reshape([1,num])
           TrainTest.append(i)
       TrainTest=np.array(TrainTest).reshape(len(TrainTestSample),num)    
       return Label, TrainTest

           


def createData(Sample,Encodingtype):
    Feature=[]
    seq_len=[]
    num=[]
    if Encodingtype == "dict":
       for i in Sample:
           seq_len=len(i)
           i=np.array(dictEncoding(i)).reshape([1,seq_len])
           Feature.append(i)
       Feature=np.array(Feature).reshape(len(Sample),seq_len)
       return Feature
    else:
       for i in Sample:
           num = len(i) * 20
           i=np.array(OnehotEncoding(i)).reshape([1,num])
           Feature.append(i)
       Feature=np.array(Feature).reshape(len(Sample),num)
       return Feature






  
    
     


     
        
        

     
     