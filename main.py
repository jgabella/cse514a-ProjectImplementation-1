import numpy as np
import math
from loadData import loadData
from trainAndTest import trainSGD, test

def main():
    Xtr, Xtest, Ytr, Ytest = loadData('Concrete_Data.xls')
    (n, d) = Xtr.shape
    w = trainSGD(Xtr, Ytr, 10000, 0.1)

if __name__ == "__main__":
    main()