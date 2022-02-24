import numpy as np
from loadData import loadData
from trainAndTest import trainSGD, test
import pandas as pd
import matplotlib.pyplot as plt
import time

def main():
    trainingTime = []
    timeElapsed = []
    MSEtr = []
    MSEtest = []
    EVTr = []
    EVTest = []
    for col in range(8):
        print(col)
        np.random.seed(165)
        Xtr, Xtest, Ytr, Ytest = loadData('Concrete_Data.xls', univariate=True, preprocess=False, column_select=col)
        (n, d) = Xtr.shape
        starttime = time.time()
        w, t = trainSGD(Xtr, Ytr, 10000, .0001)
        endtime = time.time()
        trainLoss, trainVarExp = test(Xtr, Ytr, w)
        testLoss, testVarExp = test(Xtest, Ytest, w)
        trainingTime.append(t)
        MSEtr.append(trainLoss)
        EVTr.append(trainVarExp)
        MSEtest.append(testLoss)
        EVTest.append(testVarExp)
        elapsed = endtime - starttime
        timeElapsed.append(elapsed)

        predictions = np.dot(w, Xtr.T).reshape((900,))
        x = np.delete(Xtr, 0, axis=1).T[0]
        plt.scatter(x, Ytr, s=1, alpha=0.5, c='b')
        plt.scatter(x, predictions, s=1, alpha=0.1, c='r')
        plt.show()

    np.random.seed(165)
    Xtr, Xtest, Ytr, Ytest = loadData('Concrete_Data.xls', univariate=False, preprocess=False)
    starttime = time.time()
    w, t = trainSGD(Xtr, Ytr, 10000, .0001)
    endtime = time.time()
    trainLoss, trainVarExp = test(Xtr, Ytr, w)
    testLoss, testVarExp = test(Xtest, Ytest, w)
    trainingTime.append(t)
    MSEtr.append(trainLoss)
    EVTr.append(trainVarExp)
    MSEtest.append(testLoss)
    EVTest.append(testVarExp)
    elapsed = endtime - starttime
    timeElapsed.append(elapsed)
    results = {
        "MSE on Training Set": MSEtr,
        "MSE on Test Set": MSEtest,
        "Variance Explained on Training Set": EVTr,
        "Variance Explained on Test Set": EVTest,
        "Number of Descent Iterations": trainingTime,
        "Time to Convergence": timeElapsed
    }
    results_df = pd.DataFrame(results)


    ##################################################################################

    #Now with PreProcessing

    ##################################################################################

    trainingTime = []
    MSEtr = []
    MSEtest = []
    EVTr = []
    EVTest = []
    timeElapsed = []
    for col in range(8):
        print(col)
        np.random.seed(165)
        Xtr, Xtest, Ytr, Ytest = loadData('Concrete_Data.xls', univariate=True, preprocess=True, column_select=col)
        (n, d) = Xtr.shape
        starttime = time.time()
        w, t = trainSGD(Xtr, Ytr, 10000, .0001)
        endtime = time.time()
        trainLoss, trainVarExp = test(Xtr, Ytr, w)
        testLoss, testVarExp = test(Xtest, Ytest, w)
        trainingTime.append(t)
        MSEtr.append(trainLoss)
        EVTr.append(trainVarExp)
        MSEtest.append(testLoss)
        EVTest.append(testVarExp)
        elapsed = endtime - starttime
        timeElapsed.append(elapsed)

        predictions = np.dot(w, Xtr.T).reshape((900,))
        x = np.delete(Xtr, 0, axis=1).T[0]
        plt.scatter(x, Ytr, s=1, alpha=0.5, c='b')
        plt.scatter(x, predictions, s=1, alpha=0.1, c='r')
        plt.show()

    np.random.seed(165)
    Xtr, Xtest, Ytr, Ytest = loadData('Concrete_Data.xls', univariate=False, preprocess=True)
    starttime = time.time()
    w, t = trainSGD(Xtr, Ytr, 10000, .0001)
    endtime = time.time()
    trainLoss, trainVarExp = test(Xtr, Ytr, w)
    testLoss, testVarExp = test(Xtest, Ytest, w)
    trainingTime.append(t)
    MSEtr.append(trainLoss)
    EVTr.append(trainVarExp)
    MSEtest.append(testLoss)
    EVTest.append(testVarExp)
    elapsed = endtime-starttime
    timeElapsed.append(elapsed)
    results_pp = {
        "MSE on Training Set": MSEtr,
        "MSE on Test Set": MSEtest,
        "Variance Explained on Training Set": EVTr,
        "Variance Explained on Test Set": EVTest,
        "Number of Descent Iterations": trainingTime,
        "Time to Convergence": timeElapsed
    }
    results_df_pp = pd.DataFrame(results_pp)
    writer = pd.ExcelWriter('Output.xlsx')
    results_df_pp.to_excel(writer, 'PreProcessed')
    results_df.to_excel(writer, 'NoPP')
    writer.save()
if __name__ == "__main__":
    main()