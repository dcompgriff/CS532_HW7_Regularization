import numpy as np
import time
import random as rand
import cvxpy as cvx
import matplotlib.pyplot as plt
from multiprocessing import Pool


def q1_partb():
    A = np.random.rand(8000*100).reshape((8000, 100))
    b = np.random.rand(8000).reshape((8000, 1))
    lambd = 0.01

    start1 = time.time()
    x1 = np.linalg.inv(A.T.dot(A) + lambd).dot(A.T).dot(b)
    end1 = time.time()
    print('Time to solve with transpose on right: %.2f' % ((end1 - start1)*60))
    start2 = time.time()
    x2 = A.T.dot(np.linalg.inv(A.dot(A.T) + lambd)).dot(b)
    end2 = time.time()
    print('Time to solve with transpose on left: %.2f' % ((end2 - start2) * 60))




def q2():
    A = np.array([[1, 0],
         [2, 1],
         [1, 3]])
    b = np.array([[4],
                  [5],
                  [7]])
    lambd = 0.01

    #Normal Equation Solution.
    x1 = np.linalg.inv(A.T.dot(A) + lambd).dot(A.T).dot(b)
    print('Normal Equation Solution: ')
    print(str(x1))

    #Convex optimization solution.
    x2 = cvx.Variable(2)
    lambd2 = cvx.Parameter(sign="positive")
    lambd2.value = 0.01
    objective = cvx.Minimize(cvx.sum_entries(cvx.square(A*x2 - b)) + (lambd2*cvx.sum_entries(cvx.square(x2))))
    prob = cvx.Problem(objective)
    print('\nCVX Optimal value: %f' %(prob.solve()))
    print('CVX Optimal x: ')
    print(str(x2.value))


def q3():
    global A, b
    A = np.loadtxt('./bcdata.csv', delimiter=',')
    #Pull out b as first column, and the rest of the data as A.
    b = A[:, 0].reshape((A.shape[0], 1))
    A = A[:, 1:]

    #Set up the parameters of the equation.
    q3_partab(A, b)
    q3_partc(A, b)
    q3_partd(A, b)

def q3_partab(A, b):
    lambdValues = [.001, .01, .1, .5,  1, 5, 10, 50, 100]

    # Set up the parameters of the equation.
    lambd = cvx.Parameter(sign="positive")
    lambd.value = 0.01
    Areduced = A[:100, :]
    breduced = b[:100]
    x = cvx.Variable(A.shape[1])
    error = cvx.sum_squares(Areduced * x - breduced)
    objective = cvx.Minimize(error + (lambd * cvx.norm1(x)))
    prob = cvx.Problem(objective)

    #PART A
    sq_err = []
    l1_err = []
    optimalX = []
    for val in lambdValues:
        print('Solving lambda: %f' %(val))
        #Solve the problem with the new lambda.
        lambd.value = val
        prob.solve()
        #Calculate the error and optimal x.
        sq_err.append(error.value)
        l1_err.append(cvx.norm1(x).value)
        optimalX.append(np.array(x.value))


    #Plot the trade off curve.
    plt.plot(l1_err, sq_err)
    plt.xlabel('|x|_1')
    plt.ylabel('|Ax-b|^2')
    plt.title('Trade-Off Curve for LASSO')
    plt.show()

    #PART B
    errorRateList = []
    sparsityList = []
    for xopt in optimalX:
        #Predict the training data
        ypredicted = []
        for data in Areduced:
            predVal = data.reshape(1, data.size).dot(xopt.reshape(xopt.size, 1))
            if predVal > 0:
                predVal = 1
            else:
                predVal = -1
            ypredicted.append(predVal)

        err = 1.0 - calculateAccuracy(breduced, ypredicted)
        sparsity = countNonzero(xopt)
        errorRateList.append(err)
        sparsityList.append(sparsity)


    plt.plot(sparsityList, errorRateList)
    plt.scatter(sparsityList, errorRateList, color='r')
    plt.ylabel(r'error rate')
    plt.xlabel(r'\sparsity')
    plt.title('Error Sparsity Trade-Off Curve')
    plt.show()

def q3_partc(A, b):
    lambdValues = [.001, .01, .1, .5, 1, 5, 10, 50, 100]

    # Set up the parameters of the equation.
    lambd = cvx.Parameter(sign="positive")
    lambd.value = 0.01
    Areduced = A[:100, :]
    breduced = b[:100]
    x = cvx.Variable(A.shape[1])
    error = cvx.sum_squares(Areduced * x - breduced)
    objective = cvx.Minimize(error + (lambd * cvx.norm1(x)))
    prob = cvx.Problem(objective)

    # PART C
    sq_err = []
    l1_err = []
    optimalX = []
    for val in lambdValues:
        print('Solving lambda: %f' % (val))
        # Solve the problem with the new lambda.
        lambd.value = val
        prob.solve()
        # Calculate the error and optimal x.
        sq_err.append(cvx.sum_squares(A[100:, :] * x - b[100:]).value)
        l1_err.append(cvx.norm1(x).value)
        optimalX.append(np.array(x.value))

    # Plot the trade off curve.
    plt.plot(l1_err, sq_err)
    plt.xlabel('|x|_1')
    plt.ylabel('|Ax-b|^2')
    plt.title('Trade-Off Curve for LASSO')
    plt.show()

    #Error rate tradeoff curve
    errorRateList = []
    sparsityList = []
    for xopt in optimalX:
        # Predict the training data
        ypredicted = []
        for data in A[100:, :]:
            predVal = data.reshape(1, data.size).dot(xopt.reshape(xopt.size, 1))
            if predVal > 0:
                predVal = 1
            else:
                predVal = -1
            ypredicted.append(predVal)

        err = 1.0 - calculateAccuracy(breduced, ypredicted)
        sparsity = countNonzero(xopt)
        errorRateList.append(err)
        sparsityList.append(sparsity)

    plt.plot(sparsityList, errorRateList)
    plt.scatter(sparsityList, errorRateList, color='r')
    plt.ylabel('validation error rate')
    plt.xlabel('sparsity')
    plt.title('Validation Error Sparsity Trade-Off Curve')
    plt.show()


def q3_partd(A, b):
    lambdValues = [.0001, .001, .01, .1, .5, 1, 5, 10, 50, 100]
    indexArray = list(range(0, A.shape[0]))
    subsetList = []
    subsets = []
    classes = []

    #PART i
    print('Runing q3 partd, section i')
    #Make 5 random subset index lists of size 29.
    for i in range(0, 5):
        subsetList.append([])
        for m in range(0, 29):
            randi = rand.randint(0, len(indexArray)-1)
            subsetList[i].append(indexArray[randi])
            del indexArray[randi]
    #Make 5 random subset index lists of size 30.
    for i in range(5, 10):
        subsetList.append([])
        for m in range(0, 30):
            randi = rand.randint(0, len(indexArray)-1)
            subsetList[i].append(indexArray[randi])
            del indexArray[randi]
    #Make actual data subsets out of the
    for i in range(0, len(subsetList)):
        subsets.append(A[subsetList[i]])
        classes.append(b[subsetList[i]])

    #PART ii (data sets 0 through 7)
    print('Runing q3 partd, section ii')
    Atrain = None
    btrain = None
    for i in range(0, 8):
        if Atrain == None:
            Atrain = subsets[i]
            btrain = classes[i]
        else:
            Atrain = np.vstack((Atrain, subsets[i]))
            btrain = np.vstack((btrain, classes[i]))
    # Set up the parameters of the equation for LASSO.
    lambd = cvx.Parameter(sign="positive")
    lambd.value = 0.01
    x = cvx.Variable(A.shape[1])
    objective = cvx.Minimize(cvx.sum_squares(Atrain * x - btrain) + (lambd * cvx.norm1(x)))
    prob = cvx.Problem(objective)
    optimalRidgeX = []
    optimalLassoX = []
    #Find optimal lasso x's
    for val in lambdValues:
        print('Solving lambda: %f' % (val))
        # Solve the problem with the new lambda.
        lambd.value = val
        prob.solve()
        optimalLassoX.append(np.array(x.value))
    #Find optimal ridge x's
    prob.objective = cvx.Minimize(cvx.sum_squares(Atrain * x - btrain) + (lambd *cvx.sum_entries(cvx.square(x))))
    for val in lambdValues:
        print('Solving lambda: %f' % (val))
        # Solve the problem with the new lambda.
        lambd.value = val
        prob.solve()
        optimalRidgeX.append(np.array(x.value))

    #PART iii
    print('Runing q3 partd, section iii')
    Atuning = subsets[8]
    btuning = classes[8]
    bestLassoX = optimalLassoX[0]
    bestLassoAccuracy = 0
    bestRidgeX = optimalRidgeX[0]
    bestRidgeAccuracy = 0
    #Find best Lasso
    for xopt in optimalLassoX:
        ypredicted = []
        for data in Atuning:
            predVal = data.reshape(1, data.size).dot(xopt.reshape(xopt.size, 1))
            if predVal > 0:
                predVal = 1
            else:
                predVal = -1
            ypredicted.append(predVal)
        accuracy = calculateAccuracy(btuning, ypredicted)
        if accuracy > bestLassoAccuracy:
            bestLassoAccuracy = accuracy
            bestLassoX = xopt
    #Find best Ridge
    for xopt in optimalRidgeX:
        ypredicted = []
        for data in Atuning:
            predVal = data.reshape(1, data.size).dot(xopt.reshape(xopt.size, 1))
            if predVal > 0:
                predVal = 1
            else:
                predVal = -1
            ypredicted.append(predVal)
        accuracy = calculateAccuracy(btuning, ypredicted)
        if accuracy > bestRidgeAccuracy:
            bestRidgeAccuracy = accuracy
            bestRidgeX = xopt

    #PART iv
    print('Runing q3 partd, section iv')
    #Calculate test accuracy for LASSO.
    Atest = subsets[9]
    btest = classes[9]
    ypredicted = []
    for data in Atest:
        predVal = data.reshape(1, data.size).dot(bestLassoX.reshape(bestLassoX.size, 1))
        if predVal > 0:
            predVal = 1
        else:
            predVal = -1
        ypredicted.append(predVal)
    lassoAccuracy = calculateAccuracy(btest, ypredicted)
    #Calculate test accuracy for RIDGE.
    ypredicted = []
    for data in Atest:
        predVal = data.reshape(1, data.size).dot(bestRidgeX.reshape(bestRidgeX.size, 1))
        if predVal > 0:
            predVal = 1
        else:
            predVal = -1
        ypredicted.append(predVal)
    ridgeAccuracy = calculateAccuracy(btest, ypredicted)

    print('Ridge test accuracy: %f' % (ridgeAccuracy))
    print('LASSO test accuracy: %f' % (lassoAccuracy))


def calculateAccuracy(yactual, ypredicted, epsilon=1e-10):
    metrics = {}
    metrics['accuracy'] = 0

    for i in range(0, len(yactual)):
        if ypredicted[i] - yactual[i] <= epsilon:
            metrics['accuracy'] += 1

    metrics['accuracy'] = metrics['accuracy'] / float(len(yactual))
    return metrics['accuracy']

def countNonzero(x, epsilon=1e-5):
    nonZeroCount = 0
    for entry in x:
        if abs(entry) >= epsilon:
            nonZeroCount += 1

    return nonZeroCount

def main():
    print('Running main...')
    q1_partb()
    q2()
    q3()










if __name__ == '__main__':
    main()