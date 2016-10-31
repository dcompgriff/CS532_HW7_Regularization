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
    #q3_partab(A, b)
    q3_partc(A, b)
    #q3_partd(A, b)




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

    # PART A
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
    indexArray = list(range(0, A.shape[0]))




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
    #q1_partb()
    #q2()
    q3()










if __name__ == '__main__':
    main()