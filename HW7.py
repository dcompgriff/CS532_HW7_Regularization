import numpy as np
import time
import random as rand
import cvxpy as cvx






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





def main():
    print('Running main...')
    #q1_partb()
    q2()










if __name__ == '__main__':
    main()