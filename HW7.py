import numpy as np
import time
import random as rand
import cvxpy as cvxpy






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






def main():
    print('Running main...')
    q1_partb()











if __name__ == '__main__':
    main()