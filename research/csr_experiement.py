import numpy as np
from scipy.sparse import csr_matrix

if __name__ =="__main__":
    A=np.array([[1,0,0,1,0,0],
    [0,1,1,0,0,0],
[0,0,0,1,0,0],
[0,0,0,0,0,0]])
    print(A)
    s=csr_matrix(A)
    print(s)
    print(s.todense())
