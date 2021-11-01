
import os
import logging
import pandas as pd
import joblib
import scipy.sparse as sparse
import numpy as np

## sparse matrix is nothing but a matrix which stores only non zero values . 
# Genrally matrix contains lot of zero values which needs more memory to store. So we must use sparse matrix


def save_matrix(df,matrix,out_path):
    id_matrix=sparse.csr_matrix(df.id.astype(np.int64)).T 
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T
    result = sparse.hstack([id_matrix, label_matrix, matrix], format="csr")

    msg = f"The ouput matrix {out_path} of size {result.shape} and data type: {result.dtype}"
    logging.info(msg)
    joblib.dump(result, out_path)

