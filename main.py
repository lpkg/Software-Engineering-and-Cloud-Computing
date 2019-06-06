from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import sql
import numpy as np
import time
from pyspark.mllib.linalg.distributed import DenseMatrix
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry


if __name__ == "__main__":
    # set up spark context and configuration
    conf = SparkConf().setAppName("PythonPCAOnRowMatrixExample")
    sc = SparkContext(conf=conf)
    print(sc.getConf().getAll())    
    sqlContext = sql.SQLContext(sc)

    # load data
    data = sc.textFile("gs://dataproc-ae279739-4c78-478e-9024-8b7ea842f82e-us/heart1.txt")
    entries = data.map(lambda l: l.split(' ')).map(lambda l: MatrixEntry(np.long(l[0]), np.long(l[1]), np.float(l[2])))

    # create RowMatrix   
    premat = CoordinateMatrix(entries)
    mat = premat.toIndexedRowMatrix()

    print(mat.numCols())
    print(mat.numRows())

    # gramian
    start_time = time.time()
    decomp = mat.computeGramianMatrix()
    elapsedtime = time.time() - start_time
    print(elapsedtime)

    # svd
    start_time = time.time()
    decomp = mat.computeSVD(1000)
    elapsedtime = time.time() - start_time
    print(elapsedtime)

    sc.stop()
