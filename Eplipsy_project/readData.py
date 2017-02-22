
import scipy.io as sio
def loadData(filename,mName):
    load_fn = filename
    load_data = sio.loadmat(load_fn)
    load_matrix = load_data[mName]
    return load_matrix
def loadData1(filename):
    data = sio.loadmat(filename)
    return data