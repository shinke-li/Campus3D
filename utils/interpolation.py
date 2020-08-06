import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
from dataset.data_utils import cal_knn
def interpolate(sparse_points, dense_points, nn_num=1, GPU_id=None):
    '''
    sparse_points: predicted points
    '''
    if GPU_id is not None and cal_knn.FAISS_INSTALLED:
        knn_module = cal_knn.FaissNN
    else:
        knn_module = cal_knn.Open3dNN
    knn = knn_module(GPU_id=GPU_id)
    knn.train(sparse_points)
    return knn.search(dense_points, nn_num)

if __name__ == '__main__':
    import numpy as np
    import time
    d_pts = np.random.rand(20000000, 3)
    s_pts = np.random.rand(10000000, 3)
    tic = time.time()
    _, ind = interpolate(s_pts, d_pts, nn_num=1, GPU_id=6)
    print(time.time() - tic)
    print(ind.shape)
    print(s_pts[ind].shape)