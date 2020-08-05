from .o3d import kdtree as o3d_kdtree
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
import os

class _NearestNeighbors(object):
    def __init__(self, set_k=None, **kwargs):
        self.model = None
        self.set_k = set_k

    def train(self, data):
        pass

    def search(self, data, k, return_distance=True):
        if self.set_k is not None:
            assert self.set_k == k, \
                'K not match to setting {}'.format(self.set_k)
        D, I = None, None
        return D, I


class Open3dNN(_NearestNeighbors):
    def __init__(self, set_k=None, **kwargs):
        super(Open3dNN, self).__init__(set_k, **kwargs)
        self.model = None

    def train(self, data):
        assert data.shape[1] == 3, 'Must be shape [?, 3] for point data'
        self.model = o3d_kdtree(data)

    def search(self, data, k, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if data.shape[0] == 1:
            [__, I, _] = self.model.search_knn_vector_3d(data[0], k)
        else:
            I = np.zeros((data.shape[0], k), dtype=np.int)
            with ThreadPoolExecutor(256) as executor:
                for i in range(I.shape[0]):
                    executor.submit(self._search_multiple, (self.model, I, data, k, i,))
        return None, I

    @staticmethod
    def _search_multiple(knn_searcher, I, data, k, i):
            [__, I_, _] = knn_searcher.search_knn_vector_3d(data[i, :], k)
            I[i, :] = np.asarray(I_)


class FaissNN(_NearestNeighbors):
    #GPU KNN Search for large scale
    def __init__(self, set_k=None, **kwargs):
        super(FaissNN, self).__init__(set_k, **kwargs)
        self.IVF_number = 32786
        self.GPU_id = None
        if isinstance(kwargs, dict):
            if 'IVF_number' in kwargs: self.IVF_number = kwargs['IVF_number']
            if 'GPU_id' in kwargs: self.GPU_id = kwargs['GPU_id']
        self.model = None
        self.dimension = None

    def train(self, data):
        d = data.shape[1]
        data = data.astype(np.float32)
        self.model = faiss.index_factory(int(d), 'IVF{}_HNSW32,Flat'.format(self.IVF_number)) #_HNSW32
        if self.GPU_id is not None and isinstance(self.GPU_id, int):
            res = faiss.StandardGpuResources()
            self.model = faiss.index_cpu_to_gpu(res, self.GPU_id, self.model)
        elif isinstance(self.GPU_id, list):
            #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in self.GPU_id])
            self.model = faiss.index_cpu_to_all_gpus(self.model)
        else:
            self.model = faiss.index_cpu_to_all_gpus(self.model)
        self.model.train(data)
        self.model.add(data)
        self.model.nprobe = d ** 2

    def search(self, data, k, return_distance=True):
        data = data.astype(np.float32)
        assert self.model is not None, "Model have not been trained"
        #assert self.model.is_trained, "Model not trained."
        D, I = self.model.search(data, k)
        if return_distance: D = None
        return D, I


if __name__ == "__main__":
    import sys
    import time
    import os

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    nb = 10**5
    nq = 10**5
    np.random.seed(1)
    datab = np.random.rand(nb, 3).astype('float32')
    dataq = np.random.rand(nq, 3).astype('float32')

    tic = time.time()
    nn = SkNN(set_k=3)
    nn.train(datab)
    print(time.time() - tic)
    tic = time.time()
    D, I = nn.search(dataq, 3)
    print(time.time() - tic)




