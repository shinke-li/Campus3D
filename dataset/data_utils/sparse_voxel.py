import sparse
import numpy as np
import sys
import numba as nb

@nb.jit(nopython=True)
def allocate(idx, idx_end_id, start_id, cnts, seq):
    for i in range(len(start_id)):
        idx[idx_end_id[i] - cnts[i]:idx_end_id[i]] = seq[start_id[i]: start_id[i] + cnts[i]]


def multi_slice_indexing(seq, start_id, cnts):
    if start_id.shape[0] == 0: return _empty()
    idx = np.empty((np.sum(cnts),), dtype=np.int)
    idx_end_id = np.cumsum(cnts)
    allocate(idx, idx_end_id, start_id, cnts, seq)
    return idx

def _empty(shape=(1,)):
    return np.tile(np.array([], dtype=np.int), shape)

def voxel_map_point_loc(points, min_bounds, voxel_size):
    # input: orgin points: N x d
    #        points min bound: d x 1
    #        voxel size: d x 1 size of one voxel in the points space
    # output: voxel unique index M x d
    #        sorted index of orginal points M x 1
    #        start index of the sorted index for each voxel M x 1
    #        points num counts of eah voxel
    #  EXAMPLE:
    #        index 0 -|
    #        index 1  |
    #        index 2  |= > (voxel index 0, start index 0, counts 5)
    #        index 3  |
    #        index 4 -|
    #
    #        index 5 -|
    #        index 6  |= > (voxel index 1, start index 1, counts 3)
    #        index 7 -|

    dim = points.shape[1] if len(points.shape) != 1 else 0
    voxel_index = ((points - min_bounds) / voxel_size).astype(np.int)
    if dim == 0:
        sort_index = np.argsort(points)
    else:
        sort_index = np.lexsort(np.rot90(voxel_index))
    vind_un, start_inds, cnts = np.unique(voxel_index[sort_index],
                                          axis=0, return_index=True, return_counts=True)
    return vind_un, sort_index, start_inds, cnts


class SparseVoxel(object):
    def __init__(self, points=None, voxel_size=None, min_bounds=None):
        self.sparse_COO = _empty()
        self.voxel_size = _empty()
        self.min_bounds = None
        self.values = _empty((2,1))
        self.sorted_pts_index = _empty()

        if points is not None and voxel_size is not None:
            self.create_from_points(points=points,
                                    voxel_size=voxel_size,
                                    min_bounds=min_bounds)

    def initialize(self):
        #prepare slice
        if isinstance(self.sparse_COO, sparse.COO):
            shape = self.sparse_COO.shape
            tmp_sv = self[:-1, :-1, :-1]
            tmp_sv.toindex()

    def create_from_points(self, points, voxel_size, min_bounds=None):
        if min_bounds is None:
            min_bounds = np.min(points, axis=0)
        self.min_bounds = min_bounds
        self.voxel_size = voxel_size
        voxel_indices, self.sorted_pts_index, start_ind, cnts = \
            voxel_map_point_loc(points, self.min_bounds, self.voxel_size)
        self.values = np.asarray([start_ind, cnts])
        self.sparse_COO = sparse.COO(coords=voxel_indices.transpose(),
                                     data=np.arange(len(cnts)),
                                     has_duplicates=False,)

    def create_from_indexed_sparse_voxel(self, indices, ori_sparse_voxel):
        self.__dict__.update(ori_sparse_voxel.__dict__)
        self.values = ori_sparse_voxel.values[:, indices]

    def voxelize(self, points):
        if self.min_bounds is None:
            return points
        else:
            return ((points - self.min_bounds) / self.voxel_size).astype(np.int)

    def __getitem__(self, item):
        sparse_voxel = self.sparse_COO[item]
        sv = SparseVoxel()
        sv.sparse_COO = sparse_voxel
        sv.create_from_indexed_sparse_voxel(sparse_voxel.data, self)
        return sv

    def toindex(self, indices=None):
        if self.sparse_COO.shape[0] == 0: return _empty()
        if indices is not None:
            start_ind = self.values[0, indices]
            cnts = self.values[1, indices]
        else:
            start_ind = self.values[0]
            cnts = self.values[1]
        return multi_slice_indexing(self.sorted_pts_index, start_ind, cnts)

    def __sizeof__(self):
        return sys.getsizeof(self.sorted_pts_index) + sys.getsizeof(self.values) + self.sparse_COO.nbytes

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return object.__getattribute__(self, item)
        else:
            return getattr(self.sparse_COO, item)

if __name__ == "__main__":

    import time
    import open3d

    s = np.array([1,1,1])
    num = 100
    pts = np.asarray(open3d.io.read_point_cloud("/home/lixk/data/dataset/nus3d_pcd/FOE.pcd").points)
    V = SparseVoxel(points=pts, voxel_size=s)
    print(sys.getsizeof(V))
    d = 6
    sp_i = np.random.randint(0, pts.shape[0], num)

    tic = time.time()
    V.initialize()
    print(time.time() - tic)


    tic = time.time()
    for i in range(num):
        pt = pts[sp_i[i]]
        ind = V.voxelize(pt)
        v = V[ind[0] - d:ind[0] + d, ind[1] - d:ind[1] + d]
        indexx = v.toindex()
    print((time.time() - tic) / num)


    tic = time.time()
    for i in range(num):
        pt = pts[sp_i[i]]
        ind = V.voxelize(pt)
        v = V[ind[0] - d:ind[0] + d, ind[1] - d:ind[1] + d]
        indexx = v.toindex()
    print((time.time() - tic) / num)

    tic = time.time()
    for i in range(num):
        pt = pts[sp_i[i]]
        ind = V.voxelize(pt)
        v = V[ind[0] - d:ind[0] + d, ind[1] - d:ind[1] + d]
        indexx = v.toindex()
    print((time.time() - tic) / num)

    tic = time.time()
    for i in range(num):
        pt = pts[sp_i[i]]
        ind = V.voxelize(pt)
        v = V[ind[0] - d:ind[0] + d, ind[1] - d:ind[1] + d]
        indexx = v.toindex()
    print((time.time() - tic) / num)