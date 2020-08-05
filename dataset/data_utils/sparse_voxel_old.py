
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import time
import numba as nb

@nb.jit(nopython=True)
def allocate(idx, idx_end_id, start_id, cnts, seq):
    for i in range(len(start_id)):
        idx[idx_end_id[i] - cnts[i]:idx_end_id[i]] = seq[start_id[i]: start_id[i] + cnts[i]]


def multi_slice_indexing2(seq, start_id, cnts, *args):
    if start_id.shape[0] == 0: return _empty()
    idx = np.ones(np.sum(cnts), dtype=np.int)
    idx_end_id = np.cumsum(cnts)
    allocate(idx, idx_end_id, start_id, cnts, seq)
    return idx
DTYPE = np.uint32


class SparseVoxel(object):
    """
    SparseVoxel object create from point set.
    Parameters:
        points: N x D size np.ndarray,
        voxel_size D x 1 size np.ndarray,
        min_bounds (optional): minimum bounds of points

    Returns:
        SparseVoxel object

    Object Properties:
        dim: points dimension
        voxel_size: voxel size in point space
        min_bounds: minimum bounds of points
        indices: sparse voxel indices with size V x D
        values: sparse voxel start index and point counts with size 2 x V

    Functions:
        SparseVoxel obj.toindex(indices(optional)):
            get original index of points inside the sparse voxel

        SparseVoxel obj.voxelize(points):
            get voxel index of point with same parameters of SparseVoxel object

        SparseVoxel obj.size():
            get the size of SparseVoxel object

        SparseVoxel obj.save(save_file):
            save the SparseVoxel in .npz file

        SparseVoxel.load(save_file):
            return the loaded SparseVoxel obj

    """
    def __init__(self, points, voxel_size, min_bounds=None):
        if points is None and voxel_size is None:
            self._sparse_voxel = _SparseVoxel()
        else:
            self._sparse_voxel = sparse_voxelize(points, voxel_size, min_bounds)

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return object.__getattribute__(self, item)
        else:
            return getattr(self._sparse_voxel, item)

    def __getitem__(self, item):
        inds = self._sparse_voxel[item]
        sv = SparseVoxel(None, None)
        if len(inds) != 0:
            sv._sparse_voxel = sliced_voxelize(self._sparse_voxel, inds)
        return sv

    def __len__(self):
        return len(self._sparse_voxel)

    @staticmethod
    def load(load_file):
        npz_file = np.load(load_file)
        sv = SparseVoxel(None, None)
        if len(npz_file['indices']) != 0:
            sv._sparse_voxel = load_voxelize(**npz_file)
        return  sv

    def save(self, save_file):
        save_dict = {p:getattr(self._sparse_voxel, p) for p in
                     ['indices', 'values', '_sorted_pts_index', 'min_bounds', 'voxel_size']}
        np.savez(save_file, **save_dict)



def sparse_voxelize(points, voxel_size, min_bounds=None):
    V = _SparseVoxel()
    V.create_from_points(points, voxel_size, min_bounds)
    return V

def sliced_voxelize(sparse_voxel, index):
    V = _SparseVoxel()

    V.create_from_indexed_voxel(index, sparse_voxel)
    return V

def load_voxelize(**kwargs):
    V = _SparseVoxel()
    V.create_from_arrays(**kwargs)
    return V

def _empty(shape=(1,)):
    return np.tile(np.array([], dtype=np.int), shape)


def multi_slice_indexing3(seq, start_id, cnts, pre_arrange=None):
    # pre_arrange make it a bit more faster (10%)
    process_num = start_id.shape[0]
    if process_num == 0: return _empty()
    #indexed_seq = np.ones((np.sum(cnts),), dtype=np.int)
    idx = np.ones(np.sum(cnts), dtype=np.int)
    idx[np.cumsum(cnts)[:-1]] -= cnts[:-1]
    idx = np.cumsum(idx) - 1 + np.repeat(start_id, cnts)
    return seq[idx] if len(idx) != 0 else _empty()

def multi_slice_indexing(seq, start_id, cnts, pre_arrange=None):
    # pre_arrange make it a bit more faster (10%)
    if start_id.shape[0] == 0: return _empty()
    slices = np.asarray([start_id, start_id + cnts]).transpose().tolist()
    if pre_arrange is not None:
        s = list(map(lambda x: pre_arrange[slice(*x)], slices))
    else:
        s = list(map(lambda x: np.arange(*x), slices))
    ind = np.concatenate(s)
    return seq[ind] if len(ind) != 0 else _empty()


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
    indices = np.arange(points.shape[0])[sort_index]
    return vind_un, indices, start_inds, cnts


class _SparseSlicer1D(object):
    def __init__(self, point_1d, length=None):
        self.point_un, self.ordered_ids, start_idx, cnts \
            = voxel_map_point_loc(point_1d, 0, 1)
        self.start_idx_ = np.append(start_idx, 0)
        self.cnts_ = np.append(cnts, 0)
        # self.point_min = self.point_un[0]
        self.length = length if length is not None else int(self.point_un[-1]) + 1
        self.dense_to_un_idx = self.dense_to_sparse_map()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        un_idx = np.unique(self.dense_to_un_idx[item])
        if un_idx.shape[0] == 0 or \
                (un_idx.shape[0] == 1 and un_idx[0] == len(self.point_un)):
            return _empty()  # empty or no valid index
        if isinstance(item, np.ndarray):
            return self.un_idx_array_to_ids(un_idx)
        elif isinstance(item, (int, np.integer)):
            return self.un_idx_to_ids(un_idx)
        elif isinstance(item, slice):
            if un_idx[-1] == len(self.point_un): un_idx[-1] = un_idx[-2]
            return self.un_idx_slice_to_ids(un_idx[0], un_idx[-1])
        else:
            raise TypeError('Index with type {} is invalid.'.format(type(item)))

    def un_idx_to_ids(self, un_idx):
        start = self.start_idx_[un_idx]
        end = start + self.cnts_[un_idx]
        return self.ordered_ids[int(start):int(end)]

    def un_idx_slice_to_ids(self, idx_min, idx_max):
        start = self.start_idx_[idx_min]
        end = start + np.sum(self.cnts_[idx_min:idx_max + 1])
        return self.ordered_ids[int(start):int(end)]

    def un_idx_array_to_ids(self, un_idx_arr):
        start_ind = self.start_idx_[un_idx_arr]
        cnts = self.cnts_[un_idx_arr]
        return multi_slice_indexing(self.ordered_ids, start_ind, cnts)

    def dense_to_sparse_map(self):
        start_mask = np.isin(np.arange(0, len(self)),
                             self.point_un)
        look_up = np.full((len(self),), len(self.point_un), dtype=np.int)
        look_up[start_mask] = np.arange(len(self.point_un))
        return look_up


class _SparseSlicerkD(object): #add check valid in each slicer1d
    def __init__(self, points_kd, point_size=None):
        self.dim = points_kd.shape[1]
        self._size = point_size if point_size is not None \
            else self.get_point_up_bounds(points_kd)
        self.points = points_kd  # * self._mul_shape#same dtype
        self.slicers = self.get_slicers(points_kd)

    def size(self):
        return self._size

    def __len__(self):
        return np.prod(self._size)

    def get_slicers(self, points):
        return [_SparseSlicer1D(points[:, i], self._size[i])
                for i in range(self.dim)]

    def get_point_up_bounds(self, points):
        return tuple((np.max(points, axis=0) + 1).tolist())

    def __getitem__(self, items):
        if not isinstance(items, tuple): items = tuple([items])
        if len(items) > self.dim:
            raise IndexError('Index with dim {} out of dim {}'.format(len(items), self.dim))
        m = map(self.get_voxel_index_one_dim, items, range(len(items)), )
        return self.voxel_index_intersection(*list(m))

    def get_voxel_index_one_dim(self, item, dim):
        return None if isinstance(item, slice) \
                                          and item.start is None \
                                          and item.stop is None else \
            self.slicers[dim][item]

    def voxel_index_intersection(self, *indices):
        not_none_indices = [ind for ind in indices if ind is not None]
        if len(not_none_indices) == 0: return np.arange(len(self))
        ind = not_none_indices[0]
        for ind_ in not_none_indices[1:]:
            ind = np.intersect1d(ind, ind_, assume_unique=True)
        return ind


class _DenseSlicerkD(object):  # start from 0, largest voxel point just 10**4 -10**6
    def __init__(self, points_kd, point_size=None, pre_seq=None):
        self.dim = points_kd.shape[1]
        self._size = point_size if point_size is not None \
            else self.get_point_up_bounds(points_kd)
        self._mul_shape = self.get_shape_multiple()
        self.points = points_kd  # * self._mul_shape#same dtype
        self.map_array = self.gen_map_array(pre_seq)

    def gen_map_array(self, pre_seq=None):
        if pre_seq is None:
            return  [np.arange(s) for s in self._size]
        else:
            return  [pre_seq[:s] for s in self._size]
    def size(self):
        return self._size

    def parse_items(self, items):
        parse = {int: [],
                 slice: [],
                 np.ndarray: [],
                 None: []}
        for i, it in enumerate(items):
            parse[self._type(it)].append(i)
        parse_without_none = [np.asarray(parse[k], dtype=np.int) for k in parse.keys()
                              if k is not None]
        return parse, np.concatenate(parse_without_none)

    def _type(self, s):
        t = type(s)
        if isinstance(s, slice) and s.start is None and s.stop is None:
            t = None
        if isinstance(s, (int, np.integer)):
            t = int
        return t

    def get_point_up_bounds(self, points):
        return tuple((np.max(points, axis=0) + 1).tolist())

    def get_shape_multiple(self):
        cp = np.cumprod(self._size[::-1])
        return np.insert(cp[:-1], 0, 1)[::-1]

    def __len__(self):
        return np.prod(self._size)

    def __getitem__(self, items):
        if not isinstance(items, tuple): items = tuple([items])
        if len(items) > self.dim:
            raise IndexError('Index with dim {} out of dim {}'.format(len(items), self.dim))
        parse, select_dim = self.parse_items(items)
        _, ori_ind, query_ind = np.intersect1d(self.cal_mul_map(select_dim),
                                               self.item_mul_map(parse, items),
                                               return_indices=True)
        return ori_ind

    def cal_mul_map(self, select_dim):
        # return np.sum(self.point_map[:, select_dim], axis=1)
        return np.matmul(self.points[:, select_dim], self._mul_shape[select_dim])

    def item_mul_map(self, parse, items):
        int_items = np.asarray([self.map_array[i][items[i]] for i in parse[int]], dtype=np.int)
        int_map = np.sum(int_items * self._mul_shape[parse[int]])
        # slice_items = [self.map_array[i][items[i]] for i in parse[slice]]
        # arr_items = [self.map_array[i][items[i]] for i in parse[np.ndarray]]
        all_arr = [self.map_array[i][items[i]] for i in parse[np.ndarray] + parse[slice]]
        mesh_grid = np.asarray(np.meshgrid(*all_arr), dtype=np.int)
        mesh_grid_shape = mesh_grid.shape
        r_mesh_grid_shape = tuple([mesh_grid_shape[0], np.prod(np.asarray(mesh_grid_shape[1:]))])
        all_arr_map = np.matmul(self._mul_shape[parse[slice] + parse[np.ndarray]],
                                mesh_grid.reshape(*r_mesh_grid_shape))
        return all_arr_map + int_map


class _SparseVoxel(object):
    DENSE_SPARSE_THRERSH = 300000
    def __init__(self):
        self.dim = 0
        self.voxel_size = _empty()
        self.min_bounds = None
        self.indices = _empty()
        self.values = _empty((2,1))
        self._sorted_pts_index = _empty()
        self._size = tuple([int(0)])
        self._slicers = _empty()
        self._pre_seq = None
        self._offset = _empty()

    def create_from_points(self, points, voxel_size, min_bounds=None):
        if min_bounds is None:
            min_bounds = np.min(points, axis=0)
        self.dim = points.shape[1]
        self.min_bounds = min_bounds
        self.voxel_size = voxel_size
        self.indices, self._sorted_pts_index, start_ind, cnts = \
            voxel_map_point_loc(points, self.min_bounds, self.voxel_size)
        self.values = np.asarray([start_ind, cnts])
        #self._size = np.max(self.indices + 1, axis=0)
        self._slicers = _SparseSlicerkD(self.indices)
        self._pre_seq = np.arange(len(self._sorted_pts_index))
        self._offset = np.zeros((self.dim, ), dtype=np.int)
        self._size = self._slicers.size()

    def create_from_arrays(self, indices, values, _sorted_pts_index, min_bounds, voxel_size):
        self.dim = indices.shape[1]
        self.min_bounds = min_bounds
        self.voxel_size = voxel_size
        self.indices = indices
        self._sorted_pts_index = _sorted_pts_index
        self.values = values
        self._pre_seq = np.arange(len(self._sorted_pts_index))
        self._offset =  np.min(self.indices, axis=0)
        if len(indices.shape[0]) <= self.DENSE_SPARSE_THRERSH:
            self._slicers = _DenseSlicerkD(self.indices - self._offset,
                                          pre_seq=self._pre_seq)
        else:
            self._slicers = _SparseSlicerkD(self.indices - self._offset)
        self._size = self._slicers.size()

    def create_from_indexed_voxel(self, index, ori_sparse_voxel):
        self.__dict__.update(ori_sparse_voxel.__dict__)
        self.indices = self.indices[index]
        self.values = self.values[:, index]
        self._offset = np.min(self.indices, axis=0)
        if len(index) <= self.DENSE_SPARSE_THRERSH:
            self._slicers = _DenseSlicerkD(self.indices - self._offset,
                                          pre_seq=self._pre_seq)
        else:
            self._slicers = _SparseSlicerkD(self.indices - self._offset)
        self._size = self._slicers.size()

    def voxelize(self, points):
        if self.min_bounds is None:
            return points
        else:
            return ((points - self.min_bounds) / self.voxel_size).astype(np.int)

    def size(self):
        return self._size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self._slicers[item]

    def toindex(self, indices=None):
        if len(self) == 0: return _empty()
        if indices is not None:
            start_ind = self.values[0, indices]
            cnts = self.values[1, indices]
        else:
            start_ind = self.values[0]
            cnts = self.values[1]
        return multi_slice_indexing2(self._sorted_pts_index, start_ind, cnts, self._pre_seq)


def gen_gaussian_ball(center, radius, size):
    if not isinstance(radius, np.ndarray):
        radius = np.asarray([radius, radius, radius])
    pts = [np.random.normal(loc=center[i], scale=radius[i], size=size) for i in range(center.shape[0])]
    return np.asarray(pts).transpose()


def gen_point_cloud(high, low, center_num, size, scale=1, dim=3):
    normalized_centers = np.random.rand(center_num, dim)
    centers = (high - low) * normalized_centers + low
    ball_pts_ratio = np.random.rand(center_num, )
    ball_pts_ratio = ball_pts_ratio / np.sum(ball_pts_ratio)
    ball_pts_num = (size * ball_pts_ratio).astype(np.int)
    ball_pts_num[-1] = size - np.sum(ball_pts_num[:-1])
    radius_sum = (high - low) * float(scale)
    radius = radius_sum * ball_pts_ratio

    points = []
    for i in range(center_num):
        points.append(gen_gaussian_ball(centers[i], radius[i], ball_pts_num[i]))
    return np.clip(np.vstack(points), low, high)


'''
class _Slicer1D(object):
    def __init__(self, point_1d, sorted=True):
        self.point_un, self.ids, self.start_ids, self.cnts = voxel_map_point_loc(point_1d, 0, 1)
        self.point_min = self.point_un[0]
        self.point_max = self.point_un[-1]
        self.sorted = sorted

    def get_index_from_range(self, low, high):
        #search the index of low <= point < high
        assert low < high, 'asdasdasdasd.'
        start = self.search_one_point(low)
        stop = self.search_one_point(high, start=start)
        if start == stop: return np.array([], dtype=np.int)
        start_id = self.start_ids[start]
        end_id = start_id + np.sum(self.cnts[start:stop])
        if self.sorted:
            return np.arange(start_id, end_id)
        else:
            return self.ids[start_id:end_id]

    def search_one_point(self, point, start=0):
        start = int(start)
        if point <= self.point_min:
            return 0
        elif point > self.point_max:
            return -1
        else:
            point_id = np.searchsorted(self.point_un[start:], point)
            return point_id + start

class SparseVoxelization(object):
    def __init__(self, points, voxel_size, min_bounds=None, pre_query=True):
        if min_bounds is None:
            min_bounds = np.min(points, axis=0)
        self.min_bounds = min_bounds
        self.voxel_size = voxel_size
        self.voxel_ind, self.sorted_ind, self.start_ind, self.cnts = \
            voxel_map_point_loc(points, self.min_bounds, self.voxel_size)
        self.dim = points.shape[1]
        self.x_slicer = _Slicer1D(self.voxel_ind[:, 0], sorted=True) \
            if pre_query else None
    def __len__(self):
        return len(self.voxel_ind)

    def get_bounds_index(self, min_bound, max_bound):
        if self.x_slicer is not None:
            index = self.x_slicer.get_index_from_range(min_bound[0], max_bound[0])
            mask = self._bound_logical_and_kd(self.voxel_ind[index, 1:],
                                                  min_bound[1:], max_bound[1:])
            return index[mask]
        else:
            mask = self._bound_logical_and_kd(self.voxel_ind,
                                              min_bound, max_bound)
            return np.where(mask)[0]

    def get_voxel_point_index(self, voxel_ind):
        if isinstance(voxel_ind, int): #one voxel
            return self.sorted_ind[self.start_ind[voxel_ind]:self.cnts[voxel_ind]]
        else: #multiple voxel
            start_ind = self.start_ind[voxel_ind]
            cnts = self.cnts[voxel_ind]
            slices = np.asarray([start_ind, start_ind + cnts]).transpose().tolist()
            map(lambda x: np.arange(*x), slices)
            return self.sorted_ind[np.concatenate(slices)] \
                if len(slices) != 0 else np.array([], dtype=np.int)

    def get_bounds_point_index(self, min_bound, max_bound):
        voxel_index = self.get_bounds_index(min_bound, max_bound)
        return  self.get_voxel_point_index(voxel_index)

    def _bound_logical_and_2d(self, points, min_bound, max_bound):
        return (points[:, 0] >= min_bound[0]) & (points[:, 0] < max_bound[0]) \
               & (points[:, 1] >= min_bound[1]) & (points[:, 1] < max_bound[1])

    def _bound_logical_and_3d(self, points, min_bound, max_bound):
        return (points[:, 0] >= min_bound[0]) & (points[:, 0] < max_bound[0]) \
               & (points[:, 1] >= min_bound[1]) & (points[:, 1] < max_bound[1]) \
               & (points[:, 2] >= min_bound[2]) & (points[:, 2] < max_bound[2])

    def _bound_logical_and_kd(self, points, min_bound, max_bound):
        #slower than 2d 3d
        dim = points.shape[1]
        if dim == 2:
            return self._bound_logical_and_2d(points, min_bound, max_bound)
        elif dim == 3:
            return self._bound_logical_and_3d(points, min_bound, max_bound)
        else:
            mask = None
            for i in range(dim):
                if mask is None:
                    mask = (points[:, i] >= min_bound[i]) & \
                           (points[:, i] < max_bound[i])
                else:
                    mask &= (points[:, i] >= min_bound[i]) & \
                            (points[:, i] < max_bound[i])
            return mask

            '''
if __name__ == "__main__":

    class Timer:
        def __init__(self):
            self.cnt = 0
            self.tic = 0
            self.rec = 0

        def start(self):
            self.tic = time.time()

        def end(self):
            self.rec += time.time() - self.tic
            self.cnt += 1

        def __repr__(self):
            if self.cnt != 0:
                return '{} counts: {} seconds'.format(self.cnt, self.rec / self.cnt)
            else:
                return 'not record'


    import open3d
    import sys
    import time
    pnum = 1000000
    s = np.array([1,1,1])
    #pts = gen_point_cloud(high=1000, low=1, center_num=20, size=pnum, scale=0.3)
    pts = np.asarray(open3d.io.read_point_cloud("/home/lixk/data/dataset/nus3d_pcd/FOE.pcd").points)
    print(pts.shape)
    V = SparseVoxel(points=pts, voxel_size=s)
    print(sys.getsizeof(V.values))
    print(sys.getsizeof(V.indices))
    print(V.size())
    print(len(V))

    #2D
    cc = 0
    for i in range(100):
        if i == 1: tic = time.time()
        center_ind = np.random.randint(10, pnum)
        center_vox = V.voxelize(pts[center_ind])
        s1 = slice(center_vox[0] - 5, center_vox[0] + 6)
        s2 = slice(center_vox[1] - 5, center_vox[1] + 6)
        s3 = slice(center_vox[2] - 5, center_vox[2] + 6)
        indices = V[s1, s2, :].toindex()
        cc += len(indices)
        #pt = pts[indices]
        #print(len(indices), np.min(pt, axis=0), np.max(pt, axis=0))
        #print('center', pts[center_ind])
        #newnewV = newV[0:3, 0:3, 0:3]
        #if len(newV) < 8:

            #print('asdasd', len(newV))
    #3D
    print((time.time() - tic)/100)
    print(cc/100)
    #s = np.intersect1d(np.intersect1d(a,b,assume_unique=True),c,assume_unique=True)
