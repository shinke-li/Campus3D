from ..data_utils import cal_knn
from ..data_utils import sparse_voxel
import numpy as np


class KnnQuery(object):
    def __init__(self, points, knn_module, set_k, GPU_id=None):
        knn_module = getattr(cal_knn, knn_module)  # importlib.import_module('.'.join(['cal_knn', self.knn_module]))
        self.knn_searcher = knn_module(set_k=set_k, GPU_id=GPU_id)
        self.knn_searcher.train(points)

    def search(self, point, k):
        if len(point.shape) < 2:
            point = np.expand_dims(point, axis=0)
        return self.knn_searcher.search(point, k)


class BlockQuery(object):
    def __init__(self, points, voxel_size, block_size, ignore_bounds=False):
        self.sparse_voxel = sparse_voxel.SparseVoxel(points=points,
                                                     voxel_size=voxel_size,)
        self.sparse_voxel.initialize()
        self.block_size = np.array(block_size)
        self.ignored_fine_bound = ignore_bounds
        self.search_dim_x, self.search_dim_y = self.get_block_search_dim()
        #logger.info('Getting search dictionary')


    def get_block_search_dim(self):
        search_dim = (self.block_size - self.sparse_voxel.voxel_size[:2]) / \
                     self.sparse_voxel.voxel_size[:2] / 2
        new_search_dim = np.ceil(search_dim).astype(np.int)
        if np.sum(new_search_dim != search_dim) == 0:
            self.ignored_fine_bound = True
        return [-1 * new_search_dim[0], new_search_dim[0] + 1], \
               [-1 * new_search_dim[1], new_search_dim[1] + 1]

    def search_candidates(self, center_pt):
        center_vox = self.sparse_voxel.voxelize(center_pt)
        s1 = slice(center_vox[0] + self.search_dim_x[0], center_vox[0] + self.search_dim_x[1])
        s2 = slice(center_vox[1] + self.search_dim_y[0], center_vox[1] + self.search_dim_y[1])

        return self.sparse_voxel[s1, s2].toindex()

    def search(self, center, points):
        candidate_index = self.search_candidates(center)

        if not self.ignored_fine_bound:
            candidate_points = points[candidate_index]
            mask_x = np.logical_and(candidate_points[:, 0] <= center[0] + self.block_size[0] / 2,
                                    candidate_points[:, 0] >= center[0] - self.block_size[0] / 2)
            mask_y = np.logical_and(candidate_points[:, 1] <= center[1] + self.block_size[1] / 2,
                                    candidate_points[:, 1] >= center[1] - self.block_size[1] / 2)
            return candidate_index[np.logical_and(mask_x, mask_y)]
        else:
            return candidate_index

if __name__ == "__main__":
    import open3d
    import sys
    import time
    pnum = 1000000
    s = np.array([1,1,1])
    knn_module = 'SkNN'
    #pts = gen_point_cloud(high=1000, low=1, center_num=20, size=pnum, scale=0.3)
    pts = np.asarray(open3d.io.read_point_cloud("/home/lixk/data/dataset/nus3d_pcd/FOE.pcd").points)
    sp_i = np.random.randint(1, pts.shape[0], 1000)
    knn = KnnQuery(pts, knn_module, 2048)
    tic = time.time()
    for i in range(100):
        _ = knn.search(pts[sp_i[i]], 2048)
    toc = time.time()
    print((toc - tic) / 100)