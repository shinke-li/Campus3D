import numpy as np
from ..data_utils import o3d
from ..data_utils.point_util import PointModifier
from .qurey import BlockQuery, KnnQuery


class _BasicSampler(object):
    """
    Sampler template
    One can create new sampler based on this template

    class Type:
        dataset = FileDataReader
        is_training = Bool
        modify_type = [str,]
        modify_dunc = PointModifier

    __init__(
            dataset: .reader.FileDataReader
            params: AttrDict
            is_training: bool {default: True}
            *args,
            **kwrgs
            )

    sample(index: int, set_random_machine: np.random.RandomState {default: None}, *args, **kwargs)
        Return:
            points_modified, points, labels, colors:
            np.ndarray, np.ndarray, np.ndarray, np.ndarray

    cal_length()
        Return:
            length of sampler: int

    modify_points(points: np.ndarray)
        Return:
            modified_points: np.ndarray

    """

    def __init__(self, dataset, params, is_training=True, *args, **kwargs):
        self.modify_type = ['raw']
        _update_from_config(self, params)
        self.dataset = dataset
        self.is_training = is_training
        self.modify_func = PointModifier(self.modify_type)

    def modify_points(self, points, *args, **kwargs):
        return self.modify_func(points, *args, **kwargs)

    def sample(self, ind, set_random_machine=None, *args, **kwargs):
        points = self.dataset.points[ind]
        labels = self.dataset.labels[ind]
        colors = self.dataset.colors[ind]
        points_modified = self.modify_points(points)
        return points_modified, points, labels, colors

    def cal_length(self):
        return len(self.dataset.points)

    def __len__(self):
        return self.cal_length()


def _update_from_config(obj, cfg):
    for k in obj.__dict__.keys():
        try:
            obj.__dict__[k] = cfg[k.upper()]
        except KeyError:
            raise KeyError("\'{}\' has not been defined in config file".format(k.upper()))
        except Exception as e:
            raise Exception(e)


def _gen_empty_sample(num_points_per_sample, modified_shape, label_shape, full_label=-1):
    return \
        np.zeros((num_points_per_sample, modified_shape), dtype=np.float), \
        np.zeros((num_points_per_sample, 3), dtype=np.float), \
        np.full((num_points_per_sample, label_shape), full_label, dtype=np.int), \
        np.zeros((num_points_per_sample, 3), dtype=np.float)


class BlockSampler(_BasicSampler):
    def __init__(self, dataset, params, is_training=True, seed=0):
        self.num_points_per_sample = 0
        self.box_size_x = 0
        self.box_size_y = 0
        self.sliding_ratio = 0
        self.sparse_thresh = 0
        self.modify_type = ['raw']
        self.ignore_fine_bounds = False
        super(BlockSampler, self).__init__(*[dataset, params, is_training])

        self.__max_sample_try = 10
        self.length = int(len(self.dataset) / self.num_points_per_sample) + 1
        self.random_machine = np.random.RandomState(seed)
        self.x_grid, self.y_grid, self.y_grid = None, None, None
        self.q = BlockQuery(self.dataset.points,
                            np.array([1, 1, 1], dtype=np.int),
                            [self.box_size_x, self.box_size_y],
                            ignore_bounds=self.ignore_fine_bounds)
        self.modify_func = PointModifier(self.modify_type)

        if not self.is_training:
            x, y = self._get_sliding_block_center(sliding_block_ratio=self.sliding_ratio)
            x_grid, y_grid = np.meshgrid(x, y)
            self.x_grid = x_grid.flatten()
            self.y_grid = y_grid.flatten()
            self.z_grid = np.full(len(self.x_grid), self.dataset.min_bounds[2])
            self.length = len(self.x_grid)

        #self.points_colors = self.dataset.points_colors.copy()
        #self.labels = self.dataset.labels.copy()

    def modify_points(self, points, *args, **kwargs):
        return self.modify_func(points,
                                min_bounds=self.dataset.min_bounds,
                                max_bounds=self.dataset.max_bounds,
                                block_size_x=self.box_size_x,
                                block_size_y=self.box_size_y,
                                *args,
                                **kwargs)

    def sample(self, ind, set_random_machine=None, *args, **kwargs):
        random_machine = set_random_machine \
            if set_random_machine is not None else self.random_machine
        points = self.dataset.points.view()
        scene_extract_mask = None
        if self.is_training:
            # block with random center
            # Pick a point, and crop a z-box around
            for _ in range(self.__max_sample_try):
                center_point = points[random_machine.randint(0, len(points))]
                scene_extract_mask = self._extract_block(center_point)
                if not isinstance(scene_extract_mask, list):
                    # Not sparse
                    break
            assert not isinstance(scene_extract_mask, list), "Point cloud too sparse"
        else:
            # block with sliding center
            center_point = np.array([self.x_grid[ind], self.y_grid[ind], self.z_grid[ind]])
            scene_extract_mask = self._extract_block(center_point)
            if isinstance(scene_extract_mask, list):
                # ignore too sparse block
                return _gen_empty_sample(self.num_points_per_sample,
                                         self.modify_func.shape,
                                         self.dataset.labels.shape[1])
        # return [], [], [], [], []

        sample_mask = self._get_fix_random_sample_mask(len(scene_extract_mask), random_machine)
        scene_extract_mask = scene_extract_mask[sample_mask]
        points, colors, labels = self.dataset[scene_extract_mask]
        points_centered = self.modify_points(points)
        return points_centered, points, labels, colors

    def cal_length(self):
        return self.length

    def _get_sliding_block_center(self, sliding_block_ratio):
        x = self.dataset.min_bounds[0]
        x_grid = [x]
        while x < self.dataset.max_bounds[0]:
            x += self.box_size_x * sliding_block_ratio
            x_grid.append(x)

        y = self.dataset.min_bounds[1]
        y_grid = [y]
        while y < self.dataset.max_bounds[1]:
            y += self.box_size_y * sliding_block_ratio
            y_grid.append(y)
        return x_grid, y_grid

    def _get_fix_random_sample_mask(self, points_length, random_machine):
        if points_length - self.num_points_per_sample > 0:
            true_array = np.ones(self.num_points_per_sample, dtype=bool)
            false_array = np.zeros(points_length - self.num_points_per_sample, dtype=bool)
            sample_mask = np.concatenate((true_array, false_array), axis=0)
            random_machine.shuffle(sample_mask)
        else:
            # Not enough points, recopy the data until there are enough points
            sample_mask = np.arange(points_length)
            cat_num = self.num_points_per_sample - len(sample_mask)
            cat_index = random_machine.randint(0, len(sample_mask), cat_num)
            sample_mask = np.concatenate((sample_mask, sample_mask[cat_index]), axis=0)
            random_machine.shuffle(sample_mask)
        return sample_mask

    def _extract_block(self, center_point):
        mask = self.q.search(center_point, self.dataset.points)
        if len(mask) <= self.sparse_thresh * self.num_points_per_sample:
            return []

        return mask


class RandomSampler(_BasicSampler):
    def __init__(self, dataset, params, is_training=True, seed=0, return_index=False):
        self.num_points_per_sample = 0
        self.modify_type = None
        super(RandomSampler, self).__init__(*[dataset, params, is_training])
        self.center = np.array([(self.dataset.max_bounds[0] - self.dataset.min_bounds[0]) / 2,
                                (self.dataset.max_bounds[1] - self.dataset.min_bounds[1]) / 2,
                                self.dataset.min_bounds[2]])
        self.random_machine = np.random.RandomState(seed)
        self.return_index=return_index

        self._infer_seq, self._res_num = self._gen_random_seq()
        self.modify_func = PointModifier(self.modify_type)

    def modify_points(self, points, *args, **kwargs):
        return self.modify_func(points, center=self.center)

    def cal_length(self):
        return int(len(self.dataset) / self.num_points_per_sample) + 1

    def sample(self, ind, set_random_machine=None, *args, **kwargs):
        ind = self._sample_index(ind, set_random_machine)
        if self.return_index:
            empty_pts_, empty_pts, _, empty_clrs = \
                _gen_empty_sample(self.num_points_per_sample,
                                  self.modify_func.shape,
                                  self.dataset.labels.shape[1])
            return empty_pts_, empty_pts, ind, empty_clrs
        else:
            points, colors, labels = self.dataset[ind]
            points_centered = self.modify_points(points)
            return points_centered, points, labels, colors

    def _gen_random_seq(self):
        seq = np.random.permutation(len(self.dataset))
        res = len(self) * self.num_points_per_sample - len(self.dataset)
        seq = np.concatenate([seq, seq[:res]])
        return seq, res

    def _get_train_index(self, set_random_machine=None):
        random_machine = self.random_machine \
            if set_random_machine is not None else set_random_machine
        return random_machine.permutation(len(self.dataset.points))[:self.num_points_per_sample]

    def _get_infer_index(self, ind):
        seq_ind = np.arange(ind * self.num_points_per_sample,
                            (ind + 1) * self.num_points_per_sample)
        return self._infer_seq[seq_ind]

    def _sample_index(self, ind, set_random_machine=None):
        if self.is_training:
            ind = self._get_train_index(set_random_machine)
        else:
            ind = self._get_infer_index(ind)
        return ind


class KnnSampler(_BasicSampler):
    def __init__(self, dataset, params, is_training=True, seed=0, return_index=False):
        self.num_points_per_sample = 0
        self.knn_module = ""
        self.max_workers = 64
        self.overlap_ratio = 1.0
        self.modify_type=None
        super(KnnSampler, self).__init__(*[dataset, params, is_training])

        self.return_index = return_index
        self.random_machine = np.random.RandomState(seed)
        self.q = KnnQuery(self.dataset.points, self.knn_module, set_k=self.num_points_per_sample)
        self.center_list = None if self.is_training else self._gen_center_list()
        self.modify_func = PointModifier(self.modify_type)

    def cal_length(self):
        if self.is_training:
            return int(len(self.dataset) / self.num_points_per_sample) + 1
        else:
            return int(len(self.dataset) / self.num_points_per_sample / self.overlap_ratio)

    def modify_points(self, points, *args, **kwargs):
        return self.modify_func(points, center=kwargs['center_point'])

    def sample(self, ind, set_random_machine=None, *args, **kwargs):
        random_machine = self.random_machine if set_random_machine is None else set_random_machine
        ind, center_point = self._sample_index(ind, random_machine)
        if not self.return_index:
            points, colors, labels = self.dataset[ind]
            points_centered = self.modify_points(points, center=center_point)
            #print(points_centered.shape)
            return points_centered, points, labels, colors
        else:
            empty_pts_, empty_pts, _, empty_clrs = \
                _gen_empty_sample(self.num_points_per_sample,
                                  self.modify_func.shape,
                                  self.dataset.labels.shape[1])
            return empty_pts_, empty_pts, ind, empty_clrs

    def _sample_center_index(self, ind, random_machine):
        return random_machine.randint(0, len(self.dataset))

    def _sample_index(self, ind, random_machine):
        if self.is_training:
            center_index = self._sample_center_index(ind, random_machine)
            center_point = self.dataset.points[center_index]
        else:
            center_point = self.center_list[ind]
        _, neighbour_ind = self.q.search(np.expand_dims(center_point, axis=0),
                                        self.num_points_per_sample)
        neighbour_ind = neighbour_ind[0]
        random_machine.shuffle(neighbour_ind)
        return neighbour_ind, center_point

    def _gen_center_list(self):
        centers = o3d.voxel_sampling(self.dataset.points, voxel_size=7)
        self.random_machine.shuffle(centers)
        if len(centers) >= len(self):
            return centers[:len(self)]
        else:
            res_len = len(self)-len(centers)
            random_centers_index = self.random_machine.randint(0, len(self.dataset), res_len)
            return np.concatenate([centers,  self.dataset.points[random_centers_index]])

