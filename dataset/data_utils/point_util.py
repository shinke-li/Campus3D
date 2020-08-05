import numpy as np


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

class PointModifier(object):
    """
    Collections of point modifying methods
    Add modifying fucntion as this:
        @staticmethod
        def _funcname(points, arg=None, **kwargs):
            new_points = some_func(point, arg, )
            return new_points
    Then the modify type will be 'funcname'

    __init__(modify_type:(str,))

    __call__(points: np.ndarray, *args, **kwargs):
        Return:
             modified points: np.ndarray

    """

    def __init__(self, modify_types=('global_normalization', 'block_centeralization')):
        self.funcs = [getattr(self, '_' + m) for m in modify_types]
        self.shape = len(modify_types) * 3

    def __call__(self, points, *args, **kwargs):
        points_list = []
        for i, func in enumerate(self.funcs):
            arg = args[i] if i < len(args) else None
            points_list.append(func(points, arg, **kwargs))
        return np.concatenate(points_list, axis=-1)

    @staticmethod
    def _centeralization(points, arg=None, **kwargs):
        if arg is None:
            arg = kwargs['center']
        return points - arg

    @staticmethod
    def _global_normalization(points, arg=None, **kwargs):
        if arg is None:
            arg = (kwargs['max_bounds'], kwargs['min_bounds'])
        min_bounds, max_bounds = arg
        bounds = max_bounds - min_bounds
        return (points - min_bounds) / bounds

    @staticmethod
    def _block_centeralization(points, arg=None, **kwargs):
        if arg is None:
            arg = (kwargs['block_size_x'], kwargs['block_size_y'])
        block_size_x, block_size_y = arg
        box_min = np.min(points, axis=0)
        shift = np.array([box_min[0] + block_size_x/ 2,
                          box_min[1] + block_size_y / 2,
                          box_min[2]])
        return points - shift

    @staticmethod
    def _raw(points, arg, **kwargs):
        return points