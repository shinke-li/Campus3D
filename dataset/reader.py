import os
import numpy as np
import logging
import yaml
from .data_utils.o3d import read_point_cloud
from os.path import basename, splitext
logging.basicConfig()
logger = logging.getLogger(__name__)

#More formats reader

def read_file_list(list_file, sets, use_color, color_channel=(), remove_zero_label=True,):
    with open(list_file, 'r') as yf:
        all_list = yaml.load(yf, Loader=yaml.FullLoader)
    #read data
    main_path = all_list['MAIN_DATA_PATH']
    data_file_list = [d['DATA_PATH']['DATA'] for d in all_list[sets]]
    data_file_list = [os.path.join(main_path, f) for f in data_file_list]
    prefixes = [splitext(basename(p))[0] for p in data_file_list]
    assert len(prefixes) == len(set(prefixes)), \
        'File basename cannot be same.'

    #read label
    main_path = all_list['MAIN_DATA_PATH']
    label_file_list = [d['DATA_PATH']['LABEL'] for d in all_list[sets]]
    label_file_list = [[os.path.join(main_path, f) for f in fl] for fl in label_file_list]

    reader_list = []
    for data_f, label_ps, prefix in zip(data_file_list, label_file_list, prefixes):
        reader_list.append(FileDataReader(data_f, label_ps,
                                          prefix, use_color,
                                          color_channel, remove_zero_label))
    return reader_list

def read_h_matrix_file_list(list_file, add_zero=True):
    with open(list_file, 'r') as yf:
        file_list = yaml.load(yf, Loader=yaml.FullLoader)['file_list']
    return HierarchicalMatrixReader(files=file_list, add_zero=add_zero)


class FileDataReader:
    def __init__(
            self,
            data_path,
            label_paths,
            prefix,
            use_color,
            color_channel=(),
            remove_zero_label=True,
    ):
        """
        Read file format data
        data should be save in such format:
            data_path + prefix + data_ext (ext should be in (.pcd, .ply, .npy))
        label should be saved in such format:
            label_path + prefix + postfix_1 + label_ext (ext should be in (.npy, .txt, .pts))
                                            :
            label_path + prefix + postfix_n + label_ext (ext should be in (.npy, .txt, .pts))

        class Type:
            region = str
            points = np.ndarray
            colors = np.ndarray
            labels = np.ndarray
            min_bounds = np.ndarray
            max_bounds = np.ndarray
            scene_z_size = np.float64
            label_distribution = np.ndarray (label_distribution[i]= point # of i class)

        __init__(
                data_path: str
                label_path: str
                prefix: str
                data_ext: str
                label_ext: str
                label_file_postfixes: [str, ]
                use_color: bool
                color_channel: tuple {default: ()}
                remove_zero_label: bool {default: True}
                )
        """

        self.region = prefix
        self.points_colors = None #speed up indexing
        self.points = None
        self.colors = None
        self.labels = None
        self.min_bounds = None
        self.max_bounds = None
        self.scene_z_size = None
        self.label_distribution = None

        self.__valid_file_types = ['.pcd', '.ply', '.pts', '.npy', ]
        self.__valid_label_types = ['.npy', '.txt', '.labels']
        self.__color_channel = color_channel
        self.__use_color = use_color
        self.__remove_zero = remove_zero_label

        # Load points
        self.check_valid(data_path, label_paths)
        logger.info('Reading data from prefix \'{}\''.format(prefix))
        self.read_point_cloud(data_path)
        logger.info('Reading label from prefix \'{}\''.format(prefix))
        self.read_label(label_paths)
        logger.info('Prefix \'{}\' point number: {}, label categories: {}'. \
                format(prefix, len(self.points), len(label_paths)))
        # Calculate label distribution
        self.label_distribution = self.get_label_dist()
        if self.__remove_zero:
            # Remove label zero
            self.remove_zero_label()
        self.points_colors = np.concatenate([self.points, self.colors], axis=1)

    def check_valid(self, data_path, label_paths):
        data_ext = splitext(basename(data_path))[-1]
        if data_ext not in self.__valid_file_types:
            err = '{} is not valid data file format'.format(data_path)
            raise IOError(err)
        for label_path in label_paths:
            label_ext = splitext(basename(label_path))[-1]
            if label_ext not in self.__valid_label_types:
                err = '{} is not valid label file format'.format(label_path)
                raise IOError(err)

    def read_point_cloud(self, data_path):
        data_ext = splitext(basename(data_path))[-1]
        if data_ext in ['.pcd', '.ply', 'pts']:
            return self.__read_point_cloud_open3d(data_path)
        elif data_ext == ".npy":
            return self.__read_point_cloud_numpy(data_path, self.__color_channel)

    def read_label(self, label_paths, **load_args):
        label_list = [[] for _ in range(len(label_paths))]
        for i, label_path in enumerate(label_paths):
            label_ext = splitext(basename(label_path))[-1]
            if label_ext == '.npy':
                label_list[i] = np.load(label_path)
            elif label_ext in ["labels", ".txt"]:
                label_list[i] = np.loadtxt(label_path, **load_args)
            assert len(label_list[i]) == self.points.shape[0], \
                        "Labels dimension and points dimension are not matched."
        if len(label_list) == 0:
            self.labels = np.asarray([np.zeros(len(self.points)).astype(np.int)])
            self.__remove_zero = False
        else:
            self.labels = np.asarray(label_list).transpose()

    def __read_point_cloud_open3d(self, file_):
        points, colors, self.min_bounds, self.max_bounds = \
            read_point_cloud(file_, use_color=self.__use_color, return_bounds=True)
        self.points_colors = np.concatenate([points, colors], axis=1)
        self.points = self.points_colors[:, :3] #memory view
        self.colors = self.points_colors[:, 3:] #memory view
        self.scene_z_size = self.max_bounds[2] - self.min_bounds[2]

    def __read_point_cloud_numpy(self, file_, color_channel):
        pts_array = np.load(file_)
        points = pts_array[:, :3]
        if self.__use_color:
            colors = pts_array[:, np.asarray(color_channel)]
        else:
            colors = np.zeros_like(self.points)
        self.points_colors = np.concatenate([points, colors], axis=1)
        self.points = self.points_colors[:, :3] #memory view
        self.colors = self.points_colors[:, 3:] #memory view

        self.min_bounds = np.amin(self.points, axis=0)
        self.max_bounds = np.amax(self.points, axis=0)
        self.scene_z_size = self.max_bounds[2] - self.min_bounds[2]

    def get_label_dist(self):
        label_dist = []
        for i in range(self.labels.shape[1]):
            label_dist.append(np.bincount(self.labels[:,i]))
        return label_dist

    def remove_zero_label(self):
        index = np.prod(self.labels != 0, axis=1).astype(np.bool_)

        self.points = self.points[index]
        self.colors = self.colors[index]
        self.labels = self.labels[index]

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, item):
        points_colors = self.points_colors.take(item, axis=0)
        return points_colors[:, :3], points_colors[:, 3:], self.labels.take(item, axis=0)


class HierarchicalMatrixReader(object):
    def __init__(self, matrices=None, files=None, add_zero=False):
        self.hierarchical_matrices = []
        self.project_matrices = []
        if matrices is not None:
            self.hierarchical_matrices = np.asarray(matrices,
                                                    dtype=np.object)
        elif files is not None:
            self.load_files(files)
        else:
            raise TypeError('Missing required positional argument: \'matirces\' or \'files\'.')
       
        self.layer_num = len(self.hierarchical_matrices)
        self.classes_num = np.array([arr.shape[0] for arr in self.hierarchical_matrices])
        self.sort_matrices(add_zero)
        self.all_valid_h_label = self._cal_valid_path()

    def load_files(self, files):
        self.hierarchical_matrices = np.empty((len(files),), dtype=np.object)
        for i, f in enumerate(files):
            m = np.loadtxt(f, delimiter=',')
            self.hierarchical_matrices[i] = m

    def sort_matrices(self, add_zero):
        sort_ind = np.argsort(self.classes_num)
        self.hierarchical_matrices = self.hierarchical_matrices[sort_ind]
        self.classes_num = self.classes_num[sort_ind]
        if add_zero: self._matrices_add_zero_label()
        self._all_label_project()

    def _cal_valid_path(self):
        leaf_length = self.classes_num[-1]
        layer_num = self.layer_num
        all_valid_path = np.zeros((leaf_length, layer_num), dtype=np.int)
        for i in range(leaf_length):
            leaf_label = np.array([i], dtype=np.int)
            for j in range(layer_num):
                all_valid_path[i, j] = self.projet_label(leaf_label, -1, j)
        return all_valid_path

    def get_size(self):
        s = len(self.hierarchical_matrices)
        return (s, s, )

    def _matrices_add_zero_label(self):
        self.classes_num += 1
        for i, m in enumerate(self.hierarchical_matrices):
            m = np.hstack([np.zeros((m.shape[0], 1)), m])
            add_row = np.zeros((1, m.shape[1]))
            add_row[0, 0] = 1
            self.hierarchical_matrices[i] = np.vstack([add_row, m])

    def _project_matrix(self, matrix1, matrix2):
        # M*L1 = L2
        return np.clip(np.dot(matrix1, matrix2.transpose()).transpose(), 0, 1)

    def _all_label_project(self):
        self.project_matrices = np.empty(self.get_size(), dtype=np.object)
        for i in range(self.project_matrices.shape[0]):
            for j in range(self.project_matrices.shape[1]):
                self.project_matrices[i, j] = \
                    self._project_matrix(self.hierarchical_matrices[i],
                                         self.hierarchical_matrices[j])

    def projet_label(self, labels, input_layer, output_layer, mode='num'):
        if mode == 'one_hot':
            return np.dot(labels, self[input_layer, output_layer])
        else:
            return np.argmax(self[output_layer, input_layer][labels], axis=1)

    def __getitem__(self, item):
        return self.project_matrices[item]

    '''
    def cal_HCS(self, pred_label, matrices):
        leaf_length = matrices[0].shape[1]
        num_pts = pred_label.shape[0]
        all_score = np.zeros((num_pts, leaf_length), dtype=np.int)
        for i in range(leaf_length):
            label = np.zeros((leaf_length,), dtype=np.float)
            label[i] = 1
            root_labels = [np.where(np.dot(m, label))[0][0] for m in matrices]
            root_labels = np.array(root_labels).astype(np.int)
            all_score[:, i] = np.sum(pred_label == root_labels, axis=1)
        return np.max(all_score, axis=1) / float(len(matrices))
    '''

if __name__ == '__main__':
    list_file =  [ '/home/lixk/code/campusnet/data/l2.csv',
    '/home/lixk/code/campusnet/data/l5.csv',
     '/home/lixk/code/campusnet/data/l8.csv',
     '/home/lixk/code/campusnet/data/l14.csv',
     '/home/lixk/code/campusnet/data/l3.csv']
    HM = HierarchicalMatrixReader(files=list_file, add_zero=True)
    print(HM.classes_num,)
    print( HM.hierarchical_matrices[1].shape)
    print( np.argmax(HM.hierarchical_matrices[1], axis=0))
    print( np.argmax(HM.hierarchical_matrices[2], axis=0))
    print(np.argmax(HM[3,2], axis=0))
    print(HM.all_valid_h_label.shape)
    print(HM[3,2])
    