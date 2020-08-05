from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import h5py
import os
import numpy as np


#def open_index_h5(f_name, mode, num_points_per_sample=None):



class open_index_h5(object):
    def __init__(self, f_name, mode, num_points_per_sample = None):
        self.f_name = f_name
        self.mode = mode
        self.num_points_per_sample = num_points_per_sample
        self.saver = None
    def __enter__(self):
        if not (isinstance(self.f_name, str) or isinstance(self.f_name, unicode)):
            raise TypeError("File name should be str, not {}".format(type(self.f_name)))
        if not (isinstance(self.mode, str) or isinstance(self.mode, unicode)):
            raise TypeError("Mode name should be str, not {}".format(type(self.f_name)))
        if self.mode in set(['write', 'w']):
            assert isinstance(self.num_points_per_sample, int), \
                'Number of sample is not indicated.'
            self.saver =  IndexHdf5Writer(self.f_name, self.num_points_per_sample)
        elif self.mode in set(['read', 'r']):
            self.saver = IndexHdf5Reader(self.f_name)
        else:
            raise ValueError("{} is not a valid open mode".format(self.mode))
        return self.saver
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.saver.close()

class IndexHdf5Writer:
    def __init__(self, f_name, num_points_per_sample=0):
        self.file_path = f_name
        self.length = 0
        self.data_shape = (0, num_points_per_sample)
        #uint32 = h5py.special_dtype(vlen=np.dtype('uint32'))
        #uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
        self.h5_file = None

    #def __enter__(self):
        with h5py.File(self.file_path, "w") as f:
            _ = f.create_dataset('index', self.data_shape, dtype='u4',
                                 maxshape=(None, self.data_shape[1]), chunks=True)
            __ = f.create_dataset('scene_index', (self.data_shape[0], ), dtype='u1',
                                  maxshape=(None, ), chunks=True)
        self.h5_file = h5py.File(f_name, 'a')

    def close(self):
        self.h5_file.close()

    def append(self,  scene_index, data_ind):
        data_size = len(scene_index)
        error_type = None
        try:
            self.h5_file['index'].resize(
                (self.h5_file['index'].shape[0] + data_size), axis=0
            )
            self.h5_file['scene_index'].resize(
                (self.h5_file['scene_index'].shape[0] + data_size), axis=0
            )
            self.h5_file['index'][-data_size:] = data_ind
            self.h5_file['scene_index'][-data_size:] = scene_index
        except TypeError as e:
            error_type = TypeError
        except Exception as e:
            error_type = Exception
        if error_type is not None:
            raise error_type(e)
        self.length += data_size


class IndexHdf5Reader:
    def __init__(self, f_name):
        self.file_path = f_name
        self.length = None
        self.data_shape = None
        #uint32 = h5py.special_dtype(vlen=np.dtype('uint32'))
        #uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
        self.h5_file = None

    #def __enter__(self):
        assert os.path.isfile(self.file_path), \
                "{} does not exist".format(self.file_path)
        self.h5_file = h5py.File(self.file_path, 'r')
        self.data_shape = self.h5_file['index'].shape
        self.length = self.data_shape[0]

    def close(self):
        self.h5_file.close()

    #def read(self):
       # ind = self.h5_file['index'][:]
        #scene_ind = self.h5_file['scene_index'][:]
        #return scene_ind, ind

    def __getitem__(self, item):
        # Out of range problem ignore
        #if isinstance(item, slice):
         #   assert item.stop < len(self), \
          #      'Index out of range {}'.format(len(self) - 1)
        error_type = None
        error_traceback = None
        try:
            ind = self.h5_file['index'][item]
            scene_ind = self.h5_file['scene_index'][item]
        except ValueError as e:
            error_type = ValueError
            error_traceback = e
        except Exception as e:
            error_type = Exception
            error_traceback = e

        if error_type is not None:
            raise error_type(error_traceback)
        if isinstance(item, int):
            scene_ind = np.array([scene_ind])
        return scene_ind, ind

    def __len__(self):
        return self.length


if __name__ == "__main__":
    import time
    import numpy as np
    data = np.array([[1,2,3,4], [5,6,7,8], [9, 10, 11, 12]], dtype=np.int)
    si = np.array([1,2,3], dtype=np.int8)
    f_name = 'test.hdf5'
    saver = IndexHdf5Saver(f_name, 4)

    tic = time.time()
    saver.append(data, si)
    print('append: {}'.format(time.time() - tic))

    tic = time.time()
    saver.append(data, si)
    print('append: {}'.format(time.time() - tic))
    tic = time.time()
    print(saver.read_all())
    print('read all: {}'.format(time.time() - tic))
    print(saver[np.array([1])])
    print(len(saver))

