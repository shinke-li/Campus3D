from __future__ import division
from __future__ import print_function
import numpy as np
import logging
from .reader import read_file_list
from .sampling.sampler import DatasetSampler
from torch.utils import data as D

logging.basicConfig()
logger = logging.getLogger(__name__)

class TorchDataset(D.Dataset):
    def __init__(self, set_name, params, is_training=True):
        #set name: TRAIN_SET, TEST_SET, VALIDATION_SET
        self.scene_data_list = read_file_list(params.DATA.DATA_LIST_FILE,
                              sets=set_name,
                              use_color=params.DATA.USE_COLOR,
                              color_channel=params.DATA.COLOR_CHANNEL,
                              remove_zero_label=params.DATA.REMOVE_ZERO_LABEL,
                              )
        logger.info('Building sampler for each scene.')
        self.data_sampler = DatasetSampler(self.scene_data_list,
                                          params=params,
                                          is_training=is_training,)
        self.is_training = is_training

    def __len__(self):
        return len(self.data_sampler)

    def __getitem__(self, item):
        return self.data_sampler[item]

    @staticmethod
    def np_collate_fn(batch):
        points, labels, colors, weights = zip(*batch)
        return np.asarray(points), np.asarray(labels), np.asarray(colors), np.asarray(weights)

    def __getattr__(self, item):
        if item not in self.__dict__.keys():
            if item == 'random_machine':
                return getattr(self.data_sampler, item)
            try:
                return [getattr(scene_data,item) for scene_data in self.scene_data_list]
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise AttributeError('{} is not attribute of DataSampler'.format(item))


class TorchDataLoader(D.DataLoader):
    def __init__(self, *args, **kwargs):
        super(TorchDataLoader, self).__init__(*args, **kwargs) 
        self.is_training = self.dataset.is_training 
        self.collate_fn = self.dataset.np_collate_fn 
        
    def __iter__(self):
        iterator = iter(super(TorchDataLoader, self).__iter__())
        self.dataset.random_machine.renew()
        if self.is_training:
            return iterator 
        else:
            # remove invalid samples from batch and stuff it with new samples
            return _StuffIterator(iterator, self.batch_size)


class _StuffIterator:
    def __init__(self, iterator, batch_size):
        self.it = iterator
        self.stop = False
        self.batch_size = batch_size
        self.res_batch = [[] for _ in range(4)]
        self.res_num = 0
             
    def _update_res(self, np_data_tuple, ind, initial=False):
        if initial:
            self.res_batch = [[] for _ in range(4)]
        for i in range(len(np_data_tuple)):
            self.res_batch[i].append(np_data_tuple[i][ind])

    def concatenate(self):
        return (np.concatenate(d) for d in self.res_batch)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            while True:
                batch_data = next(self.it)
                # index with label not equal to -1
                full_ind = np.where(batch_data[1][:, 0, 0] >= 0)[0]
                valid_num = len(full_ind)
                if valid_num == 0:
                    continue
                delta_size = self.res_num + valid_num - self.batch_size
                if delta_size == 0:
                    self._update_res(batch_data, full_ind)
                    np_data = self.concatenate()
                    self.res_num = 0
                    self._update_res([], 0, initial=True)
                    return np_data
                elif delta_size > 0:
                    self._update_res(batch_data, full_ind[delta_size:])
                    np_data = self.concatenate()
                    self.res_num = delta_size
                    self._update_res(batch_data, full_ind[:delta_size], initial=True)
                    return np_data
                else:
                    self._update_res(batch_data, full_ind)
                    self.res_num = self.batch_size + delta_size
        except StopIteration:
            if not self.stop:
                self.stop = True
                np_data = self.concatenate()
                return np_data
            else:
                raise StopIteration
        except Exception as e:
            raise Exception(e)

    next = __next__
        
        

            
    

if __name__ == "__main__":
    import yaml
    import sys
    import time
    with open("../configs/default/dataset.yaml", "r") as f:
        p = yaml.load(f)
    print(p)
    ds = TorchDataset(["PGP"], p, is_training=False)

    dd = TorchDataLoader(ds, batch_size=32, num_workers=4)
    #a = Nus3dTorchDataset(["PGP"], p)
    tic = time.time()
    l = len(dd)
    i = 0
    for datas, labels, colors, weights in dd:
        i += 1
        sys.stdout.write("{}/{}\n".format(i,l))

        sys.stdout.flush()
    print("TOTAL", i)
    print(time.time() - tic)
