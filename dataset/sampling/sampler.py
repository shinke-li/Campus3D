from .sampler_collections import *
from ..data_utils.random_machine import RandomMachine as RM

class DatasetSampler(object):
    '''
    Sampler for data of scenes
    dataset sampler ->  [sampler_1(scene_2) ... sampler_n(scene_n)]-> randomly select a sampler -> sample

    class Type
        set_name = str
        list_sampler = [Sampler, ]
        is_training = bool
        return_index = bool
        num_classes = [int, ]
        label_distribution = [np.ndarray, ]
        label_weights = [np.ndarray, ]
        random_machine = data_utils.random_machine.RandomMachine

    __init__(
            set_name: str,
            params: AttrDict,
            is_training: bool {default: True},
            )

    __getitem__(index: int)
        Returns:
            points_centered, labels, colors, raw_points:
            np.ndarray, np.ndarray, np.ndarray, np.ndarray

    __getattr__(attr: str)
        Returns:
            list of attribute: [sampler.attr, ]

    '''
    def __init__(self, data_list,  params, is_training=True,):

        # Dataset parameters
        self.is_training = is_training

        self.num_classes = params.DATA.LABEL_NUMBER
        self.return_index = params.SAMPLE.RETURN_INDEX
        self.__sampler_type = params.SAMPLE.SAMPLER_TYPE
        self.__random_scene = params.SAMPLE.RANDOM_SCENE_CHOOSE  # just for training, otherwise traverse all scenes
        self.__params = params

        # create list of sampler for each dataset
        self.list_sampler = self.get_list_sampler(data_list, params)
        self.__length = sum([len(sp) for sp in self.list_sampler])
        self.__num_scenes = len(self.list_sampler)
        self.random_machine = RM(basis_seed=params.SAMPLE.RANDOM_SEED_BASIS, call_length=len(self))
        self.label_distribution = self.merge_label_distribution()
        self.label_weights = self.cal_label_weights(self.label_distribution,
                                                    setting=params.SAMPLE.LABEL_WEIGHT_POLICY)

        # Pre-compute the probability of picking a scene
        self.__scene_probs = [float(len(sp.dataset)) / float(self.get_total_num_points())
                            for sp in self.list_sampler] if self.__random_scene else None
        self.__scene_sample_counter = self.scene_points_counts()

    def get_list_sampler(self, data_list, params):
        sampler_gen = globals()[self.__sampler_type]
        list_sampler = [self.gen_sampler(sampler_gen, data, params.SAMPLE.SETTING)
                        for data in data_list]
        return list_sampler

    def scene_points_counts(self):
        counts = [0]
        for s in self.list_sampler:
            counts.append(counts[-1] + len(s))
        return np.array(counts)[1:]

    def get_traversing_scene_ind(self, ind):
        if ind >= len(self):
            raise IndexError('Out of bounds for axis 0 with size {}'.format(len(self)))
        scene_ind = int(np.argmax(ind < self.__scene_sample_counter))
        ind_in_scene = ind - self.__scene_sample_counter[scene_ind]
        return scene_ind, ind_in_scene

    def gen_sampler(self, gen_func, file_data, setting):
        return gen_func(
            dataset=file_data,
            params=setting,
            is_training=self.is_training,
            #return_index=self.return_index,
        )

    def __getattr__(self, attr):
        return [getattr(sampler.dataset, attr) for sampler in self.list_sampler]

    def get_total_num_points(self):
        list_num_points = [len(sp.dataset) for sp in self.list_sampler]
        return np.sum(list_num_points)

    def merge_label_distribution(self):
        label_dist = []
        for i, num_class in enumerate(self.num_classes):
            one_dist = np.zeros(num_class)
            for sp in self.list_sampler:
                d = sp.dataset.label_distribution[i]
                d = np.pad(d, (0, num_class - len(d)), 'constant')
                one_dist += d
            label_dist.append(one_dist)
        return label_dist

    def _get_random_machine(self, ind):
        return self.random_machine.get_fix_machine(ind)

    @staticmethod
    def cal_label_weights(label_dist, setting="log", log_shift=1.2):
        all_label_weights = []
        if setting == "log":
            for label_weights in label_dist:
                label_weights = label_weights / np.sum(label_weights)
                label_weights = 1 / np.log(log_shift + label_weights)
                all_label_weights.append(label_weights)
        elif setting == "ones":
            for label_weights in label_dist:
                label_weights = np.full(len(label_weights), 1.0)
                all_label_weights.append(label_weights)

        return all_label_weights

    def __len__(self):
        return self.__length

    def __get_scene_index(self, ind, random_machine, enable_random_scene=True):
        if self.__random_scene and enable_random_scene:
            scene_index = random_machine.choice(
                np.arange(0, len(self.list_sampler)),
                p=self.__scene_probs)
        else:
            scene_index, ind = self.get_traversing_scene_ind(ind)
        return scene_index, ind

    def get_label_weights(self, labels):
        weights = [self.label_weights[i][labels[..., i]] for i in range(len(self.label_weights))]
        weights = np.asarray(weights)
        if len(weights.shape) == 1: np.expand_dims(weights, axis=0)
        permut = list(range(1, len(weights.shape)))
        permut.append(0)
        return weights.transpose(*permut)

    def __getitem__(self, ind):
        assert isinstance(ind, int) or isinstance(ind, np.int), \
            'index must be int.'
        random_machine = self._get_random_machine(ind)
        scene_index, ind = self.__get_scene_index(ind, random_machine,
                                                  enable_random_scene=self.is_training)
        points_centered, points, labels, colors = \
            self.list_sampler[scene_index].sample(ind, random_machine)
        if not self.return_index:
            points = self.get_label_weights(labels) if self.is_training else points
            return points_centered, labels, colors,  points
        else:
            return scene_index, labels, None, None


if __name__ == "__main__":
   pass 
