import numpy as np


class RandomMachine(object):
    """
    List of numpy random state
    """

    MAX_SEED = int(10**9)

    def __init__(self, basis_seed=0, call_length=0):
        self.rand_state = np.random.RandomState(basis_seed)
        self.length = int(call_length)
        self.rand_seed_seq = None
        self.update_seed_seq()
        self.list_random_state = self.gen_random_state_list()

    def renew(self):
        update_nums = self.rand_state.randint(self.MAX_SEED, size=(len(self), ))
        for i in range(len(self.list_random_state)):
            self.list_random_state[i].randint(update_nums[i])

    def gen_random_state_list(self):
        return [np.random.RandomState(self.rand_seed_seq[i])
                for i in range(len(self))]

    def update_seed_seq(self):
        self.rand_seed_seq = self.rand_state.randint(self.MAX_SEED, size=(len(self), ))

    def get_fix_machine(self, ind):
        new_seed = self.list_random_state[ind].randint(self.MAX_SEED)
        return np.random.RandomState(new_seed)

    def __len__(self):
        return self.length
