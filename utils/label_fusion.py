import numpy as np

#def heirarchical_ensemble(logits, h_matrices, weight=np.full((5,), 1.0)):
   # pass

def heirarchical_ensemble(logits, h_matrix,  weight=np.full((5,), 1.0)):
    #if not zero_removed: logits = [lgs[:, 1:] for lgs in logits]
    probs_list = [get_transfered_probs(m, lgs) for m, lgs in zip(h_matrix.hierarchical_matrices, logits)]
    path_probs = np.stack(probs_list).transpose(1,2,0).dot(weight)
    leaf_label = np.argmax(path_probs, axis=1)
    return h_matrix.all_valid_h_label[leaf_label]

def cal_label_map(m1, m2):
    # Label1 * M = Label2
    return (np.dot(m2, m1.transpose()) > 0).transpose()

def get_bayes_prob(m, n, probm, energyn):
    t = (np.dot(n, m.transpose()) > 0).transpose()
    prob_proj = np.dot(probm, t)
    t = np.dot(t.transpose(), t)
    energy_normalizer = np.dot(energyn, t)
    return energyn / energy_normalizer * prob_proj

def get_rid_zero_energy(label):
    label = np.squeeze(label)
    return np.exp(label[:, 1:])


def get_transfered_probs(m, logits):
    probs = softmax_probs(logits)
    return np.dot(probs, m)


def softmax_probs(logits):
    # cal softmax for logits
    exp_logits = np.exp(logits)
    label_size = exp_logits.shape[1]
    return exp_logits / np.tile(np.sum(exp_logits, axis=1),
                                (label_size, 1)).transpose()

def remove_zero(gt_label, *pred_labels):
    non_zero_ind = gt_label != 0
    return_labels = []
    for l in pred_labels:
        return_labels.append(l[non_zero_ind])
    return gt_label[non_zero_ind], return_labels

def get_masked_logits(m1, m2, logits1, logits2):
    # label from logits1 as logits2 prior knowledge
    # Mask logits2 with logits1 label
    label1 = np.argmax(logits1, axis=1)
    t = cal_label_map(m1, m2)[label1]
    return t.astype(np.float) * logits2

def label2label(m1, m2, label1):
    t = cal_label_map(m1, m2)[label1]
    if (np.sum(t, axis=1) > 1).any():
        print('Warn: some label may missing')
    return np.argmax(t, axis=1)

def greedy_fusion(logits, r_matrices, zero_removed=False):
    if not zero_removed: logits = [lgs[:, 1:] for lgs in logits]
    logits_result = [logits[0]]
    for i in range(len(r_matrices) - 1):
        logits_result.append(get_masked_logits(r_matrices[i],
                                               r_matrices[i + 1],
                                               logits_result[-1],
                                               logits[i + 1]))
    return np.stack([np.argmax(r, axis=1) for r in logits_result]).transpose()



if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
    from dataset.reader import HierarchicalMatrixReader
    list_file =  [ '/home/lixk/code/campusnet/data/l2.csv',
    '/home/lixk/code/campusnet/data/l5.csv',
     '/home/lixk/code/campusnet/data/l8.csv',
     '/home/lixk/code/campusnet/data/l14.csv',
     '/home/lixk/code/campusnet/data/l3.csv']
    HM = HierarchicalMatrixReader(files=list_file, add_zero=True)
    N = 10
    print(HM.classes_num)
    logits = [np.random.randint(1, 10, (N, cls)) for cls in HM.classes_num]
    print(path_fusion(logits, HM, zero_removed=False, weight=np.full((5,), 1.0)))

