import numpy as np
import traceback

class ConfusionMatrix(object):
    def __init__(self, label_range):
        self.confusion_matrix = np.zeros((len(label_range), len(label_range)),
                                         dtype=np.int)
        self.label_range = label_range

    def update(self, pred, target):
        update_m = self.cal_iou_matrix(pred, target, label_range=self.label_range)
        self.confusion_matrix += update_m

    @staticmethod
    def cal_iou_matrix(pred, target, label_range=[]):
        '''
        :param pred:
        :param target:
        :param label_range:
        :return: matrix: label size x label size
                            target
                            l1 l2 l3 l4 l5
                        l1
                result  l2
                        l3
                        l4
                        l5
        '''
        IouMetric._check_shape(pred, target)
        if pred.shape >= 2:
            pred = pred.flatten()
            target = target.flatten()
        label_size = len(label_range)
        point_size = pred.shape[0]
        matrix = np.zeros((label_size, label_size), dtype=np.int64)
        label_repeat = np.tile(np.array(label_range), (point_size, 1)).astype(np.int64).transpose()
        pred_i = np.tile(pred , (label_size, 1)) == label_repeat
        target_i = np.tile(target, (label_size, 1)) == label_repeat
        for i, l in enumerate(label_range):
            new_target_i = np.tile(target_i[i, :], (label_size, 1))
            matrix[:, i] = np.sum(pred_i * new_target_i, axis=1)
        return matrix


class HierarchicalConsistency(object):
    @staticmethod
    def cal_consistency_proportion_per_point(h_matrix, labels):
        num_pts = labels.shape[0]
        leaf_length = h_matrix.classes_num[-1]
        all_score = np.zeros((num_pts, leaf_length), dtype=np.int)
        for i in range(leaf_length):
            one_path = h_matrix.all_valid_h_label[i]
            all_score[:, i] = np.sum(labels == one_path, axis=1)
        return np.max(all_score, axis=1) / float(h_matrix.layer_num)

    @staticmethod
    def cal_consistency_rate(h_matrix, labels, cp_thresh=1.0):
        cp = HierarchicalConsistency.cal_consistency_proportion_per_point(h_matrix, labels)
        return np.sum(cp >= cp_thresh) / float(labels.shape[0])

class AccuracyMetric:
    def __init__(self, label_range):
        self.confusion_matrix = ConfusionMatrix(label_range)
        # np.zeros((len(label_range), len(label_range)),
        #        dtype=np.int)
        self.label_range = label_range

    def __repr__(self):
        return 'Confusion Matrix \n :{}\n'.format(str(self.confusion_matrix))

    def overall_accuracy(self):
        return self.matrix2OA(self.confusion_matrix)

    def avg_accuracy(self):
        return self.matrix2AA(self.confusion_matrix)

    def update(self, pred, target):
        self.confusion_matrix.update(pred, target)

    @staticmethod
    def cal_oa(pred, target):
        IouMetric._check_shape(pred, target)
        return np.sum(pred==target) / float(pred.shape[0])

    @staticmethod
    def matrix2OA(matrix):
        total_num = np.sum(matrix)
        return np.trace(matrix) / float(total_num)

    @staticmethod
    def matrix2AA(matrix):
        total_num = np.sum(matrix, axis=0)
        per_class_acc = np.diagonal / total_num
        return np.mean(per_class_acc), per_class_acc

    @staticmethod
    def _check_shape(pred, target):
        try:
            assert pred.shape == target.shape
        except AssertionError:
            raise ValueError('Shapes of {} and {} are not matched'.format(pred.shape, target.shape))
        except Exception as e:
            traceback.print_exc()


class IouMetric:
    def __init__(self, label_range):
        self.confusion_matrix = ConfusionMatrix(label_range)
        #np.zeros((len(label_range), len(label_range)),
                                 #        dtype=np.int)
        self.label_range = label_range

    def __repr__(self):
        return 'Confusion Matrix \n :{}\n'.format(str(self.confusion_matrix))

    def iou(self):
        return self.matrix2iou(self.confusion_matrix.confusion_matrix)

    def avg_iou(self):
        return self.matrix2avg_iou(self.confusion_matrix.confusion_matrix)

    def update(self, pred, target):
        #update_m = self.cal_iou_matrix(pred, target, label_range=self.label_range)
        #self.confusion_matrix += update_m
        self.confusion_matrix.update(pred, target)

    @staticmethod
    def _check_shape(pred, target):
        try:
            assert pred.shape == target.shape
        except AssertionError:
            raise ValueError('Shapes of {} and {} are not matched'.format(pred.shape, target.shape))
        except Exception as e:
            traceback.print_exc()

    @staticmethod
    def cal_iou(pred, target, label_range=[]):
        IouMetric._check_shape(pred, target)
        iou = []
        for l in label_range:
            pi = (pred==l)
            ti = (target==l)
            i = np.sum(pi*ti)
            u = (np.sum((pi + ti) != 0))
            iou_l = float(i) / float(u) if u != 0 else -1.0
            iou.append(iou_l)
        return np.asarray(iou)

    @staticmethod
    def cal_avg_iou(pred, target, label_range=[]):
        IouMetric._check_shape(pred, target)
        iou = IouMetric.cal_iou(pred, target, label_range)
        return IouMetric._average_iou(iou)


    @staticmethod
    def matrix2iou(matrix):
        size = matrix.shape[0]
        iou = []
        for j in range(size):
             i = matrix[j, j]
             u = np.sum(matrix[j, :]) + np.sum(matrix[:, j]) - i
             iou_one = float(i) / float(u) if u != 0 else -1.0
             iou.append(iou_one)
        return np.asarray(iou)

    @staticmethod
    def matrix2avg_iou(matrix):
        iou = IouMetric.matrix2iou(matrix)
        return IouMetric._average_iou(iou)

    @staticmethod
    def _average_iou(iou):
        mask = iou != -1
        if np.sum(mask) == 0:
            return np.sum(mask).astype(np.float)
        return np.sum(iou[mask]) / np.sum(mask).astype(np.float)

def iou_compare(best_mean_iou, best_class_iou, mean_iou, class_iou):
    if len(best_mean_iou) == 0:
        return True
    if len(mean_iou) <= 1:
        return mean_iou[0] > best_mean_iou
    ratio = 0.67
    bmi = np.asarray(best_mean_iou)
    mi = np.asarray(mean_iou)
    mi_ratio = np.sum(bmi <= mi).astype(float) / float(len(bmi))
    mean_iou_flag = mi_ratio > ratio
    all_iou_flag = []
    for i in range(len(class_iou)):
        bai = np.asarray(best_class_iou[i])
        ai = np.asarray(class_iou[i])
        ai_ratio = np.sum(bai <= ai).astype(float) / float(len(bai))
        iou_flag = ai_ratio > ratio
        all_iou_flag.append(int(iou_flag))
    all_iou_flag = float(sum(all_iou_flag)) / float(len(all_iou_flag))
    if all_iou_flag or mean_iou_flag:
        return True
    else:
        return False

if __name__ == "__main__":
    pred = np.random.randint(0, 5, 100)
    label = np.random.randint(0, 5, (100,5))
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
    from dataset.reader import read_h_matrix_file_list
    h_matrix = read_h_matrix_file_list('test/matrix_file_list.yaml')
    cr = HierarchicalConsistency.cal_consistency_rate(h_matrix, label,)
    print(cr)