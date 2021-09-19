import random
import cv2
import os


class Loader(object):
    def __init__(self):
        self.img_path = './dataSet/Basketball/img/'
        self.gt_path = './dataSet/Basketball/'
        self.min_r = 0
        self.max_r = 0

    def get_shape(self):
        img = cv2.imread(self.img_path + os.listdir(self.img_path)[0])
        return [img.shape[1], img.shape[0]]

    def get_gt(self):
        ground_truth = []
        ground_truth_str = open(self.gt_path + 'groundtruth_rect.txt', 'r').readlines()
        for gt in ground_truth_str:
            gt = list(map(int, gt.split(',')))
            ground_truth.append(gt)
        return ground_truth

    def test_sample(self, ground_zero, limit, ran=[10, 20], k=1.0):
        positive_samples = []
        total_sample = 30
        self.min_r = min(ran)
        radius = int(self.min_r * k)
        positive_range = [[ground_zero[0] - radius, ground_zero[0] + radius],
                          [ground_zero[1] - radius, ground_zero[1] + radius]]
        for i in range(total_sample):
            pos_x = random.uniform(positive_range[0][0], positive_range[0][1])
            pos_y = random.uniform(positive_range[1][0], positive_range[1][1])

            positive_samples.append(list(map(int, [pos_x, pos_y])))

        index_list = []
        for i in range(len(positive_samples)):
            if positive_samples[i][0] < 0 or \
                    positive_samples[i][1] < 0 or \
                    positive_samples[i][0] > limit[0] - self.max_r or \
                    positive_samples[i][1] > limit[1] - self.max_r:
                index_list.append(i)
        positive_samples = [positive_samples[i] for i in range(0, len(positive_samples), 1) if i not in index_list]

        return positive_samples
