import math

import numpy as np
import torch


class AveragePrecisionMeter(object):

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):

        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):

        if self.scores.numel() == 0:
            return 0

        ap = torch.full((self.scores.size(1),), -1.)

        non_zero_labels = 0
        non_zero_ap_sum = 0
        for k in range(self.scores.size(1)):
            targets = self.targets[:, k]

            if targets.sum() == 0:
                continue
            non_zero_labels += 1

            scores = self.scores[:, k]

            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
            non_zero_ap_sum += ap[k]

        mAP = non_zero_ap_sum / non_zero_labels

        return mAP, ap.tolist()

    @staticmethod
    def average_precision(output, target, difficult_examples=False):

        sorted, indices = torch.sort(output, dim=0, descending=True)

        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.

        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue

            if label == 1:
                pos_count += 1

            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count

        if pos_count != 0:
            precision_at_i /= pos_count

        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0

            Ng[k] = np.sum(targets == 1)

            Np[k] = np.sum(scores >= 0)

            Nc[k] = np.sum(targets * (scores >= 0))

        if np.sum(Np) == 0:
            OP = -1
            OR = -1
            OF1 = -1
        else:
            OP = np.sum(Nc) / np.sum(Np)
            OR = np.sum(Nc) / np.sum(Ng)
            OF1 = (2 * OP * OR) / (OP + OR)

        CP_SUM = 0
        CP_CNT = 0
        CR_SUM = 0
        CR_CNT = 0
        CP = -1
        CR = -1
        CF1 = -1

        for i in range(n_class):
            if Np[i] != 0:
                CP_CNT += 1
                CP_SUM += Nc[i] / Np[i]
            if Ng[i] != 0:
                CR_CNT += 1
                CR_SUM += Nc[i] / Ng[i]
        if CP_CNT != 0:
            CP = CP_SUM / CP_CNT
        if CR_CNT != 0:
            CR = CR_SUM / CR_CNT
        if CP != -1 and CR != -1:
            CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1
