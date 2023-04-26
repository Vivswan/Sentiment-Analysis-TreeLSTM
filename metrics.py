import torch
import torch.nn as nn


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        # hack cai nay cho no thanh accuracy
        x = torch.tensor(predictions)
        y = torch.tensor(labels)
        x -= x.mean()
        x /= x.std()
        y -= y.mean()
        # label is a list, not tensor
        y /= y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = torch.tensor(predictions)
        y = torch.tensor(labels)
        return nn.MSELoss()(x, y).data[0]

    def sentiment_accuracy_score(self, predictions, labels, fine_gained=True):
        correct = (predictions == labels).sum()
        total = labels.size(0)
        acc = float(correct) / total
        return acc
