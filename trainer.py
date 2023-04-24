import torch
from tqdm import tqdm


class SentimentTrainer:
    """
    For Sentiment module
    """

    def __init__(self, args, model, embedding_model, criterion, optimizer):
        super(SentimentTrainer, self).__init__()
        self.args = args
        self.model = model
        self.embedding_model = embedding_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        # torch.manual_seed(789)
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, input, _ = dataset[indices[idx]]
            if self.args.cuda:
                input = input.cuda()
            emb = torch.unsqueeze(self.embedding_model(input), 1)
            output, loss = self.model.forward(tree, emb, training=True)
            # params = self.model.childsumtreelstm.getParameters()
            # params_norm = params.norm()
            loss = loss / self.args.batchsize  # + 0.5*self.args.reg*params_norm*params_norm # custom bias
            loss.backward()

            total_loss += loss
            k += 1
            if k == self.args.batchsize:
                for f in self.embedding_model.parameters():
                    f.data.sub_(f.grad.data * self.args.emblr)
                self.optimizer.step()
                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        total_loss = 0
        predictions = torch.zeros(len(dataset))

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            tree, input, label = dataset[idx]
            target = torch.tensor([label]).type(torch.long)

            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()

            emb = torch.unsqueeze(self.embedding_model(input), 1)

            output, _ = self.model(tree, emb)  # size(1,5)
            total_loss += self.criterion(output, target)

            output[:, 1] = -9999  # no need middle (neutral) value
            val, pred = torch.max(output, 1)
            predictions[idx] = pred.data[0]
            # predictions[idx] = torch.dot(indices, torch.exp(output.data))
        return total_loss / len(dataset), predictions


class Trainer:
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            ltree, linput, rtree, rinput, label = dataset[indices[idx]]
            target = torch.tensor([label]).type(torch.long)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree, linput, rtree, rinput)
            loss = self.criterion(output, target)
            loss.backward()
            total_loss += loss
            k += 1
            if k % self.args.batchsize == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        total_loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.range(1, dataset.num_classes)
        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            ltree, linput, rtree, rinput, label = dataset[idx]
            target = torch.tensor([label]).type(torch.long)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree, linput, rtree, rinput)
            total_loss += self.criterion(output, target)
            predictions[idx] = torch.dot(indices, torch.exp(output.data))
        return total_loss / len(dataset), predictions
