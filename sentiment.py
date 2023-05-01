import datetime
import gc
import json
import os
from pathlib import Path

import torch.optim as optim

import utils
# CONFIG PARSER
from config import parse_args
# DATASET CLASS FOR SICK DATASET
from dataset import SSTDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# IMPORT CONSTANTS
# NEURAL NETWORK MODULES/LAYERS
from model import *
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import SentimentTrainer
# UTILITY FUNCTIONS
from utils import load_word_vectors
# DATA HANDLING CLASSES
from vocab import Vocab


# lt.monkey_patch()


# MAIN BLOCK
def main():
    global args
    args = parse_args(type=1)
    args.input_dim = 300
    if args.model_name == 'dependency':
        args.mem_dim = 168
    if args.model_name == 'constituency':
        args.mem_dim = 150
    if args.fine_grain:
        args.num_classes = 5  # 0 1 2 3 4
    else:
        args.num_classes = 3  # 0 1 2 (1 neutral)

    args.cuda = args.cuda and torch.cuda.is_available()

    print(args)
    args.data = Path(args.data)

    is_preprocessing_data = False  # let program turn off after preprocess data

    vocab_file = args.data.joinpath('vocab-cased.pth')
    if os.path.isfile(vocab_file):
        vocab = Vocab().load_state_dict(torch.load(vocab_file))
    else:
        vocab = Vocab(filename=args.data.joinpath('vocab-cased.txt'))
        torch.save(vocab.state_dict(), vocab_file)
        is_preprocessing_data = True

    print(f'==> SST vocabulary size : {vocab.size():d} ')

    # train
    train_file = args.data.joinpath(f'sst_train_{args.model_name}_state_dict.pth')
    if os.path.isfile(train_file):
        train_dataset = SSTDataset().load_state_dict(torch.load(train_file))
    else:
        train_dir = args.data.joinpath('train')
        train_dataset = SSTDataset(train_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(train_dataset, train_file.with_name(train_file.name.replace('_state_dict', '')))
        torch.save(train_dataset.state_dict(), train_file)
        is_preprocessing_data = True

    # dev
    dev_file = args.data.joinpath(f'sst_dev_{args.model_name}_state_dict.pth')
    if os.path.isfile(dev_file):
        dev_dataset = SSTDataset().load_state_dict(torch.load(dev_file))
    else:
        dev_dir = args.data.joinpath('dev')
        dev_dataset = SSTDataset(dev_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(dev_dataset, dev_file.with_name(dev_file.name.replace('_state_dict', '')))
        torch.save(dev_dataset.state_dict(), dev_file)
        is_preprocessing_data = True

    # test
    test_file = args.data.joinpath(f'sst_test_{args.model_name}_state_dict.pth')
    if os.path.isfile(test_file):
        test_dataset = SSTDataset().load_state_dict(torch.load(test_file))
    else:
        test_dir = args.data.joinpath('test')
        test_dataset = SSTDataset(test_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(test_dataset, test_file.with_name(test_file.name.replace('_state_dict', '')))
        torch.save(test_dataset.state_dict(), test_file)
        is_preprocessing_data = True

    criterion = nn.NLLLoss()
    # initialize model, criterion/loss_function, optimizer
    model = TreeLSTMSentiment(
        args.cuda, vocab.size(),
        args.input_dim, args.mem_dim,
        args.num_classes, args.model_name, criterion
    )

    embedding_model = nn.Embedding(vocab.size(), args.input_dim)

    if args.cuda:
        embedding_model = embedding_model.cuda()
        model = model.cuda()
        criterion = criterion.cuda()

    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad([{'params': model.parameters(), 'lr': args.lr}], lr=args.lr, weight_decay=args.wd)
    else:
        raise Exception("Invalid optimizer selection: --optim={}".format(args.optim))

    metrics = Metrics(args.num_classes)
    utils.count_param(model)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = args.data.joinpath(f'sst_embed_{args.model_name}.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'glove.840B.300d'))
        print(f'==> GLOVE vocabulary size: {glove_vocab.size():d} ')

        emb = torch.zeros(vocab.size(), glove_emb.size(1))

        for word in vocab.labelToIdx.keys():
            if glove_vocab.get_index(word):
                emb[vocab.get_index(word)] = glove_emb[glove_vocab.get_index(word)]
            else:
                emb[vocab.get_index(word)] = torch.Tensor(emb[vocab.get_index(word)].size()).normal_(-0.05, 0.05)
        torch.save(emb, emb_file)
        is_preprocessing_data = True  # flag to quit
        print('done creating emb, quit')

    if is_preprocessing_data:
        print('done preprocessing data, quit program to prevent memory leak')
        print('please run again')
        quit()

    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()

    # model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)
    embedding_model.state_dict()['weight'].copy_(emb)

    # create trainer object for training and testing
    trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)

    mode = 'DEBUG'
    if mode == 'PRINT_TREE':
        for i in range(0, 1):
            ttree, tsent, tlabel = dev_dataset[i]
            utils.print_tree(ttree, 0)
            print('_______________')
        print('break')
        quit()

    elif mode == "DEBUG":
        for epoch in range(args.epochs):
            dev_loss = trainer.train(dev_dataset, epoch=epoch)
            _, test_pred = trainer.test(test_dataset, epoch=epoch)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            print(f'==> Epoch: {epoch} \t Dev loss: {dev_loss:f} \t Test Accuracy: {test_acc * 100:.3f}%')

    elif mode == "EXPERIMENT":
        accuracies = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for epoch in range(args.epochs):
            train_loss = trainer.train(train_dataset, epoch=epoch)
            dev_loss, dev_pred = trainer.test(dev_dataset, epoch=epoch)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            print()
            print(f'==> Epoch: {epoch} \t Train loss: {train_loss:f} \t Dev Accuracy: {dev_acc * 100:.3f}%')
            print()

            accuracies.append((dev_acc, epoch))
            torch.save(model.state_dict(), f'{args.saved}/{timestamp}_{args.model_name}_model_state_dict_{epoch}.pth')
            torch.save(embedding_model.state_dict(), f'{args.saved}/{timestamp}_{args.model_name}_embedding_state_dict_{epoch}.pth')
            gc.collect()

        # save accuracies to json
        with open(f'{args.saved}/{timestamp}_{args.model_name}_accuracies.json', 'w') as f:
            accuracies = sorted(accuracies, key=lambda x: x[1])
            json.dump(accuracies, f)

        accuracies = sorted(accuracies, key=lambda x: x[0], reverse=True)

        # remove rest of the files except the best one
        for _, epoch in accuracies[2:]:
            Path(f'{args.saved}/{timestamp}_{args.model_name}_model_state_dict_{epoch}.pth').unlink(missing_ok=True)
            Path(f'{args.saved}/{timestamp}_{args.model_name}_embedding_state_dict_{epoch}.pth').unlink(missing_ok=True)

        max_dev, max_dev_epoch = accuracies[0]
        print(f'epoch {accuracies} dev score of {max_dev}')
        print('eva on test set ')

        model.load_state_dict(
            torch.load(f'{args.saved}/{timestamp}_{args.model_name}_model_state_dict_{max_dev_epoch}.pth')
        )
        embedding_model.load_state_dict(
            torch.load(f'{args.saved}/{timestamp}_{args.model_name}_embedding_state_dict_{max_dev_epoch}.pth')
        )

        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
        _, test_pred = trainer.test(test_dataset, epoch=max_dev_epoch)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
        print(f'Epoch with max dev:{max_dev_epoch} | Test Accuracy {test_acc * 100:.3f}%')

    elif mode == "TEST":
        timestamp = "20230425161219"
        epoch = 9

        model_filepath = Path(f'{args.saved}/{timestamp}_{args.model_name}_model_state_dict_{epoch}.pth')
        embedding_filepath = Path(f'{args.saved}/{timestamp}_{args.model_name}_embedding_state_dict_{epoch}.pth')

        if model_filepath is None:
            raise ValueError("No model found")

        if embedding_filepath is None:
            raise ValueError("No embedding model found")

        epoch = int(model_filepath.name.split("_")[-1].replace(".pth", ""))
        model.load_state_dict(torch.load(model_filepath))
        embedding_model.load_state_dict(torch.load(embedding_filepath))
        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)

        _, train_pred = trainer.test(train_dataset, epoch=epoch)
        train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
        print(f'Train Accuracy: {train_acc * 100:.3f}%')

        _, dev_pred = trainer.test(dev_dataset, epoch=epoch)
        dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
        print(f'  Dev Accuracy: {dev_acc * 100:.3f}%')

        _, test_pred = trainer.test(test_dataset, epoch=epoch)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
        print(f' Test Accuracy: {test_acc * 100:.3f}%')

        print()
        print()

        print(f'Train Accuracy: {train_acc * 100:.3f}%')
        print(f'  Dev Accuracy: {dev_acc * 100:.3f}%')
        print(f' Test Accuracy: {test_acc * 100:.3f}%')

    else:
        raise ValueError("Invalid value for 'mode'")


if __name__ == "__main__":
    # # log to console and file
    # logger1 = log_util.create_logger("temp_file", print_console=True)
    # logger1.info("LOG_FILE")  # log using loggerba
    # # attach log to stdout (print function)
    # s1 = log_util.StreamToLogger(logger1)
    # sys.stdout = s1
    # print('_________________________________start___________________________________')
    main()
