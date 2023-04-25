import datetime
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
    # torch.manual_seed(args.seed)
    # if args.cuda:
    # torch.cuda.manual_seed(args.seed)

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # write unique words from all token files
    # token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
    vocab_file = os.path.join(args.data, 'vocab-cased.txt')  # use vocab-cased
    # build_vocab(token_files, vocab_file) NO, DO NOT BUILD VOCAB,  USE OLD VOCAB

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file)
    print(f'==> SST vocabulary size : {vocab.size():d} ')

    # Load SST dataset splits

    is_preprocessing_data = False  # let program turn off after preprocess data

    # train
    train_file = os.path.join(args.data, f'sst_train_{args.model_name}.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SSTDataset(train_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(train_dataset, train_file)
        is_preprocessing_data = True

    # dev
    dev_file = os.path.join(args.data, f'sst_dev_{args.model_name}.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SSTDataset(dev_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(dev_dataset, dev_file)
        is_preprocessing_data = True

    # test
    test_file = os.path.join(args.data, f'sst_test_{args.model_name}.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SSTDataset(test_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(test_dataset, test_file)
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
        # optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        optimizer = optim.Adagrad([{'params': model.parameters(), 'lr': args.lr}], lr=args.lr, weight_decay=args.wd)
    else:
        raise Exception("Invalid optimizer selection: --optim={}".format(args.optim))
    
    metrics = Metrics(args.num_classes)
    utils.count_param(model)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, f'sst_embed_{args.model_name}.pth')
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

    mode = 'EXPERIMENT'
    if mode == 'PRINT_TREE':
        for i in range(0, 1):
            ttree, tsent, tlabel = dev_dataset[i]
            utils.print_tree(ttree, 0)
            print('_______________')
        print('break')
        quit()
    elif mode == "DEBUG":
        for epoch in range(args.epochs):
            dev_loss = trainer.train(dev_dataset)
            _, test_pred = trainer.test(test_dataset)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            print(f'==> Epoch: {epoch} \t Dev loss: {dev_loss:f} \t Test Accuracy: {test_acc * 100:.3f}%')
    elif mode == "EXPERIMENT":
        max_dev = 0
        max_dev_epoch = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for epoch in range(args.epochs):
            train_loss = trainer.train(train_dataset)
            dev_loss, dev_pred = trainer.test(dev_dataset)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            print(f'==> Epoch: {epoch} \t Train loss: {train_loss:f} \t Dev Accuracy: {dev_acc * 100:.3f}%')

            torch.save(model, f'{args.saved}/{timestamp}_{args.model_name}_model_{epoch}.pth')
            torch.save(embedding_model, f'{args.saved}/{timestamp}_{args.model_name}_embedding_{epoch}.pth')
            if dev_acc > max_dev:
                # remove previous model file
                Path(f'{args.saved}/{timestamp}_{args.model_name}_model_{max_dev_epoch}.pth').unlink(missing_ok=True)
                Path(f'{args.saved}/{timestamp}_{args.model_name}_embedding_{max_dev_epoch}.pth').unlink(missing_ok=True)

                max_dev = dev_acc
                max_dev_epoch = epoch
                
        print(f'epoch {max_dev_epoch} dev score of {max_dev}')
        print('eva on test set ')

        model = torch.load(f'{args.saved}/{timestamp}_{args.model_name}_model_{max_dev_epoch}.pth')
        embedding_model = torch.load(f'{args.saved}/{timestamp}_{args.model_name}_embedding_{max_dev_epoch}.pth')

        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)
        _, test_pred = trainer.test(test_dataset)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)

        # remove rest of the files except the best one
        for epoch in range(args.epochs):
            if epoch == max_dev_epoch:
                continue
        
            Path(f'{args.saved}/{timestamp}_{args.model_name}_model_{epoch}.pth').unlink(missing_ok=True)
            Path(f'{args.saved}/{timestamp}_{args.model_name}_embedding_{epoch}.pth').unlink(missing_ok=True)

        print(f'Epoch with max dev:{max_dev_epoch} | Test Accuracy {test_acc * 100:.3f}%')
    elif mode == "TEST":
        timestamp = ""

        model_filepath = None
        embedding_filepath = None
        # find the model with timestamp
        for file in Path(args.saved).iterdir():
            if "_model_" not in file.name:
                continue
            if not file.name.startswith(f"{timestamp}_"):
                continue
            if file.suffix != ".pth":
                continue

            model_filepath = file
            embedding_filepath = Path(args.saved) / file.name.replace("_model_", "_embedding_")
            break
        
        if model_filepath is None:
            raise ValueError("No model found")
        
        if embedding_filepath is None:
            raise ValueError("No embedding model found")
        
        model = torch.load(model_filepath)
        embedding_model = torch.load(embedding_filepath)
        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer)

        _, train_pred = trainer.test(train_dataset)
        _, dev_pred = trainer.test(dev_dataset)
        _, test_pred = trainer.test(test_dataset)

        train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
        dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
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
