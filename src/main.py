import argparse
import random

import pandas as pd
import torch
from data_helpers import *
from flota import *
from sklearn.metrics import f1_score
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, required=True, help='Name of model.')
    parser.add_argument('--data', default=None, type=str, required=True, help='Name of data.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    parser.add_argument('--base', default=False, action='store_true', help='Use base tokenization.')
    parser.add_argument('--flota', default=False, action='store_true', help='Use FLOTA.')
    parser.add_argument('--first', default=False, action='store_true', help='Use FIRST.')
    parser.add_argument('--longest', default=False, action='store_true', help='Use LONGEST.')
    parser.add_argument('--hs', default=False, action='store_true', help='Hyperparameter search.')
    parser.add_argument('--strict', default=False, action='store_true', help='Use strict FLOTA.')
    parser.add_argument('--noise', default=None, type=str, required=False, help='Type of noise.')
    parser.add_argument('--k', default=None, type=int, required=False, help='Number of subwords.')
    args = parser.parse_args()

    train = pd.read_csv('../data/{}_train.csv'.format(args.data))
    train_dataset = ClassificationDataset(train)
    dev = pd.read_csv('../data/{}_dev.csv'.format(args.data))
    dev_dataset = ClassificationDataset(dev)
    test = pd.read_csv('../data/{}_test.csv'.format(args.data))
    test_dataset = ClassificationDataset(test)

    print('Model: {}'.format(args.model))
    print('Data: {}'.format(args.data))
    print('Number of classes: {}'.format(train_dataset.n_classes))
    print('Batch size: {:02d}'.format(args.batch_size))
    print('Learning rate: {:.0e}'.format(args.lr))

    filename = '{}_{}'.format(args.model, args.data)

    if args.flota:
        print('Using FLOTA...')
        filename += '_flota_{}'.format(args.k)
        tok = FlotaTokenizer(args.model, args.k, args.strict, 'flota')
    elif args.first:
        print('Using first subwords...')
        filename += '_first_{}'.format(args.k)
        tok = FlotaTokenizer(args.model, args.k, args.strict, 'first')
    elif args.longest:
        print('Using longest subwords...')
        filename += '_longest_{}'.format(args.k)
        tok = FlotaTokenizer(args.model, args.k, args.strict, 'longest')
    else:
        print('Using base tokenizer...')
        filename += '_base'
        tok = AutoTokenizer.from_pretrained(args.model, model_max_length=512)
        if args.model == 'gpt2':
            tok.padding_side = 'left'
            tok.pad_token = tok.eos_token

    if args.hs:
        filename += '_hs'

    theta = 0.3

    if args.noise == 'train':
        filename += '_noise_train'
        train_collator = ClassificationCollator(tok, args.base, theta)
        test_collator = ClassificationCollator(tok, args.base, theta)
    elif args.noise == 'test':
        filename += '_noise_test'
        train_collator = ClassificationCollator(tok, args.base, 0)
        test_collator = ClassificationCollator(tok, args.base, theta)
    else:
        train_collator = ClassificationCollator(tok, args.base, 0)
        test_collator = ClassificationCollator(tok, args.base, 0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_collator, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=test_collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_collator)

    best_f1, _, _, _ = get_best('results/{}.txt'.format(filename))
    print('Best F1 so far: {}'.format(best_f1))

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=train_dataset.n_classes)
    if args.model == 'gpt2':
        model.config.pad_token_id = model.config.eos_token_id
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Train model...')
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        for i, (batch_tensors, labels) in enumerate(train_loader):
            input_ids = batch_tensors['input_ids'].to(device)
            attention_mask = batch_tensors['attention_mask'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output[0]
            loss.backward()
            optimizer.step()

        model.eval()
        y_true = list()
        y_pred = list()
        with torch.no_grad():
            for (batch_tensors, labels) in dev_loader:
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                labels = labels.to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                y_true.extend(labels.tolist())
                y_pred.extend(output[1].argmax(dim=1).tolist())

        f1_dev = f1_score(y_true, y_pred, average='macro')

        y_true = list()
        y_pred = list()
        with torch.no_grad():
            for (batch_tensors, labels) in test_loader:
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                labels = labels.to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                y_true.extend(labels.tolist())
                y_pred.extend(output[1].argmax(dim=1).tolist())

        f1_test = f1_score(y_true, y_pred, average='macro')

        print(f1_dev, f1_test, args.lr, epoch)

        with open('results/{}.txt'.format(filename), 'a+') as f:
            f.write('{:.4f}\t{:.4f}\t{:.0e}\t{:02d}\n'.format(f1_dev, f1_test, args.lr, epoch))


if __name__ == '__main__':
    main()
