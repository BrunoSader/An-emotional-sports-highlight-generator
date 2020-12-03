from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
from tqdm import tqdm
import progressbar
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=3)
#model.load_state_dict(torch.load('audio/trained_models/camembert_weights.pth'))
#model.eval()

def flat_accuracy(preds, labels):  # A function to predict Accuracy
    correct = 0
    for i in range(0, len(labels)):
        if (preds[i] == labels[i]):
            correct += 1
    return (correct / len(labels)) * 100


def sentiment_accuracy(preds, labels):  # A function to predict Accuracy by sentiment
    acc_neg = 0
    acc_neut = 0
    acc_pos = 0
    neg = 0
    neut = 0
    pos = 0
    for i in range(0, len(labels)):
        if labels[i] == 0:
            neg += 1
            if preds[i] == labels[i]:
                acc_neg += 1
        if labels[i] == 1:
            neut += 1
            if preds[i] == labels[i]:
                acc_neut += 1
        if labels[i] == 2:
            pos += 1
            if preds[i] == labels[i]:
                acc_pos += 1
    return (acc_neg / neg) * 100, (acc_neut / neut) * 100, (acc_pos / pos) * 100

def evaluate(model, test_loader, device, flat=True):
    model.eval()  # Testing our Model
    acc = []
    lab = []
    t = 0
    for inp, lab1 in tqdm(test_loader):
        inp.to(device)
        lab1.to(device)
        t += lab1.size(0)
        outp1 = model(inp.to(device))
        [acc.append(p1.item()) for p1 in torch.argmax(outp1[0], axis=1).flatten()]
        [lab.append(z1.item()) for z1 in lab1]
    if flat:
        print("Total Examples : {} Accuracy {}".format(t, flat_accuracy(acc, lab)))
    else:
        acc_neg, acc_neut, acc_pos = sentiment_accuracy(acc, lab)
        print("Accuracy by sentiment : neg {}, neut {}, pos {}".format(acc_neg, acc_neut, acc_pos))
    return flat_accuracy(acc, lab)


def test_file(w_path, test_loader, device, model, flat=True):
    model.load_state_dict(torch.load(w_path))
    model.eval()
    acc = []
    lab = []
    t = 0
    pred = []
    for inp, lab1 in tqdm(test_loader):
        inp.to(device)
        lab1.to(device)
        t += lab1.size(0)
        outp1 = model(inp.to(device))
        [acc.append(p1.item()) for p1 in torch.argmax(outp1[0], axis=1).flatten()]
        [lab.append(z1.item()) for z1 in lab1]
        for p1 in torch.argmax(outp1[0], axis=1).flatten():
            if p1.item() == 0:
                pred.append('negative')
            elif p1.item() == 1:
                pred.append('neutral')
            elif p1.item() == 2:
                pred.append('positive')
    if flat:
        print("Total Examples : {} Accuracy {}".format(t, flat_accuracy(acc, lab)))
        return flat_accuracy(acc, lab)
    else:
        acc_neg, acc_neut, acc_pos = sentiment_accuracy(acc, lab)
        print("Total Examples : {} Accuracy {}".format(t, flat_accuracy(acc, lab)))
        print("Accuracy by sentiment : neg {}, neut {}, pos {}".format(acc_neg, acc_neut, acc_pos))
        conf_mat = confusion_matrix(lab, acc)
        true_neg = conf_mat[0][0]
        true_neut = conf_mat[1][1]
        true_pos = conf_mat[2][2]
        false_neg = conf_mat[1][0] + conf_mat[2][0]
        false_neut = conf_mat[0][1] + conf_mat[2][1]
        false_pos = conf_mat[0][2] + conf_mat[1][2]
        print("True negative: {}, True neutral: {}, True positive: {}".format(true_neg, true_neut, true_pos))
        print("False negative: {}, False neutral: {}, False positive: {}".format(false_neg, false_neut, false_pos))
        print("Precision negative: {}, Precision neutral: {}, Precision positive: {}".format(
            true_neg / (true_neg + false_neg), true_neut / (true_neut + false_neut), true_pos / (true_pos + false_pos)))
        return pred

def analyse(test_loader, device, model):
    pred = []
    for loader in progressbar.progressbar(test_loader, prefix="FR sentiments: "):
        inp, label = loader
        outp1 = model(inp.to(device))
        for p1 in torch.argmax(outp1[0], axis=1).flatten():
            if p1.item() == 0:
                pred.append('negative')
            elif p1.item() == 1:
                pred.append('neutral')
            elif p1.item() == 2:
                pred.append('positive')

    return pred

def predict(data_to_predict, prediction_key, batch_size=1):
    data=None
    if type(data_to_predict) is str:
        data = pd.read_json(data_to_predict)
    elif type(data_to_predict) is list:
        data = pd.DataFrame.from_dict(data_to_predict)
    else:
        raise TypeError("Unexpected type for data_to_predict: {}".format(type(data_to_predict).__name__))

    sentences = [sent for sent in data[prediction_key]]

    tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
    tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]

    ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

    labels = np.zeros(len(ids))

    # Getting max len in order to pad tokenized ids
    max1 = len(ids[0])
    for i in ids:
        if (len(i) > max1):
            max1 = len(i)
    # print(max1)
    MAX_LEN = max1

    input_ids2 = pad_sequences(ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    Xtest = torch.tensor(input_ids2)
    Ytest = torch.tensor(labels)

    test_data = TensorDataset(Xtest, Ytest)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    data['sentiment'] = analyse(test_loader, device, model)

    return data.to_dict(orient="records")

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-a", "--analyse", action="store_true")
    parser.add_argument("-o", "--output", type=str, default="camembert_data_out.json")
    parser.add_argument("-op", "--option", type=str, default="text")
    parser.add_argument("-f", "--file", type=str, default="audio/Speech_classification.json")
    parser.add_argument("-e", "--epoch", type=int, default=10)
    args = parser.parse_args()

    DATA_PATH = Path('data/')
    LOG_PATH = Path('logs/')
    MODEL_PATH = Path('model/')
    LABEL_PATH = Path('labels/')

    if args.test:
        test_data = pd.read_json(args.file)
        test_data = test_data[test_data.sentiment != "N/A"]
        # test_data = test_data[test_data.sentiment != "neutral"]
        test_data = test_data[test_data.sentiment != "mixed"]
        sentences = []
        for sentence in test_data[args.option]:
            sentences.append(sentence)

        tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
        tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]

        ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

        labels = test_data['sentiment'].astype('category').cat.codes.values

        # Getting max len in order to pad tokenized ids
        max1 = len(ids[0])
        for i in ids:
            if (len(i) > max1):
                max1 = len(i)
        # print(max1)
        MAX_LEN = max1

        input_ids2 = pad_sequences(ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        Xtest = torch.tensor(input_ids2)
        Ytest = torch.tensor(labels)
        batch_size = 1
        test_dataset = TensorDataset(Xtest, Ytest)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=3)
        pred = test_file('camembert_weights.pth', test_loader, device, model,
                         flat=False)
        test_data['pred'] = pred
        # data = pd.concat([test_data, test_data], ignore_index=True, axis=1)
        test_data.to_json(args.output, orient='index', date_format='iso')
    elif args.analyse:
        data = pd.read_json(args.file)
        # data = data.sample(frac=1) #Shuffle the dataset
        sentences = []
        for sentence in data[args.option]:
            sentences.append(sentence)

        tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
        tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]

        ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

        labels = np.zeros(len(ids))

        # Getting max len in order to pad tokenized ids
        max1 = len(ids[0])
        for i in ids:
            if (len(i) > max1):
                max1 = len(i)
        # print(max1)
        MAX_LEN = max1

        input_ids2 = pad_sequences(ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        Xtest = torch.tensor(input_ids2)
        Ytest = torch.tensor(labels)
        batch_size = 1
        test_data = TensorDataset(Xtest, Ytest)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=3)
        data['pred'] = analyse(test_loader, device, model)

        data.to_json(args.output, orient='index', date_format='iso')
    else:
        data_train = pd.read_json(args.file)
        #data_train = data_train[['preproc_text', 'sentiment', 'full_text']]
        data_train = data_train[data_train.sentiment != "N/A"]
        # data_train = data_train[data_train.sentiment != "neutral"]
        data_train = data_train[data_train.sentiment != ""]
        # print(len(data_train['sentiment']))

        sentences = []
        for sentence in data_train[args.option]:
            sentences.append(sentence)

        tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
        tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]

        ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

        data_train['sentiment_cat'] = data_train['sentiment'].astype('category').cat.codes
        labels = data_train['sentiment_cat'].values

        # Getting max len in order to pad tokenized ids
        max1 = len(ids[0])
        for i in ids:
            if (len(i) > max1):
                max1 = len(i)
        # print(max1)
        MAX_LEN = max1

        input_ids2 = pad_sequences(ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        xtrain, xtest, ytrain, ytest = train_test_split(input_ids2, labels, test_size=0.25, random_state=69)

        Xtrain = torch.tensor(xtrain)
        Ytrain = torch.tensor(ytrain)
        Xtest = torch.tensor(xtest)
        Ytest = torch.tensor(ytest)

        batch_size = 3

        train_data = TensorDataset(Xtrain, Ytrain)
        test_data = TensorDataset(Xtest, Ytest)
        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=3)
        #model.load_state_dict(torch.load('camembert_weights.pth'))

        optimizer = AdamW(model.parameters(), lr=4e-5)

        criterion = nn.CrossEntropyLoss()

        no_train = 0
        epochs = args.epoch
        acc = 0
        for epoch in tqdm(range(epochs)):
            model.train()
            loss1 = []
            steps = 0
            train_loss = []
            l = []
            for inputs, labels1 in tqdm(loader):
                inputs.to(device)
                labels1 = labels1.long()
                labels1.to(device)
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                loss = criterion(outputs[0], labels1.to(device)).to(device)
                logits = outputs[0]
                # ll=outp(loss)
                [train_loss.append(p.item()) for p in torch.argmax(outputs[0], axis=1).flatten()]  # our predicted
                [l.append(z.item()) for z in labels1]  # real labels
                loss.backward()
                optimizer.step()
                loss1.append(loss.item())
                no_train += inputs.size(0)
                steps += 1
            print("Current Loss is : {} Step is : {} number of Example : {} Accuracy : {}".format(loss.item(), epoch,
                                                                                                  no_train,
                                                                                                  flat_accuracy(
                                                                                                      train_loss, l)))
            acc_eval = evaluate(model, test_loader, device, True)
            print("Eval accuracy : {}".format(acc_eval))
            if acc - acc_eval < 0:
                acc = acc_eval
                print("saving weights ...")
                torch.save(model.state_dict(), "camembert_weights.pth")
        print("Saving last weights...")
        torch.save(model.state_dict(), "camembert_weights_last.pth")
        print("Best eval accuracy : {}".format(acc))
