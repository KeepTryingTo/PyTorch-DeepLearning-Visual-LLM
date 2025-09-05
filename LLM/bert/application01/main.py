"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/9/2-21:12
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
import torch.nn as nn
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
    该任务主要是判断句子语法的可接收性，将在语言可接受性语料库(The Corpus of Linguistic
    Aoceptability,CoLA)上进行训练。下游任务取自这篇论文：Neural Network Acceptability
    Judgments:论文的作者是 Alex Warstadt、Amanpreet Singh 和 Samuel R. Bowman。
    将微调一个 BERT 模型，该模型将确定句子的语法可接受性。
    
    马修斯（Matthews） 相关系数（MCC）：适用于 类别不平衡的数据集 二分类任务 的 评估指标
    https://blog.csdn.net/u013172930/article/details/146156848
    
    bert-base-uncased模型下载：
    https://huggingface.co/google-bert/bert-base-uncased/tree/main
"""

#TODO 指定输入句子打最大长度
MAX_LEN = 128
batch_size = 8
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadDataset():
    df = pd.read_csv("in_domain_train.tsv", delimiter='\t',
                     header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])


    #@ Creating sentence, label lists and adding Bert tokens
    sentences = df.sentence.values

    # TODO Adding CLS and SEP tokens at the beginning and end of each sentence for BERT
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.label.values

    return sentences, labels

def load_tokenizer(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    #TODO 将每个句子都转换为token表示（这里还是文本的表示方式，还没有转换为对应id）
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print ("Tokenize the first sentence:")
    print (tokenized_texts[0])

    return tokenizer,tokenized_texts

def getInputs(tokenizer,tokenized_texts):
    # TODO Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # TODO Pad our input tokens，根据给定的一个句子最大长度进行填充，并采用后部去掉多余和后部填充
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    return input_ids

def split_train_val():
    sentences, labels = loadDataset()
    tokenizer, tokenized_texts = load_tokenizer(sentences)
    input_ids = getInputs(tokenizer, tokenized_texts)

    attention_masks = []

    #TODO Create a mask of 1s for each token followed by 0s for padding
    #TODO 生成一个句子掩码，true的地方是有token，false的地方表示填充的，用于后面学习
    for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)


    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)


    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    return train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks

def packetDataset():
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because,
    # unlike a for loop,with an iterator the entire dataset does not need to be loaded into memory
    train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks = split_train_val()

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader

try:
    import transformers
except:
    print("Installing transformers")


def loadModel():
    from transformers import BertModel, BertConfig
    configuration = BertConfig()
    #TODO  Initializing a model from the bert-base-uncased style configuration
    model = BertModel(configuration)
    # TODO Accessing the model configuration
    configuration = model.config
    print(configuration)
    #TODO 我们这里做的是预测当前句子是否为下一个句子，二分类问题
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = nn.DataParallel(model)
    model.to(device)

    return model

train_dataloader, validation_dataloader = packetDataset()
model = loadModel()
# This code is taken from:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102

def load_optimizer():
    # Don't apply weight decay to any parameters whose names include these tokens.
    # (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    # Separate the `weight` parameters from the `bias` parameters.
    # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
    # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
    optimizer_grouped_parameters = [
        #TODO  Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.1},

        # TODO Filter for parameters which *do* include those.
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=2e-5,
    #                      warmup=.1)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                      )
    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives
    # us the number of batches.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    return optimizer, scheduler

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

optimizer, scheduler = load_optimizer()

def train():
    t = []

    # Store our loss and accuracy for plotting
    train_loss_set = []

    for _ in trange(epochs, desc="Epoch"):
        #TODO  Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs['loss']
            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()

            scheduler.step()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))


        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = logits['logits'].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    plot(train_loss_set)

def plot(train_loss_set):
    # @title Training Evaluation
    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.show()

def val():
    sentences, _ = loadDataset()
    tokenizer,_ = load_tokenizer(sentences=sentences)
    # @title Predicting and Evaluating Using the Holdout Dataset
    df = pd.read_csv("out_of_domain_dev.tsv", delimiter='\t', header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])

    # Create sentence and label lists
    sentences = df.sentence.values

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.label.values

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    MAX_LEN = 128

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    batch_size = 8

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    model.eval()

    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = logits['logits'].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        #TODO 保存每一个batch里面句子的预测结果
        predictions.append(logits)
        true_labels.append(label_ids)


    from sklearn.metrics import matthews_corrcoef
    matthews_set = []

    for i in range(len(true_labels)):
      matthews = matthews_corrcoef(true_labels[i],
                     np.argmax(predictions[i], axis=1).flatten())
      matthews_set.append(matthews)


    # TODO Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
    #TODO 装预测每一个句子的结果展平成一维向量
    flat_predictions = [item for sublist in predictions for item in sublist]
    #TODO 针对每一个二分类的结果找到预测概率最大对应的索引
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    matthews_corrcoef(flat_true_labels, flat_predictions)


if __name__ == '__main__':
    train()
    val()

