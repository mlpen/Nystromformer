
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple
import os
import sys
import pickle
import time

curr_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

from encoders.backbone import BackboneWrapper

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        self.token_type_embeddings = nn.Embedding(config["num_sen_type"], config["embedding_dim"])

        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)
        torch.nn.init.normal_(self.token_type_embeddings.weight, std = 0.02)

        self.has_dense = config["embedding_dim"] != config["dim"]
        if self.has_dense:
            self.dense = nn.Linear(config["embedding_dim"], config["dim"])

        self.norm = nn.LayerNorm(config["dim"])
        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def forward(self, input_ids, position_ids, type_ids):
        X_token = self.word_embeddings(input_ids)
        X_pos = self.position_embeddings(position_ids)
        X_seq = self.token_type_embeddings(type_ids)
        X = X_token + X_pos + X_seq
        if self.has_dense:
            X = self.dense(X)
        X = self.dropout(self.norm(X))
        return X

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = Embeddings(config)
        self.backbone = BackboneWrapper(config)

        self.pooler = nn.Linear(config["dim"], config["embedding_dim"])
        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def forward(self, inputs):

        def select_index(x, index):
            batch_size = x.size()[0]
            step_size = x.size()[1]
            dim = x.size()[2]
            batch_idx = torch.arange(start = 0, end = batch_size, device = x.device)
            global_index = index + batch_idx[:, None] * step_size
            selected = torch.index_select(x.reshape(-1, dim), 0, global_index.reshape(-1))
            selected = torch.reshape(selected, [batch_size, -1, dim])
            return selected

        masked_sentence = inputs["masked_sentence"].to(torch.int64)
        pos_ids = inputs["pos_ids"].to(torch.int64)
        segment_ids = inputs["segment_ids"].to(torch.int64)
        sentence_mask = inputs["sentence_mask"].to(torch.float32)
        mask_pos_ids = inputs["mask_pos_ids"].to(torch.int64)

        X = self.embeddings(masked_sentence, pos_ids, segment_ids)

        X = self.backbone(X, sentence_mask)

        token_out = select_index(X, mask_pos_ids)
        sentence_out = torch.tanh(self.pooler(X[:, 0, :]))
        sentence_out = self.dropout(sentence_out)

        return token_out, sentence_out

class MLMHead(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        self.dense = nn.Linear(config["dim"], config["embedding_dim"])
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(config["embedding_dim"])
        self.mlm_class = nn.Linear(config["embedding_dim"], config["vocab_size"])
        self.mlm_class.weight = embeddings.weight

    def forward(self, X):

        X = self.act(self.dense(X))
        X = self.norm(X)
        scores = self.mlm_class(X)

        return scores


class ModelWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.vocab_size = config["vocab_size"]
        self.num_classes = config["num_classes"]
        self.model = Model(config)
        self.mlm = MLMHead(config, self.model.embeddings.word_embeddings)
        self.sen_class = nn.Linear(config["embedding_dim"], self.num_classes)

        self.init_sen_class()

        self.output_sen_pred = False
        if "output_sen_pred" in config:
            self.output_sen_pred = config["output_sen_pred"]

        self.num_valid_classes = self.num_classes
        if "num_valid_classes" in config:
            assert config["num_valid_classes"] <= self.num_classes
            self.num_valid_classes = config["num_valid_classes"]

    def init_sen_class(self):
        torch.nn.init.zeros_(self.sen_class.bias)
        torch.nn.init.normal_(self.sen_class.weight, std = 0.02)

    def test_prediction(self, inputs):
        _, sentence_out = self.model(inputs)
        sent_scores = self.sen_class(sentence_out)
        return (sent_scores.argmax(dim = -1)).to(torch.float32)

    def forward(self, inputs):

        mask_label = inputs["mask_label"].to(torch.int64)
        label_mask = inputs["label_mask"].to(torch.float32)

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            token_out, sentence_out = self.model(inputs)

            mlm_scores = self.mlm(token_out)
            sent_scores = self.sen_class(sentence_out)

            if self.num_valid_classes != self.num_classes:
                sent_scores = sent_scores[:, :self.num_valid_classes]

            valid_count = torch.sum(label_mask) + 1e-6
            batch_size = torch.tensor(sent_scores.size(0), dtype = torch.float, device = sent_scores.device)

            mlm_loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
            mlm_loss = mlm_loss_fct(mlm_scores.reshape(-1, self.vocab_size), mask_label.reshape(-1))
            mlm_loss = mlm_loss * label_mask.reshape(-1)
            mlm_loss = torch.sum(mlm_loss) / valid_count

            mlm_correct = (mlm_scores.argmax(dim = -1) == mask_label).to(torch.float32)
            mlm_accu = torch.sum(mlm_correct * label_mask) / valid_count

            if "sentence_label" in inputs:
                sentence_label = inputs["sentence_label"].to(torch.int64)

                sen_loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
                sen_loss = sen_loss_fct(sent_scores, sentence_label)
                sen_loss = torch.mean(sen_loss)

                sen_correct = (sent_scores.argmax(dim = -1) == sentence_label).to(torch.float32)
                sen_accu = torch.mean(sen_correct)

                total_loss = mlm_loss + sen_loss

                outputs = {
                    "mlm_loss":mlm_loss, "mlm_accu":mlm_accu,
                    "sen_loss":sen_loss, "sen_accu":sen_accu,
                    "total_loss":total_loss,
                    "valid_count":valid_count, "batch_size_per_device":batch_size
                }
            else:
                total_loss = mlm_loss

                outputs = {
                    "mlm_loss":mlm_loss, "mlm_accu":mlm_accu,
                    "total_loss":total_loss,
                    "valid_count":valid_count, "batch_size_per_device":batch_size
                }

            outputs = {key:value[None] for key, value in outputs.items()}

        if not self.output_sen_pred:
            return outputs
        else:
            return outputs, sent_scores.argmax(dim = -1)
