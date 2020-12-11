
import torch
import torch.nn as nn
import math
from model import Model, Approx_GeLU

class MLMHead(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        self.dense = nn.Linear(config["transformer_dim"], config["embedding_dim"])
        self.act = Approx_GeLU(config)
        self.norm = nn.LayerNorm(config["embedding_dim"])
        self.mlm_class = nn.Linear(config["embedding_dim"], config["vocab_size"])
        self.mlm_class.weight = embeddings.weight

    def forward(self, X):

        X = self.act(self.dense(X))
        X = self.norm(X)
        scores = self.mlm_class(X)

        return scores

class ModelForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.vocab_size = config["vocab_size"]
        self.model = Model(config)
        self.mlm = MLMHead(config, self.model.embeddings.word_embeddings)

    def forward(self, input_ids, labels):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            token_out = self.model(input_ids)

            mlm_scores = self.mlm(token_out)

            label_mask = (labels != -100).float()
            valid_count = torch.sum(label_mask) + 1e-6
            batch_size = torch.tensor(mlm_scores.size(0), dtype = torch.float, device = mlm_scores.device)

            mlm_loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
            mlm_loss = mlm_loss_fct(mlm_scores.reshape(-1, self.vocab_size), labels.reshape(-1))
            mlm_loss = torch.sum(mlm_loss * label_mask.reshape(-1)) / valid_count

            mlm_correct = (mlm_scores.argmax(dim = -1) == labels).to(torch.float32)
            mlm_accu = torch.sum(mlm_correct * label_mask) / valid_count

            outputs = {
                "loss":mlm_loss, "mlm_accu":mlm_accu,
                "valid_count":valid_count, "batch_size_per_device":batch_size
            }

            outputs = {key:value[None] for key, value in outputs.items()}

        return outputs

class SequenceClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dropout_0 = nn.Dropout(config["dropout_prob"])
        self.dense = nn.Linear(config["transformer_dim"], config["transformer_dim"])
        self.dropout_1 = nn.Dropout(config["dropout_prob"])
        self.classifer = nn.Linear(config["transformer_dim"], config["num_classes"])

    def forward(self, X):

        sen_out = X[:, 0, :]
        sen_score = self.classifer(self.dropout_1(torch.tanh(self.dense(self.dropout_0(sen_out)))))

        return sen_score

class ModelForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.vocab_size = config["vocab_size"]
        self.num_classes = config["num_classes"]

        self.model = Model(config)
        self.mlm = MLMHead(config, self.model.embeddings.word_embeddings)
        self.sen_classifer = SequenceClassificationHead(config)

    def forward(self, input_ids, attention_mask, labels = None):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            token_out = self.model(input_ids)

            mlm_scores = self.mlm(token_out)
            mlm_correct = (mlm_scores.argmax(dim = -1) == input_ids).to(torch.float32)
            mlm_accu = torch.sum(mlm_correct * attention_mask) / (torch.sum(attention_mask) + 1e-6)

            sent_scores = self.sen_classifer(token_out)

            outputs = {
                "mlm_accu":mlm_accu[None], "sent_scores":sent_scores
            }

            if labels is not None:
                sen_loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
                sen_loss = sen_loss_fct(sent_scores, labels)
                sen_loss = torch.mean(sen_loss)

                sen_correct = (sent_scores.argmax(dim = -1) == labels).to(torch.float32)
                sen_accu = torch.mean(sen_correct)

                outputs["loss"] = sen_loss[None]
                outputs["accu"] = sen_accu[None]

        return outputs


class ModelForQA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.vocab_size = config["vocab_size"]
        self.model = Model(config)
        self.mlm = MLMHead(config, self.model.embeddings.word_embeddings)
        self.qahead = nn.Linear(config["transformer_dim"], 2)

    def forward(self, input_ids, attention_mask, start_positions = None, end_positions = None):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            token_out = self.model(input_ids, attention_mask)

            mlm_scores = self.mlm(token_out)
            mlm_correct = (mlm_scores.argmax(dim = -1) == input_ids).to(torch.float32)
            mlm_accu = torch.sum(mlm_correct * attention_mask) / (torch.sum(attention_mask) + 1e-6)

            preds = self.qahead(token_out)
            assert preds.size(-1) == 2
            start_logits, end_logits = preds[:, :, 0], preds[:, :, 1]

            outputs = {
                "mlm_accu":mlm_accu[None], "start_logits":start_logits, "end_logits":end_logits
            }

            if start_positions is not None and end_positions is not None:
                ignored_index = start_logits.size(1)
                torch.clamp(start_positions, 0, ignored_index)
                torch.clamp(end_positions, 0, ignored_index)

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index = ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

                outputs["loss"] = loss[None]

            return outputs

class ModelForWiKiHop(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enable_amp = config["mixed_precision"]

        self.vocab_size = config["vocab_size"]
        self.model = Model(config)
        self.mlm = MLMHead(config, self.model.embeddings.word_embeddings)
        self.qahead = nn.Linear(config["transformer_dim"], 1)

    def forward(self, input_ids, attention_mask, candidate_mask = None, answer_positions = None):

        with torch.cuda.amp.autocast(enabled = self.enable_amp):

            token_out = self.model(input_ids, attention_mask)

            mlm_scores = self.mlm(token_out)
            mlm_correct = (mlm_scores.argmax(dim = -1) == input_ids).to(torch.float32)
            mlm_accu = torch.sum(mlm_correct * attention_mask.float()) / (torch.sum(attention_mask.float()) + 1e-6)

            logits = self.qahead(token_out)[:, :, 0]
            if candidate_mask is not None:
                logits = logits - 1e9 * (1 - candidate_mask.float())

            outputs = {
                "mlm_accu":mlm_accu[None], "logits":logits
            }

            if answer_positions is not None:
                loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
                loss = loss_fct(logits, answer_positions).mean()
                answer_accu = (logits.argmax(dim = -1) == answer_positions).to(torch.float32).mean()

                outputs["loss"] = loss[None]
                outputs["accu"] = answer_accu[None]

            return outputs
