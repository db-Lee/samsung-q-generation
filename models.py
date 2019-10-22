import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_scatter import scatter_max
from transformers import BertModel, BertTokenizer

def return_mask_lengths(ids):
    mask = torch.sign(ids).float()
    lengths = mask.sum(dim=1).long()
    return mask, lengths

def return_num(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def cal_attn(left, right, mask):
    mask = (1.0 - mask.float()) * -10000.0
    attn_logits = torch.matmul(left, right.transpose(-1, -2).contiguous())
    attn_logits = attn_logits + mask
    attn_weights = F.softmax(input=attn_logits, dim=-1)
    attn_outputs = torch.matmul(attn_weights, right)
    return attn_outputs, attn_logits

class Embedding(nn.Module):
    def __init__(self, bert_model):
        super(Embedding, self).__init__()
        bert_embeddings = BertModel.from_pretrained(bert_model).embeddings
        self.word_embeddings = bert_embeddings.word_embeddings
        self.token_type_embeddings = bert_embeddings.token_type_embeddings
        self.position_embeddings = bert_embeddings.position_embeddings
        self.LayerNorm = bert_embeddings.LayerNorm
        self.dropout = bert_embeddings.dropout

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class ContextualizedEmbedding(nn.Module):
    def __init__(self, bert_model):
        super(ContextualizedEmbedding, self).__init__()
        bert = BertModel.from_pretrained(bert_model)
        self.embedding = bert.embeddings
        self.encoder = bert.encoder
        self.num_hidden_layers = bert.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embedding(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        return sequence_output

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, P, Q):
        log_P = P.log()
        log_Q = Q.log()
        kl = (P * (log_P - log_Q)).sum(dim=-1).sum(dim=-1)
        return kl

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=False):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        if dropout > 0.0 and num_layers == 1:
            dropout = 0.0

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, input, input_lengths, state=None):
        batch_size, total_length, _ = input.size()

        input_packed = pack_padded_sequence(input, input_lengths,
                                            batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        output_packed, state = self.lstm(input_packed, state)

        output = pad_packed_sequence(output_packed, batch_first=True, total_length=total_length)[0]
        output = self.dropout(output)

        return output, state

class ContextEncoderforQG(nn.Module):
    def __init__(self, embedding, emsize,
                 nhidden, nlayers,
                 dropout=0.0):
        super(ContextEncoderforQG, self).__init__()
        self.embedding = embedding
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.context_lstm = CustomLSTM(input_size=emsize,
                                       hidden_size=nhidden,
                                       num_layers=nlayers,
                                       dropout=dropout,
                                       bidirectional=True)
        self.context_linear = nn.Linear(2 * nhidden, 2 * nhidden)
        self.fusion = nn.Linear(4 * nhidden, 2 * nhidden, bias=False)
        self.gate = nn.Linear(4 * nhidden, 2 * nhidden, bias=False)

    def forward(self, c_ids, a_ids):
        c_mask, c_lengths = return_mask_lengths(c_ids)
        c_embeddings = self.embedding(c_ids, c_mask, a_ids)
        c_outputs, c_state = self.context_lstm(c_embeddings, c_lengths)

        c_state = (c_state[0].view(self.nlayers, 2, -1, self.nhidden).contiguous(),
                   c_state[1].view(self.nlayers, 2, -1, self.nhidden).contiguous())
        c_state = (c_state[0].transpose(1, 2).contiguous(),
                   c_state[1].transpose(1, 2).contiguous())
        c_state = (c_state[0].view(self.nlayers, -1, 2*self.nhidden).contiguous(),
                   c_state[1].view(self.nlayers, -1, 2*self.nhidden).contiguous())
        # attention
        mask = torch.matmul(c_mask.unsqueeze(2), c_mask.unsqueeze(1))
        c_attned_by_c, _ = cal_attn(self.context_linear(c_outputs),
                                    c_outputs,
                                    mask)
        c_concat = torch.cat([c_outputs, c_attned_by_c], dim=2)
        c_fused = self.fusion(c_concat).tanh()
        c_gate = self.gate(c_concat).sigmoid()
        c_outputs = c_gate * c_fused + (1 - c_gate) * c_outputs
        return c_outputs, c_state

class QuestionDecoder(nn.Module):
    def __init__(self, sos_id, eos_id,
                 embedding, contextualized_embedding, emsize,
                 nhidden, ntokens, nlayers,
                 dropout=0.0,
                 max_q_len=64):
        super(QuestionDecoder, self).__init__()

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.emsize = emsize
        self.embedding = embedding
        self.nhidden = nhidden
        self.ntokens = ntokens
        self.nlayers = nlayers
        # this max_len include sos eos
        self.max_q_len = max_q_len

        self.context_lstm = ContextEncoderforQG(contextualized_embedding, emsize,
                                                nhidden // 2, nlayers, dropout)

        self.question_lstm = CustomLSTM(input_size=emsize,
                                        hidden_size=nhidden,
                                        num_layers=nlayers,
                                        dropout=dropout,
                                        bidirectional=False)

        self.question_linear = nn.Linear(nhidden, nhidden)

        self.concat_linear = nn.Sequential(nn.Linear(2*nhidden, 2*nhidden),
                                           nn.Dropout(dropout),
                                           nn.Linear(2*nhidden, 2*emsize))

        self.logit_linear = nn.Linear(emsize, ntokens, bias=False)

        # fix output word matrix
        self.logit_linear.weight = embedding.word_embeddings.weight
        for param in self.logit_linear.parameters():
            param.requires_grad = False

    def postprocess(self, q_ids):
        eos_mask = q_ids == self.eos_id
        no_eos_idx_sum = (eos_mask.sum(dim=1) == 0).long() * (self.max_q_len - 1)
        eos_mask = eos_mask.cpu().numpy()
        q_lengths = np.argmax(eos_mask, axis=1) + 1
        q_lengths = torch.tensor(q_lengths).to(q_ids.device).long() + no_eos_idx_sum
        batch_size, max_len = q_ids.size()
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len))
        idxes = idxes.unsqueeze(0).to(q_ids.device).repeat(batch_size, 1)
        q_mask = (idxes < q_lengths.unsqueeze(1))
        q_ids = q_ids.long() * q_mask.long()
        return q_ids

    def forward(self, c_ids, q_ids, a_ids):
        batch_size, max_q_len = q_ids.size()

        c_outputs, init_state = self.context_lstm(c_ids, a_ids)

        c_mask, c_lengths = return_mask_lengths(c_ids)
        q_mask, q_lengths = return_mask_lengths(q_ids)

        # question dec
        q_embeddings = self.embedding(q_ids)
        q_outputs, _ = self.question_lstm(q_embeddings, q_lengths, init_state)

        # attention
        mask = torch.matmul(q_mask.unsqueeze(2), c_mask.unsqueeze(1))
        c_attned_by_q, attn_logits = cal_attn(self.question_linear(q_outputs),
                                              c_outputs,
                                              mask)

        # gen logits
        q_concated = torch.cat([q_outputs, c_attned_by_q], dim=2)
        q_concated = self.concat_linear(q_concated)
        q_maxouted, _ = q_concated.view(batch_size, max_q_len, self.emsize, 2).max(dim=-1)
        gen_logits = self.logit_linear(q_maxouted)

        # copy logits
        bq = batch_size * max_q_len
        c_ids = c_ids.unsqueeze(1).repeat(1, max_q_len, 1).view(bq, -1).contiguous()
        attn_logits = attn_logits.view(bq, -1).contiguous()
        copy_logits = torch.zeros(bq, self.ntokens).to(c_ids.device)
        copy_logits = copy_logits - 10000.0
        copy_logits, _ = scatter_max(attn_logits, c_ids, out=copy_logits)
        copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)
        copy_logits = copy_logits.view(batch_size, max_q_len, -1).contiguous()

        logits = gen_logits + copy_logits

        return logits

    def generate(self, c_ids, a_ids):
        c_mask, c_lengths = return_mask_lengths(c_ids)
        c_outputs, init_state = self.context_lstm(c_ids, a_ids)

        batch_size = c_ids.size(0)

        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1)
        q_ids = q_ids.to(c_ids.device)
        token_type_ids = torch.zeros_like(q_ids)
        position_ids = torch.zeros_like(q_ids)
        q_embeddings = self.embedding(q_ids, token_type_ids, position_ids)

        state = init_state

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            position_ids = position_ids + 1
            q_outputs, state = self.question_lstm.lstm(q_embeddings, state)

            # attention
            mask = c_mask.unsqueeze(1)
            c_attned_by_q, attn_logits = cal_attn(self.question_linear(q_outputs),
                                                  c_outputs,
                                                  mask)

            # gen logits
            q_concated = torch.cat([q_outputs, c_attned_by_q], dim=2)
            q_concated = self.concat_linear(q_concated)
            q_maxouted, _ = q_concated.view(batch_size, 1, self.emsize, 2).max(dim=-1)
            gen_logits = self.logit_linear(q_maxouted)

            # copy logits
            attn_logits = attn_logits.squeeze(1)
            copy_logits = torch.zeros(batch_size, self.ntokens).to(c_ids.device)
            copy_logits = copy_logits - 10000.0
            copy_logits, _ = scatter_max(attn_logits, c_ids, out=copy_logits)
            copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)

            logits = gen_logits + copy_logits.unsqueeze(1)

            q_ids = torch.argmax(logits, 2)
            all_q_ids.append(q_ids)

            q_embeddings = self.embedding(q_ids, token_type_ids, position_ids)

        q_ids = torch.cat(all_q_ids, 1)
        q_ids = self.postprocess(q_ids)

        return q_ids

class QG(nn.Module):
    def __init__(self, args):
        super(QG, self).__init__()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        padding_idx = tokenizer.vocab['[PAD]']
        sos_id = tokenizer.vocab['[CLS]']
        eos_id = tokenizer.vocab['[SEP]']
        ntokens = len(tokenizer.vocab)

        bert_model = args.bert_model
        if "large" in bert_model:
            emsize = 1024
        else:
            emsize = 768

        self.dec_q_nhidden = dec_q_nhidden = args.dec_q_nhidden
        self.dec_q_nlayers = dec_q_nlayers = args.dec_q_nlayers
        dec_q_dropout = args.dec_q_dropout

        max_q_len = args.max_q_len

        embedding = Embedding(bert_model)
        contextualized_embedding = ContextualizedEmbedding(bert_model)
        for param in embedding.parameters():
            param.requires_grad = False
        for param in contextualized_embedding.parameters():
            param.requires_grad = False

        self.question_decoder = QuestionDecoder(sos_id, eos_id,
                                                embedding, contextualized_embedding, emsize,
                                                dec_q_nhidden, ntokens, dec_q_nlayers,
                                                dec_q_dropout,
                                                max_q_len)

        self.q_rec_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, c_ids, q_ids, a_ids):

        # question decoding
        q_logits = self.question_decoder(c_ids, q_ids, a_ids)

        # q rec loss
        loss_q_rec = self.q_rec_criterion(q_logits[:, :-1, :].transpose(1, 2).contiguous(),
                                          q_ids[:, 1:])

        loss = loss_q_rec

        return loss

    def generate(self, c_ids, a_ids):

        q_ids = self.question_decoder.generate(c_ids, a_ids)

        return q_ids
