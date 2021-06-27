
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ENAScontroller(nn.Module):
    """
    code adapted from 
    https://github.com/MengTianjian/enas-pytorch/blob/master/micro_controller.py
    """
    def __init__(self, args):
        super(ENAScontroller, self).__init__()
        self.args = args
        self.device = args.device

        # argumnets
        self.branch_num = args.child_branch_num # 5
        self.unit_num = args.child_unit_num # 5
        self.lstm_size = args.controller_lstm_size # 64
        self.lstm_num_layers = args.controller_lstm_layer_num # 1
        self.temperature = args.controller_temperature # 5.0
        self.tanh_constant = args.controller_tanh_constant # 0.44
 
        # skeleton
        self.encoder = nn.Embedding(self.branch_num+1, self.lstm_size)
        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)

        # classifiers
        self.w_soft = nn.Linear(self.lstm_size, self.branch_num, bias=False)
        b_soft = torch.zeros(1, self.branch_num)
        b_soft[:, 0:2] = 10
        self.b_soft = nn.Parameter(b_soft)
        b_soft_no_learn = np.array([0.25, 0.25] + [-0.25] * (self.branch_num-2))
        b_soft_no_learn = np.reshape(b_soft_no_learn, [1, self.branch_num])
        self.b_soft_no_learn = torch.Tensor(b_soft_no_learn).requires_grad_(False).to(self.device)

        # attentions
        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

        self.init_params()
        self.to(self.device)

    def init_params(self):
        for name, param in self.named_parameters():
            if 'b_soft' not in name:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self):
        # convolutional cell
        arc_seq_1, entropy_1, log_prob_1, c, h = self.run_sampler(use_bias=True)
        # reduction cell
        arc_seq_2, entropy_2, log_prob_2, _, _ = self.run_sampler(prev_c=c, prev_h=h)

        sample_arc = (arc_seq_1, arc_seq_2)
        sample_entropy = entropy_1 + entropy_2
        sample_log_prob = log_prob_1 + log_prob_2

        return sample_arc, sample_log_prob, sample_entropy

    def run_sampler(self, prev_c=None, prev_h=None, use_bias=False):
        if prev_c is None:
            prev_c = torch.zeros(1, self.lstm_size, device=self.device)
            prev_h = torch.zeros(1, self.lstm_size, device=self.device)

        inputs = self.encoder(torch.zeros(1).long().to(self.device))

        anchors, anchors_w_1 = [], []
        for unit_id in range(2):
            embed = inputs
            next_h, next_c = self.lstm(embed, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            anchors.append(torch.zeros_like(next_h, device=self.device))
            anchors_w_1.append(self.w_attn_1(next_h))
        unit_id += 1

        entropy, log_prob = [], []
        arc_seq = []
        while unit_id < self.unit_num + 2:
            prev_layers = []
            for i in range(2): # index_1, index_2
                embed = inputs
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))
                prev_c, prev_h = next_c, next_h
                query = torch.stack(anchors_w_1[:unit_id], dim=1)
                query = query.view(unit_id, self.lstm_size)
                query = torch.tanh(query + self.w_attn_2(next_h))
                query = self.v_attn(query)  # attention for skip connection
                logits = query.view(1, unit_id)  
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    logits = self.tanh_constant * torch.tanh(logits)
                prob = F.softmax(logits, dim=-1)
                index = torch.multinomial(prob, 1).long().view(1)
                arc_seq.append(index)
                arc_seq.append(0)
                curr_log_prob = F.cross_entropy(logits, index)
                log_prob.append(curr_log_prob)
                curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()

                entropy.append(curr_ent)
                prev_layers.append(anchors[index])
                inputs = prev_layers[-1].view(1, -1).requires_grad_()

            for i in range(2): # op_1, op_2
                embed = inputs
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))
                prev_c, prev_h = next_c, next_h
                logits = self.w_soft(next_h) + self.b_soft.requires_grad_()  # b_soft increases the prob of certain operations
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    op_tanh = self.tanh_constant
                    logits = op_tanh * torch.tanh(logits)
                if use_bias:
                    logits += self.b_soft_no_learn
                prob = F.softmax(logits, dim=-1)
                op_id = torch.multinomial(prob, 1).long().view(1)
                arc_seq[2*i-3] = op_id
                curr_log_prob = F.cross_entropy(logits, op_id)
                log_prob.append(curr_log_prob)
                curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()
                entropy.append(curr_ent)
                inputs = self.encoder(op_id+1)

            next_h, next_c = self.lstm(inputs, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            anchors.append(next_h)
            anchors_w_1.append(self.w_attn_1(next_h))
            inputs = self.encoder(torch.zeros(1).long().to(self.device))
            unit_id += 1

        arc_seq = torch.tensor(arc_seq)
        entropy = sum(entropy)
        log_prob = sum(log_prob)
        last_c = next_c
        last_h = next_h

        return arc_seq, entropy, log_prob, last_c, last_h