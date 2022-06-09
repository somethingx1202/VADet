# -*- coding: utf8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional

# from .BasicModule import BasicModule


class VaeEncoder(nn.Module):
    '''
    compute the \mu and the vector \sigma
    where the \sigma^2 should be the variance for
    circled Gaussian
    '''

    def __init__(self, param_dim_encoder, param_dim_vocab,
                 param_dim_hidden):
        '''
        compute the encoder
        =============================
        params:
        ----------
        param_dim_encoder: encoder hiddenlayer size (\pi)
        param_dim_vocab: vocabulary size
                         has switched to albert_hiddenlayer_size +
                         albert_hiddenlayer_size // 512
        param_dim_hidden: hidden semantic size

        return:
        ----------
        None
        '''
        super(VaeEncoder, self).__init__()

        self.dim_encoder = param_dim_encoder
        self.dim_vocab = param_dim_vocab
        self.dim_hidden = param_dim_hidden

        # MLP for \pi
        # # ---------- changed in the albert front version
        self.W_encoder_pi = nn.Parameter(torch.Tensor(
            param_dim_encoder, param_dim_vocab))
        # param_dim_vocab has been switched to albert_hiddenlayer_size
        # self.W_encoder_pi = nn.Parameter(torch.Tensor(
        #     param_dim_encoder, param_dim_vocab))
        # # ---------- changed in the albert front version
        self.b_encoder_pi = nn.Parameter(torch.Tensor(
            param_dim_encoder))

        # MLP for \mu
        self.W_encoder_mu = nn.Parameter(torch.Tensor(
            param_dim_hidden, param_dim_encoder))
        self.b_encoder_mu = nn.Parameter(torch.Tensor(
            param_dim_hidden))
        # MLP for \sigma, here, sigma is a vector log_sqrt
        # because we only want to produce circle Gaussian
        self.W_encoder_sigma = nn.Parameter(torch.Tensor(
            param_dim_hidden, param_dim_encoder))
        self.b_encoder_sigma = nn.Parameter(torch.Tensor(
            param_dim_hidden))

        self.W_encoder_pi.data.normal_(mean=0.0, std=0.1)
        self.b_encoder_pi.data.fill_(0)
        self.W_encoder_mu.data.normal_(std=0.1)
        self.b_encoder_mu.data.normal_(std=0.1)

        self.W_encoder_sigma.data.normal_(std=0.1)
        self.b_encoder_sigma.data.normal_(std=0.1)

    def forward(self, input_xnwc):
        '''
        compute the mu and sigma_log_pow (log(sigma^2)
        =============================
        params:
        ----------
        input_xnwc: B, (V_SIZE, V_SIZE)
                    has switched to B, ALBERT_hiddenlayer_size

        return:
        ----------
        output: \mu, \sigma_log_pow
        '''
        B, ALBERT_HIDDEN_SIZE = input_xnwc.size()
        # + <=> add, if scalar, then +all, if vec, then + vec
        # * <=> mul, while mul is [a1b1,a2b2]
        # mv & mm is different, and torch.dot is vv
        input_xnwc = input_xnwc.permute(1, 0)
        pi_before_relu = torch.mm(self.W_encoder_pi, input_xnwc)\
            + self.b_encoder_pi.expand(B, self.dim_encoder).permute(1, 0)
        pi = torch.nn.functional.relu(pi_before_relu)

        # \mu, allow neg, so no relu was put
        mu = torch.mm(self.W_encoder_mu, pi)\
            + self.b_encoder_mu.expand(B, self.dim_hidden).permute(1, 0)

        # \sigma, also allow neg, and it's the log \sigma^2 in the first place
        sigma_log_pow = torch.mm(self.W_encoder_sigma, pi)\
            + self.b_encoder_sigma.expand(B, self.dim_hidden).permute(1, 0)

        mu_permuted = mu.permute(1, 0)
        sigma_log_pow_permuted = sigma_log_pow.permute(1, 0)
        return mu_permuted, sigma_log_pow_permuted


if __name__ == '__main__':

    # seq len = 40
    # if 2 * 100, then triggering error due to memory limitation
    # under the circumstance of anaconda35
    test = Variable(torch.Tensor(2048, 2 * 1000))
    # attention size = 30, hidden size =50
    att = VaeEncoder(200, 1000, 4000)
    (res_mu, res_sigma) = att(test)
    print(res_mu.size())
    print(res_sigma.size())
