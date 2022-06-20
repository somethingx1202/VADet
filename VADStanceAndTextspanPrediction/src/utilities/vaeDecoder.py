# -*- coding: utf8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional


class VaeDecoder(nn.Module):
    '''
    compute p_theta using z_i, before sampling
    '''

    def __init__(self, param_dim_topic, param_dim_vocab,
                 param_dim_wncs,
                 param_dim_hidden):
        '''
        compute the de
        =============================
        params:
        ----------
        param_dim_topic: topic count
        param_dim_vocab: vocabulary size
                         has switched to albert_hiddenlayer_size
        param_dim_wncs: vocabulary size
                        has switched to albert_hiddenlayer_size
        param_dim_hidden: hidden semantic size

        return:
        ----------
        a sum of log_softmax
        '''
        super(VaeDecoder, self).__init__()

        self.dim_topic = param_dim_topic
        self.dim_vocab = param_dim_vocab
        self.dim_wncs = param_dim_wncs
        self.dim_hidden = param_dim_hidden

        # MLP for p_xn_unsoftmaxed
        self.W_decoder_xn = nn.Parameter(torch.Tensor(
            param_dim_vocab, param_dim_hidden))
        self.b_decoder_xn = nn.Parameter(torch.Tensor(
            param_dim_vocab))

        # MLP for theta
        self.W_decoder_zeta = nn.Parameter(torch.Tensor(
            param_dim_topic, param_dim_hidden))
        self.b_decoder_zeta = nn.Parameter(torch.Tensor(
            param_dim_topic))

        # topic matrix, each line * \zeta
        # should be a log vocab distribution
        # you don't need to softmax each line when visualizing
        # please refer to the note book
        self.MATRIX_decoder_beta = nn.Parameter(torch.Tensor(
            param_dim_topic, param_dim_wncs))
        # log background voca distribution, for smoothing
        self.background_decoder_beta = nn.Parameter(torch.Tensor(
            param_dim_wncs))

        self.W_decoder_xn.data.normal_(mean=0.0, std=0.1)
        self.b_decoder_xn.data.fill_(0)

        self.W_decoder_zeta.data.normal_(mean=0.0, std=0.1)
        self.b_decoder_zeta.data.fill_(0)

        self.MATRIX_decoder_beta.data.normal_(std=0.1)
        self.background_decoder_beta.data.normal_(std=0.1)
        return None

    def forward(self, input_zn):
        '''
        compute the mu and sigma_log_pow (log(sigma^2)
        =============================
        params:
        ----------
        input_zn: hidden variable z, computed by encoder

        return:
        ----------
        output: p_xn, p_wc
        # output: p_xn_at_input_xn, IIp_wc_at_input_wc
        '''

        # B1, VOCABSIZE = input_xn.size()
        B, HIDDENSIZE = input_zn.size()
        # assert B1 == B2
        # B = B1

        # produce p_xn
        # zn permuted
        # VOCABSIZE, B1
        input_zn = input_zn.permute(1, 0)

        # # ---------- removed in the ALBERT front version
        p_xn_unsoftmaxed = torch.nn.functional.relu(
            (torch.mm(self.W_decoder_xn, input_zn) +
             self.b_decoder_xn.expand(B, self.dim_vocab).permute(1, 0)))
        # print(p_xn_unsoftmaxed.size())

        # VOCASIZE * BATCHSIZE
        # p_xn = torch.nn.functional.softmax(p_xn_unsoftmaxed, dim=0)
        # normalisation is performed after concatenation
        p_xn = p_xn_unsoftmaxed
        # BATCHSIZE * VOCASIZE
        p_xn = p_xn.permute(1, 0)
        # # ---------- removed in the ALBERT front version

        # produce p_wc
        zeta = torch.nn.functional.relu(
            (torch.mm(self.W_decoder_zeta, input_zn) +
             self.b_decoder_zeta.expand(B, self.dim_topic).permute(1, 0)))
        # zeta: TOPICSIZE * BATCHSIZE
        p_wc_untanhed = torch.mm(
            self.MATRIX_decoder_beta.permute(1, 0), zeta)\
            + self.background_decoder_beta.expand(
                B, self.dim_wncs).permute(1, 0)
        # dim_vocab -> dim_wncs
        # relu is point-wise
        p_wc_unsoftmaxed = torch.nn.functional.relu(p_wc_untanhed)

        # simply called for normalization, if this is 300-dim, then this
        # can be directly used for sampling
        # # ---------- changed in the ALBERT version
        # p_wc = torch.nn.functional.softmax(p_wc_unsoftmaxed, dim=0)
        p_wc = p_wc_unsoftmaxed
        # # ---------- changed in the ALBERT version
        # BATCHSIZE * VOCASIZE
        p_wc = p_wc.permute(1, 0)

        # # ---------- changed in the ALBERT version
        # return p_xn, p_wc
        # # ---------- changed in the ALBERT version
        # return p_wc
        return p_xn

    def forward_obtain_zeta(
            self,
            input_zn):
        '''
        compute the zeta from z, typically it's computed
        from \mu(log(sigma^2))
        =============================
        params:
        ----------
        input_zn: hidden variable z, computed by encoder

        return:
        ----------
        output: zeta
        '''
        B, HIDDENSIZE = input_zn.size()
        input_zn = input_zn.permute(1, 0)
        zeta_untanhed = torch.mm(
            self.W_decoder_zeta, input_zn)\
            + self.b_decoder_zeta.expand(B, self.dim_topic).permute(1, 0)
        zeta = torch.nn.functional.relu(zeta_untanhed)
        zeta = zeta.permute(1, 0)
        return zeta


if __name__ == '__main__':

    # seq len = 40
    # test_xn = Variable(torch.Tensor(10, 40))
    # test_wc = Variable(torch.Tensor(10, 40))
    test_zn = Variable(torch.Tensor(13, 40))
    test_zn.data.normal_(std=0.1)
    # attention size = 30, hidden size = 50
    att = VaeDecoder(param_dim_topic=11,
                     param_dim_vocab=3,
                     param_dim_hidden=40)
    # (res_p_xn, res_p_wc) = att(test_zn)
    res_p_wc = att(test_zn)

    # print(res_p_xn.size())
    print(res_p_wc.size())
    # print(res_p_xn[0])
    print('end')
