import torch
import torch.nn as nn
from torch.autograd import Variable
# from .BasicModule import BasicModule


class Attention(nn.Module):
    def __init__(self, attention_size, hidden_size):
        '''
        compute the attentioned representation
        =============================
        params:
        ----------
        attention_size: attention size, correspond to doc_seq_len
        usually the attention_size should be the max_doc size,
        if exceed, then padding to zero
        hidden_size: hidden size, arbitatrlly, I
            personally set as biLSTM hidden_size concatenated

        return:
        ----------
        None
        '''

        super(Attention, self).__init__()

        self.W_omega = nn.Parameter(torch.Tensor(attention_size, hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size))
        self.W_omega.data.normal_(std=0.1)
        self.b_omega.data.normal_(std=0.1)
        self.u_omega.data.normal_(std=0.1)

    def forward(self, input_):
        '''
        compute the attentioned representation
        =============================
        params:
        ----------
        input: B, Seq, HiddenRepSize

        return:
        ----------
        output: B, HiddenRepSize
        '''
        # input (B,T,D)
        B, Seq, CharEmb = input_.size()
        # Seq correspond
        A = self.W_omega.size(0)
        # AttentionSize

        # B, Seq, CharEmb -> CharEmb, B, Seq
        input_permuted_contiguous_viewed = \
            input_.permute(2, 0, 1).contiguous().view(CharEmb, -1)
        w_mul_input = torch.mm(
            self.W_omega, input_permuted_contiguous_viewed)

        # print(w_mul_input.size())
        # attention_size, batch_size, doc_seq for v
        w_mul_input = w_mul_input.view(A, B, Seq)

        v = torch.tanh(
            w_mul_input + self.b_omega.expand(B, Seq, A).permute(2, 0, 1))
        # without allocating the new memory

        v = v.view(A, B, Seq)
        # v is attention_size, batch_size, doc_seq
        v = v.permute(1, 2, 0)
        # v is batch_size, doc_seq, attention_size
        vu = torch.mv(v.view(-1, A), self.u_omega)
        # u is the weight in the softmax

        vu = vu.view(B, Seq)
        # vu ( batch_size, doc_seq)
        alphas = torch.nn.functional.softmax(vu, dim=1)
        # # dim= 0 means sum( a[i][1][3]) = 1
        # alphas (batch_size, doc_seq) # alpha is batch-related

        # here input is (batch_size, doc_seq, CharEmb)
        # print( alphas.size())

        # input_permuted = input_.permute( 1, 0, 2 ).
        # contiguous().view( Seq, -1 )
        # print(input_permuted.size())
        # weighed = torch.mm( alphas,  input_permuted) # full multiply

        # B, S, D
        input_permuted = input_.permute(0, 1, 2)
        # print(input_permuted.size())
        # B, S => B, 1, S
        alphas = alphas.unsqueeze(dim=1)
        weighed = torch.bmm(alphas, input_permuted)
        # bmm means batch_related mm, since mm only accept matrix and matrix
        weighed = weighed.squeeze()
        # print( weighed.size())

        # imagine weighted is 1-dim, then weighted should be permuted.

        # output = torch.sum( weighed, dim=1 )
        return weighed


if __name__ == '__main__':

    test = Variable(torch.Tensor(10, 40, 50))
    # seq len = 40, batch size = 10, word vec size = 50
    att = Attention(30, 50)
    # attention size = 30, hidden size =50
    res = att(test)
    print(res.size())
