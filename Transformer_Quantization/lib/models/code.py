from .ffn import FFN
from torch import nn
import torch

class TFEncCodeDecoder(nn.Module):
    def __init__(self, args, value_dim):
        super(TFEncCodeDecoder, self).__init__()
        self.embed = nn.Embedding(2**args.cb_bits, args.embed_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=args.embed_dim, 
            nhead=args.codebooks, 
            dim_feedforward=args.ffn_dim, 
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=args.decoder_layers)
        self.ffn = FFN(args.embed_dim, args.source_dim, args.codebooks * args.topk)

    def forward(self, x):
        x = self.embed(x)
        x = self.decoder(x)
        out = self.ffn(x)

        return out

class CodeConcatBlock(nn.Module):
    def __init__(self, topk, embed_dim):
        super(CodeConcatBlock, self).__init__()
        self.fc = nn.Linear(topk * embed_dim, embed_dim)

    def forward(self, x):
        bs, m, topk, embed_dim = x.shape
        x = x.view(bs, m, topk * embed_dim)
        x = self.fc(x)

        return x

class CodeSumBlock(nn.Module):
    def __init__(self):
        super(CodeSumBlock, self).__init__()

    def forward(self, x):
        x = x.sum(dim=2)

        return x

class CodeAttnDecoder(nn.Module):
    def __init__(self, args, value_dim):
        super(CodeAttnDecoder, self).__init__()
        self.m = args.codebooks

        self.tablePE = torch.arange(self.m, device=args.device) * (2**args.cb_bits)
        self.tablePE = self.tablePE.view(1, self.m, 1)
        self.table = nn.Embedding((2**args.cb_bits) * self.m, args.embed_dim)

        if args.model == 'TFDec-code-top1q':
            topk_query = 1
        else:
            topk_query = args.topk

        if args.sum_topk:
            self.merge = CodeSumBlock()
        else:
            self.merge = CodeConcatBlock(topk_query, args.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.embed_dim, 
            nhead=self.m, 
            dim_feedforward=args.ffn_dim, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.decoder_layers)
        self.ffn = FFN(args.embed_dim, args.source_dim, self.m)

    def forward(self, xq, xkv):
        bs, qN = xq.shape
        xq = xq.view(bs, self.m, qN // self.m)
        xq = xq + self.tablePE.repeat(bs, 1, qN // self.m)
        xq = self.table(xq)
        xq = self.merge(xq)

        kvN = xkv.shape[1]
        xkv = xkv + self.tablePE.repeat(bs, 1, kvN // self.m).view(bs, -1)
        xkv = self.table(xkv)

        x = self.decoder(xq, xkv)
        out = self.ffn(x)

        return out