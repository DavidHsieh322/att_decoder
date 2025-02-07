from .ffn import FFN
from torch import nn

class TFEncVecDecoder(nn.Module):
    def __init__(self, args, value_dim):
        super(TFEncVecDecoder, self).__init__()
        self.embed = nn.Linear(value_dim, args.embed_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=args.embed_dim, 
            nhead=args.codebooks, 
            dim_feedforward=args.ffn_dim, 
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=args.decoder_layers)
        self.ffn = FFN(args.embed_dim, args.source_dim, args.codebooks)

    def forward(self, x):
        x = self.embed(x)
        x = self.decoder(x)
        out = self.ffn(x)

        return out

class VecAttnDecoder(nn.Module):
    def __init__(self, args, value_dim) -> None:
        super(VecAttnDecoder, self).__init__()
        self.m = args.codebooks

        if args.model == 'TFDec-vec-top1q' and not args.sum_topk:
            self.linear_q = nn.Linear(value_dim // args.topk, args.embed_dim)
        else:
            self.linear_q = nn.Linear(value_dim, args.embed_dim)
        if args.sum_topk:
            self.linear_v = nn.Linear(value_dim, args.embed_dim)
        else:
            self.linear_v = nn.Linear(value_dim // args.topk, args.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.embed_dim, 
            nhead=self.m, 
            dim_feedforward=args.ffn_dim, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.decoder_layers)
        self.ffn = FFN(args.embed_dim, args.source_dim, self.m)

    def forward(self, xq, xkv):
        xq = self.linear_q(xq)
        xkv = self.linear_v(xkv)
        x = self.decoder(xq, xkv)
        out = self.ffn(x)

        return out