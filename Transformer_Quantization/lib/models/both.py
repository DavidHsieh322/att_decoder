from .ffn import FFN
from torch import nn

class TFDecVecCodeDecoder(nn.Module):
    def __init__(self, args, value_dim):
        super(TFDecVecCodeDecoder, self).__init__()
        self.embed_c = nn.Embedding(2**args.cb_bits, args.embed_dim)
        self.embed_v = nn.Linear(value_dim, args.embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.embed_dim, 
            nhead=args.codebooks, 
            dim_feedforward=args.ffn_dim, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.decoder_layers)
        self.ffn = FFN(args.embed_dim, args.source_dim, args.codebooks)

    def forward(self, xc, xv):
        xc = self.embed_c(xc)
        xv = self.embed_v(xv)
        x = self.decoder(xv, xc)
        out = self.ffn(x)

        return out