import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from module.transformer_encoder import TransformerEncoder, TransformerInterEncoder
from module.transformer_decoder import TransformerDecoder


def get_generator(dec_hidden_size, vocab_size, device):
    generate_function = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        generate_function
    )
    generator.to(device)

    return generator


class Model(nn.Module):
    def __init__(self, args, word_padding_idx, vocab_size, device, checkpoint=None):
        super(Model, self).__init__()

        self.args = args
        self.vocab_size = vocab_size
        self.device = device

        src_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        tgt_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)

        if self.args.share_embeddings:
            tgt_embeddings.weight = src_embeddings.weight

        if self.args.hier:
            self.encoder = TransformerInterEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                   self.args.ff_size, self.args.enc_dropout, src_embeddings,
                                                   inter_layers=self.args.inter_layers,
                                                   inter_heads=self.args.inter_heads, device=device)
        else:
            self.encoder = TransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                              self.args.ff_size,
                                              self.args.enc_dropout, src_embeddings)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.heads,
            d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.args.dec_hidden_size, self.vocab_size, device)
        if self.args.share_decoder_embeddings:
            self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            keys = list(checkpoint['model'].keys())

            for k in keys:
                if 'a_2' in k:
                    checkpoint['model'][k.replace('a_2', 'weight')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
                if 'b_2' in k:
                    checkpoint['model'][k.replace('b_2', 'bias')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])

            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)

    def forward(self, src, tgt):
        tgt = tgt[:-1]

        src_features, mask_hier = self.encoder(src)
        dec_state = self.decoder.init_decoder_state(src, src_features)

        if self.args.hier:
            decoder_outputs = self.decoder(tgt, src_features, dec_state, memory_masks=mask_hier)
        else:
            decoder_outputs = self.decoder(tgt, src_features, dec_state)

        return decoder_outputs
