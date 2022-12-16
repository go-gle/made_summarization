import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hid_dim,
                ):
        super().__init__()
        self.hid_dim = hid_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=1, bidirectional=True, batch_first=True)
        # Projector
        self.Wh = nn.Linear(hid_dim * 2, hid_dim * 2, bias=False)
        
        self.reduce_h = nn.Linear(hid_dim * 2, hid_dim)
        self.reduce_c = nn.Linear(hid_dim * 2, hid_dim)

        self.hid_dim = hid_dim
        
    def forward(self, inp):
        # inp [batch, seq_len]
        
        # embedded [batch, seq_len, emb_dim]
        embedded = self.embed(inp)
        
        # enc_out [batch, seq_len, 2 * hid_dim], hid = (hidden, cell)
        enc_out, hid = self.rnn(embedded)
        enc_out = enc_out.contiguous()
        
        # enc_feats [batch * seq_len, 2 * hid_dim]
        # this is needed for attention
        enc_feats = self.Wh(enc_out.view(-1, 2 * self.hid_dim))
        

        h, c = hid # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hid_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, self.hid_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
        
        hid = (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))

        return enc_out, enc_feats, hid


class CovAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        # For coverage
        self.Wc = nn.Linear(1, 2 * hid_dim, bias=False)

        # Usual Bahdanau Attention
        self.Ws = nn.Linear(2 * hid_dim, 2 * hid_dim)
        self.v = nn.Linear(2 * hid_dim, 1, bias=False)

    def forward(self, s_t_hat, enc_out, enc_feats, enc_padding_mask, coverage):
        batch, seq_len, dims = enc_out.shape
        
        # dec_feats: [batch, 2 * hid_dim]
        dec_feats = self.Ws(s_t_hat)
        # dec_feats: [batch, seq_len, 2 * hid_dim]
        dec_feats = dec_feats.unsqueeze(1).expand(batch, seq_len, dims).contiguous()
        # dec_feats: [batch * seq_len, 2 * hid_dim]
        dec_feats = dec_feats.view(-1, dims)
        
        # Coverage
        # [batch * seq_len, 1]
        cov_input = coverage.view(-1, 1) 
        cov_feats = self.Wc(cov_input)
        
        # [batch * seq_len, 2 * hid_dim]
        attn_feats = enc_feats + dec_feats + cov_feats
        
        # [batch, seq_len]
        energy = self.v(torch.tanh(attn_feats)).view(batch, seq_len)
        
        # Apply mask and renorm attentions
        # [batch, seq_len]
        a = F.softmax(energy, dim=1) * enc_padding_mask 
        a = a / a.sum(1, keepdim=True)
        # [batch, 1, seq_len]
        a = a.unsqueeze(1)
        
        
        # [B, 1, 2 * hid_dim]
        c_t = torch.bmm(a, enc_out)
        # [B, 2 * hid_dim]
        c_t = c_t.view(-1, self.hid_dim * 2)
        
        # [batch, seq_len]
        a = a.view(-1, seq_len)

        # [batch, seq_len]
        coverage = coverage.view(-1, seq_len)
        coverage = coverage + a

        return c_t, a, coverage


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hid_dim,
                ):
        
        super().__init__()
        self.hid_dim = hid_dim
        
        self.attn = CovAttention(hid_dim)
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=1, batch_first=True)
        
        self.x_context = nn.Linear(2 * hid_dim + emb_dim, emb_dim)
        self.p_gen = nn.Linear(4 * hid_dim + emb_dim, 1)
        
        self.out_proj = nn.Linear(3 * hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim, vocab_size)
    
    def forward(self, y_t, s_t_1, enc_outs, enc_feats,
                enc_padding_mask, c_t_1, extra_zeros,
                enc_batch_extend_vocab, coverage):
        '''
        y_t_1 - token
        s_t_1 - rnn state
        c_t_1 - context vector from attention
        '''
            
        y_t_1_embd = self.embed(y_t)
        x_t_1 = self.x_context(torch.cat((c_t_1, y_t_1_embd), dim=1))
        lstm_out, s_t = self.rnn(x_t_1.unsqueeze(1), s_t_1)
        
        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.hid_dim), c_decoder.view(-1, self.hid_dim)), dim=1)
        c_t, attn_dist, coverage_next = self.attn(s_t_hat, enc_outs, enc_feats, enc_padding_mask, coverage)
        coverage = coverage_next
        
        p_gen_input = torch.cat((c_t, s_t_hat, x_t_1), 1)
        p_gen = self.p_gen(p_gen_input)
        p_gen = torch.sigmoid(p_gen)
        
        output = torch.cat((lstm_out.view(-1, self.hid_dim), c_t), dim=1)
        output = self.out(self.out_proj(output))
        vocab_dist = p_gen * torch.softmax(output, dim=1)
        
        attn_dist_ = (1 - p_gen) * attn_dist
        
        # if we have oovs
        if extra_zeros is not None:
            vocab_dist = torch.cat([vocab_dist, extra_zeros], 1)

            final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class PGen(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hid_dim,
                ):
        super().__init__()
        encoder = Encoder(vocab_size, emb_dim, hid_dim)
        decoder = Decoder(vocab_size, emb_dim, hid_dim)

        # shared the embedding between encoder and decoder
        decoder.embed.weight = encoder.embed.weight

        self.encoder = encoder
        self.decoder = decoder

