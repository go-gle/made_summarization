from collections import Counter
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize


SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
OOV = '<OOV>'


tokenize = lambda sent: word_tokenize(sent.lower())


def get_counters(src, summary):
    src_counter = Counter()
    #Creatae vocab from texts and summaries
    for i in tqdm(range(len(src))):
        src_counter.update(tokenize(src[i]))
        src_counter.update(tokenize(summary[i]))
    return src_counter


def encode(article_tokens, vocab):
    """
    Encodes article with vocab
    Return:
        idxs - encoded with vocab
    """
    article_tokens = [SOS] + article_tokens + [EOS]
    return [vocab[tok] for tok in article_tokens]


def encode_ext_article(article_tokens, vocab):
    """
    Encodes articles with extented vocab: oov token is replaces by token from ext vocab
    Return:
        idx - encoded with ext vocab
        oovs - list of ooc tokens
    """
    article_tokens = [SOS] + article_tokens + [EOS]
    oovs = []
    idxs = []
    oov_idx = vocab[OOV]
    for tok in article_tokens:
        tok_idx = vocab[tok]
        if tok_idx == oov_idx:
            if tok not in oovs:
                oovs.append(tok)
            oov_num = oovs.index(tok)
            idxs.append(len(vocab) + oov_num)
        else:
            idxs.append(tok_idx)
    return idxs, oovs


def encode_ext_abstract(abstract_tokens, vocab, article_oovs):
    """
    Encodes abstract with extented vocab: oov token is replaces by token from ext vocab
    If token is oov and in article_oovs it is replaced from article_oovs
    Return:
        idx - encoded with ext vocab
    """
    abstract_tokens = [SOS] + abstract_tokens + [EOS]
    idxs = []
    oov_idx = vocab[OOV]
    for tok in abstract_tokens:
        tok_idx = vocab[tok]
        if tok_idx == oov_idx:
            if tok in article_oovs:
                vocab_indx = len(vocab) + article_oovs.index(tok)
                idxs.append(vocab_indx)
            else:
                idxs.append(oov_idx)
        else:
            idxs.append(tok_idx)
    return idxs


def decode(idxs, vocab, oovs):
    itos = vocab.get_itos()
    words = []
    for idx in idxs:
        if idx >= len(itos):
            words.append(oovs[idx - len(itos)])
        else:
            words.append(itos[idx])
    return words


def preporocess_text(text, max_len):
    tokens = tokenize(text)
    return tokens[:max_len]


class SummDataset(Dataset):
    """Saves tokenized and shortened articles"""
    def __init__(self,
                 articles,
                 abstracts,
                 max_art_len = 400,
                 max_abs_len = 100,
                ):
        assert len(articles) == len(abstracts)
        self.articles = []
        self.abstracts = []
        for i in tqdm(range(len(articles))):
            art_toks = preporocess_text(articles[i], max_art_len)
            abs_toks = preporocess_text(abstracts[i], max_abs_len)
            
            self.articles.append(art_toks)
            self.abstracts.append(abs_toks)

    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        return self.articles[idx], self.abstracts[idx]


class PointerDataPoint:
    def __init__(self,
                 art_toks,
                 abs_toks,
                 vocab,
                ):
        self.art_idxs = encode(art_toks, vocab)
        self.abs_idxs = encode(abs_toks, vocab)
        self.art_ext_idxs, self.art_oovs = encode_ext_article(art_toks, vocab)
        self.abs_ext_idxs = encode_ext_abstract(abs_toks, vocab, self.art_oovs)
