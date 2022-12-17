from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from pointer_gen.utils import preporocess_text


def create_target_mask(tokens, abst):
    target_mask = []
    abst_set = set(abst)
    for t in tokens:
        if t in abst_set:
            target_mask.append(1)
        else:
            target_mask.append(0)
    return target_mask


class TaggingDataset(Dataset):
    """Saves tokenized and shortened articles"""
    def __init__(self,
                 articles,
                 abstracts,
                 max_art_len = 400,
                 max_abs_len = 100,
                ):
        assert len(articles) == len(abstracts)
        self.articles = []
        self.targets = []
        for i in tqdm(range(len(articles))):
            art_toks = preporocess_text(articles[i], max_art_len)
            abs_toks = preporocess_text(abstracts[i], max_abs_len)
            target_mask = create_target_mask(art_toks, abs_toks)
            
            self.articles.append(art_toks)
            self.targets.append(target_mask)

    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        return self.articles[idx], self.targets[idx]
