import torch
from abc import abstractmethod

import numpy as np

from transformers import MBartTokenizer, MBartForConditionalGeneration

from pointer_gen.utils import preporocess_text, PointerDataPoint, decode
from pointer_gen.beam_decode import beam_decode

class Predictor:
    def __init__(self,
                 model_path: str,
                 vocab_path: str,
                 device:str = 'cpu',
                ):
        self.device = device
        self.vocab = torch.load(vocab_path)
        self.model = torch.load(model_path,
                           map_location=torch.device('cpu')).to(self.device)
        
    @abstractmethod
    def predict_one_sample(self, text):
        pass


class PGenPredictor(Predictor):
    
    def __init__(self,
                 model_path: str,
                 vocab_path: str,
                 device: str = 'cpu',
                ):
        super().__init__(model_path=model_path,
                         vocab_path=vocab_path,
                         device=device)
    
    def predict_one_sample(self, text):
        art_toks = preporocess_text(text, 400)
        point = PointerDataPoint(art_toks, [''], self.vocab)
        with torch.no_grad():
            src = torch.tensor(np.array(point.art_idxs)).unsqueeze(0).to(self.device)
            src_mask = torch.ones(len(point.art_idxs)).unsqueeze(0).to(self.device)
            src_ext = torch.tensor(np.array(point.art_ext_idxs)).unsqueeze(0).to(self.device)
            extra_zeros = torch.zeros(len(point.art_oovs)).unsqueeze(0).to(self.device)

            decoded = beam_decode(model=self.model,
                                  max_len=100,
                                  beam_width=5,
                                  src=src,
                                  src_mask=src_mask,
                                  src_ext=src_ext,
                                  extra_zeros=extra_zeros,
                                  device=self.device,
                                  vocab=self.vocab,
                                 )
        return ' ' .join(decode([i.item() for i in decoded[1:-1]], self.vocab, point.art_oovs))


class MBartPredictor(Predictor):
    def __init__(self, device='cpu'):
        self.device = device
        model_name = "IlyaGusev/mbart_ru_sum_gazeta"
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def predict_one_sample(self, text):
        input_ids = self.tokenizer(
            [text],
            max_length=600,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        
        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4,
        )[0]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
