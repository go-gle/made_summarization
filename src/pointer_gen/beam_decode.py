import torch
from queue import PriorityQueue
from tqdm.notebook import tqdm

from pointer_gen.utils import SOS, EOS,OOV


class BeamSearchNode(object):
    def __init__(self, prev_node, s_t, c_t, y_t, cov_t, logprob, length):
        """
        prev_node - prev node ref
        s_t - hidden state
        c_t - context from attention
        y_t - token
        cov_t - coverage vector
        log_prob - log probability of y_t
        length - lenght os seq
        """
        self.s_t = s_t
        self.c_t = c_t
        self.cov_t = cov_t
        self.prev_node = prev_node
        self.y_t = y_t
        self.logprob = logprob
        self.length = length

    def get_logprob_score(self):
        return self.logprob / float(self.length - 1 + 1e-8)
    
    def __lt__(self, other):
        return self.y_t < other.y_t


def beam_decode(model, max_len, beam_width, src, src_mask, src_ext, extra_zeros, device, vocab):
    batch_size = src.shape[0]
    decoded_output = []
    
    encoder_outputs, encoder_feature, s_t = model.encoder(src)
    
    for batch_ind in tqdm(range(batch_size)):
        # Create input for 1st decoding step
        y_t = torch.LongTensor([vocab[SOS]]).to(device)
        term_nodes = []
        c_t = torch.zeros((1, 2 * s_t[0].shape[-1]), requires_grad=True).to(device)
        cov_t = torch.zeros((1, src.shape[-1]), requires_grad=True).to(device)
        
        cur_node = BeamSearchNode(prev_node=None,
                                  s_t=s_t,
                                  c_t=c_t,
                                  y_t=y_t,
                                  cov_t=cov_t,
                                  logprob=0,
                                  length=1
                                 )
        nodes = PriorityQueue()
        nodes.put((-cur_node.get_logprob_score(), cur_node))

        for i in tqdm(range(max_len)):
            next_nodes = []
            
            for j in range(beam_width):
                if nodes.qsize() == 0:
                    # First Iteration
                    continue
                _, cur_node = nodes.get(False)
                y_t = cur_node.y_t
                s_t = cur_node.s_t
                c_t = cur_node.c_t
                cov_t = cur_node.cov_t

                if y_t.item() == vocab[EOS]:
                    next_nodes.append(cur_node)
                    continue

                y_t = y_t if y_t < len(vocab) else torch.LongTensor([vocab[OOV]]).to(device)


                dist, s_t,  c_t, attn_dist, p_gen, cov_t = model.decoder(y_t, s_t,
                                                                         encoder_outputs,
                                                                         encoder_feature,
                                                                         src_mask,
                                                                         c_t,
                                                                         extra_zeros,
                                                                         src_ext,
                                                                         cov_t,
                                                                        )
                log_prob = torch.log_softmax(dist, dim=-1)
                log_prob, indexes = torch.topk(log_prob, beam_width)
                for k in range(beam_width):
                    next_nodes.append(BeamSearchNode(prev_node=cur_node,
                                                    s_t=s_t,
                                                    c_t=c_t,
                                                    y_t=indexes[:, k],
                                                    cov_t=cov_t,
                                                    logprob=cur_node.logprob + log_prob[0, k].item(),
                                                    length=cur_node.length + 1,
                                                   ))

            for n in next_nodes:
                nodes.put((-n.get_logprob_score(), n))
        best_decoding = []
        _, cur_node = nodes.get()
        while cur_node:
            best_decoding.append(cur_node.y_t)
            cur_node = cur_node.prev_node
        best_decoding = best_decoding[::-1]
        decoded_output.append(best_decoding)
    return best_decoding