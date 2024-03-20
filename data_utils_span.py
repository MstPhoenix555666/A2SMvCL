'''
Description: data process for scope-labeled datasets
version: 
'''

import json
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
from tree import head_to_tree, tree_to_adj

def ParseData(data_path):
    with open(data_path) as infile:
        polar_dict = {'1':'positive', '-1':'negative', '0':'neutral'}
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = polar_dict[str(aspect['polarity'])]   # label
                pos = list(d['postag'])         # pos_tag 
                head = list(d['edges'])       # head
                head = [int(x) for x in head]
                deprel = list(d['deprels'])   # deprel
                # position
                aspect_post = [int(aspect['from']), int(aspect['to'])+1] 
                post = [i-int(aspect['from']) for i in range(int(aspect['from']))] \
                       +[0 for _ in range(int(aspect['from']), int(aspect['to'])+1)] \
                       +[i-int(aspect['to'])+1 for i in range(int(aspect['to'])+1, length)]                     
                # span
                s_b,s_e = aspect['scope'] 
                
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(int(aspect['from']))] \
                       +[1 for _ in range(int(aspect['from']), int(aspect['to'])+1)] \
                       +[0 for _ in range(int(aspect['to'])+1, length)]            
                # span mask
                span = [0 for _ in range(s_b)] + [1 for _ in range(s_b,s_e+1)] \
                    +[0 for _ in range(s_e+1,length)]
                
                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list, 'span_mask': span}
                all_data.append(sample)

    return all_data

def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSASpanData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive':2, 'negative':0, 'neutral':1}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            head = obj['head']
            
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]
                        
            # syntax-tree
            length = len(text_list)
            def inputs_to_tree_reps(maxlen, head, words, l):
                tree = head_to_tree(head, words, l)
                adj = tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(maxlen, maxlen)      
                return adj
            adj = inputs_to_tree_reps(length, head, text, length)
            assert len(text_list) == adj.shape[0] == adj.shape[1], '{}-{}-{}'.format(len(text_list), text_list, adj.shape)          
            
            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)                  
                    left_tok2ori_map.append(ori_i)          
            asp_start = len(left_tokens)  
            offset = len(left) 
            for ori_i, w in enumerate(term):        
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term) 
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i+offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()
                    
            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
                    
            adj_bert = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')
            for i in range(truncate_tok_len):
                for j in range(truncate_tok_len):
                    adj_bert[i][j] = adj[tok2ori_map[i]][tok2ori_map[j]]

            context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            #aspect token
            asp_tokens = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + [tokenizer.sep_token_id]
            asp_tokens = asp_tokens + [0] * (opt.max_length-len(asp_tokens))
            asp_tokens = np.asarray(asp_tokens, dtype='int64')
            
            #pad adj
            ori_adj_matrix = np.zeros(
                (tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
            ori_adj_matrix[1:context_len + 1, 1:context_len + 1] = adj_bert
            
            #span-mask pad
            span = obj['span_mask']
            tok_adj = np.zeros(context_len, dtype='int64') #bert后token的长度
            for i in range(context_len):     
                tok_adj[i] = span[tok2ori_map[i]]
                
            span_mask = np.zeros(tokenizer.max_seq_len).astype('int64')
            pad_adj = np.zeros(context_asp_len).astype('int64') #[cls]text[sep]aspect[sep]
            pad_adj[1:context_len + 1] = tok_adj
            span_mask[:context_asp_len] = pad_adj
            
            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'asp': asp_tokens,
                'adj_matrix': ori_adj_matrix, 
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
                'len':context_len,
                'asp_len':context_asp_len,
                'span': span_mask,
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
