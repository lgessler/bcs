
import os
import pickle
import re

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


PREPS = ["about", "above", "across", "after", "afterward", "against", "ago", "among", "apart", "around", "as", "at",
         "atop", "away", "back", "before", "behind", "below", "besides", "between", "beyond", "but", "by", "circa",
         "despite", "down", "downstairs", "during", "except", "for", "from", "home", "in", "indoors", "inside", "into",
         "like", "minus", "near", "nearby", "of", "off", "on", "onto", "out", "outside", "over", "past", "per", "plus",
         "round", "since", "than", "through", "throughout", "till", "to", "together", "toward", "under", "unlike",
         "until", "up", "upon", "via", "with", "within", "without"]


TARGET_PREPOSITION_PATTERN = re.compile('>>(' + '|'.join(PREPS) + ')<<', re.IGNORECASE)


def remove_and_index_target_prep(tokenized_text):
    for i in range(2, len(tokenized_text) - 2):
        if tokenized_text[i-2:i] == ['>', '>'] and tokenized_text[i+1:i+3] == ['<', '<']:
            tokenized_text = tokenized_text[:i-2] + tokenized_text[i:i+1] + tokenized_text[i+3:]
            return tokenized_text, i - 2

# cell 2
def sent2vec(model, tokenizer, sentence, CUDA, ignore_case=True, collapse_bert_tokens=True):
    """"""
    # extract target prep that has been indicated with >>arrows<< and remove arrows from sent
    target_preps = re.findall(TARGET_PREPOSITION_PATTERN, sentence)
    if len(target_preps) != 1:
        raise Exception("sentence did not contain exactly one arrowed target prep: " + sentence)

    marked_text = '[CLS] ' + sentence + ' [SEP]'
    tokenized_text = tokenizer.tokenize(marked_text)
    # remove arrows around >>prep<< and note where it appeared
    tokenized_text, target_index = remove_and_index_target_prep(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    if CUDA:
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensor = segments_tensor.to('cuda')

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensor)

    token_embeddings = torch.stack(encoded_layers, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1 (we want to get rid of "batch" dim and change order in the tensor).
    token_embeddings = token_embeddings.permute(1, 0, 2)

    if CUDA:
        token_embeddings = token_embeddings.to('cpu')

    # pooling strategy here: sum last 4 layers
    # Stores the token vectors, with shape [#tokens x 768]
    token_vecs_sum = []

    for token in token_embeddings:
        # `token` is a [12 x 768] tensor
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    return np.array([np.array(token_vecs_sum[target_index])])


# cell 3
def read_sentences(filepath):
    with open(filepath, 'r') as f:
        for sentence in f:
            yield sentence.strip()


def read_pickle(filepath):
    """Generate embeddings for target tokens for a given file where every line is a plaintext sentence"""
    cachepath = filepath + '.cache'  # ("_".join(target_tokens) if len(target_tokens) < 5 else str(hash(''.join(target_tokens)))) + '.cache'
    if os.path.isfile(cachepath):
        with open(cachepath, 'rb') as f:
            return pickle.load(f)


def query(model, tokenizer, vecs, sentences, query_sentences, CUDA):
    query_vecs = []
    for query_sentence in query_sentences:
        vec = sent2vec(model, tokenizer, query_sentence, CUDA)
        query_vecs.append(torch.tensor(vec))
    query_vec = torch.mean(torch.stack(query_vecs), dim=0)
    sims = cosine_similarity(vecs, query_vec)
    pairs = [(sim[0], sentences[i]) for i, sim in enumerate(sims)]
    return sorted(pairs, reverse=True, key=lambda x: x[0])[:500]


def query_encow(query_id, sentences, N=5000):
    # GPU available?
    CUDA = torch.cuda.is_available()

    # initialize the bert model
    print(f"Initializing BERT model {'with' if CUDA else 'without'} CUDA...", end='')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    if CUDA:
        model = model.to('cuda')
    model.eval()
    print(" OK.")

    aggregated_pairs = []
    for i in range(80):
        vecs, sents = read_pickle(f'encow/encow_sent.txt.{str(i).zfill(3)}')
        aggregated_pairs += query(model, tokenizer, vecs, sents, sentences, CUDA)
        print("Querying " + f'({i + 1}/80)' + ('.' * ((i % 3) + 1)) + '      ', end='\r')

    return query_id, sorted([(sim, sent[6:-6]) for sim, sent in aggregated_pairs], reverse=True, key=lambda x: x[0])[:N]
