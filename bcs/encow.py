
import os
import pickle
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

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


# cell 2
def sent2vec(sentence, target_tokens=["in", "of", "to"], ignore_case=True, collapse_bert_tokens=True):
    """Take a sentence and a target token and return a 2-tuple where the first element an np array whose rows are the
    averaged last 4 BERT layers of the target token for each occurrence of the target token in the sentence, and the
    second element is a string representation of the corresponding token in context. E.g.,

    >>> sent2vec("She finished it through and through", target_token="through")
    (array([[...]]), ["She finished it >>through<< and through", "She finished it through and >>through<<"])
    """
    marked_text = '[CLS] ' + sentence + ' [SEP]'
    tokenized_text = tokenizer.tokenize(marked_text)
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

    # now find the indexes the token appears in, pool the embedding for these indexes (i.e the token emb.)
    token_tuples = []
    for token_str in enumerate(tokenized_text):
        token_tuples.append(token_str)
    token_list = [list(elem) for elem in token_tuples]

    # this is an example for pooling token embeddings for the word "in"
    vectors = []
    sentences = []
    target_tokens = target_tokens if not ignore_case else [t.lower() for t in target_tokens]
    for i, token in token_list:
        if (token.lower() if ignore_case else token) in target_tokens:
            vectors.append(np.array(token_vecs_sum[i]))
            sentences.append(" ".join([t if j != i else '>>' + t + '<<' for j, t in token_list]))
            if collapse_bert_tokens:
                sentences[-1] = sentences[-1].replace(" ##", "")
    return np.array(vectors), sentences


# cell 3
def read_sentences(filepath):
    with open(filepath, 'r') as f:
        for sentence in f:
            yield sentence.strip()


def read_pickle(filepath):
    """Generate embeddings for target tokens for a given file where every line is a plaintext sentence"""
    cachepath = filepath + '.' + "-7913747506583000201.cache"  # ("_".join(target_tokens) if len(target_tokens) < 5 else str(hash(''.join(target_tokens)))) + '.cache'
    if os.path.isfile(cachepath):
        with open(cachepath, 'rb') as f:
            return pickle.load(f)


def query(vecs, sentences, query_sentences, target_tokens):
    query_vecs = []
    for query_sentence in query_sentences:
        vec, _ = sent2vec(query_sentence, target_tokens=target_tokens)
        query_vecs.append(torch.tensor(vec))
    query_vec = torch.mean(torch.stack(query_vecs), dim=0)
    sims = cosine_similarity(vecs, query_vec)
    pairs = [(sim[0], sentences[i]) for i, sim in enumerate(sims)]
    return sorted(pairs, reverse=True, key=lambda x: x[0])[:500]


def query_encow(query_id, target_prep, sentences, N=5000):
    aggregated_pairs = []
    for i in range(80):
        vecs, sents = read_pickle(f'encow/encow_sent.txt.{str(i).zfill(3)}')
        aggregated_pairs += query(vecs, sents, sentences, [target_prep])
        print("Querying " + f'({i + 1}/80)' + ('.' * ((i % 3) + 1)) + '      ', end='\r')

    return query_id, sorted([(sim, sent[6:-6]) for sim, sent in aggregated_pairs], reverse=True, key=lambda x: x[0])[:N]
