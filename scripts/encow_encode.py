import math
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle
from itertools import islice
import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel

PREPS = ["about", "above", "across", "after", "afterward", "against", "ago", "among", "apart", "around", "as", "at",
         "atop", "away", "back", "before", "behind", "below", "besides", "between", "beyond", "but", "by", "circa",
         "despite", "down", "downstairs", "during", "except", "for", "from", "home", "in", "indoors", "inside", "into",
         "like", "minus", "near", "nearby", "of", "off", "on", "onto", "out", "outside", "over", "past", "per", "plus",
         "round", "since", "than", "through", "throughout", "till", "to", "together", "toward", "under", "unlike",
         "until", "up", "upon", "via", "with", "within", "without"]

# GPU available?
CUDA = torch.cuda.is_available()
print("CUDA: ", CUDA)

#initialize the bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
if CUDA:
    model = model.to('cuda')
model.eval()


# Creating the list of vectors, corresponding to the list of sentences
def batched_seq(iterable, size):
    iterator = iter(iterable)
    for first in iterator:  # stops when iterator is depleted
        def chunk():  # construct generator for next chunk
            yield first  # yield element from for loop
            for more in islice(iterator, size - 1):
                yield more  # yield more elements from the iterator

        yield chunk()


def batched_sent2vec(sentences, target_tokens, ignore_case=True, collapse_bert_tokens=True, batch_size=1024,
                     total=None):
    vectors = []
    output_sentences = []

    for batch in tqdm.tqdm(batched_seq(sentences, batch_size), total=total):
        tokenized_text_list = []
        indexed_tokens_list = []
        segments_ids_list = []
        for sentence in batch:
            marked_text = '[CLS] ' + sentence + ' [SEP]'
            tokenized_text = tokenizer.tokenize(marked_text)
            if len(tokenized_text) > 500:
                print(f"Skipping long sentence: \"{sentence}\"")
                continue
            tokenized_text_list.append(tokenized_text)
            indexed_tokens_list.append(tokenizer.convert_tokens_to_ids(tokenized_text))
            segments_ids_list.append([1] * len(tokenized_text))

        # Convert inputs to PyTorch tensors
        tokens_tensor = pad_sequence([torch.tensor(seq) for seq in indexed_tokens_list], batch_first=True)
        segments_tensor = pad_sequence([torch.tensor(seq) for seq in segments_ids_list], batch_first=True)
        if CUDA:
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensor = segments_tensor.to('cuda')

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensor)

        # Shape is (12 x batch_size x max_seq_length x 768)
        token_embeddings_matrix = torch.stack(encoded_layers, dim=0)
        # Rearrange dims, yielding (batch_size x max_seq_length x 12 x 768)
        token_embeddings_matrix = token_embeddings_matrix.permute(1, 2, 0, 3)

        if CUDA:
            token_embeddings_matrix = token_embeddings_matrix.to('cpu')

        for tokenized_text, token_embeddings in zip(tokenized_text_list, token_embeddings_matrix):
            # pooling strategy here: sum last 4 layers
            # Stores the token vectors, with shape [#tokens x 768]
            token_vecs_sum = []

            # now find the indexes the token appears in, pool the embedding for these indexes (i.e the token emb.)
            token_tuples = []
            for token_str in enumerate(tokenized_text):
                token_tuples.append(token_str)
            token_list = [list(elem) for elem in token_tuples]

            # scan for target tokens and build output if we find it
            target_tokens = target_tokens if not ignore_case else [t.lower() for t in target_tokens]
            for i, token in token_list:
                if (token.lower() if ignore_case else token) in target_tokens:
                    # Append the sum of the last four layers
                    vectors.append(np.array(torch.sum(token_embeddings[i][-4:], dim=0)))
                    output_sentences.append(" ".join([t if j != i else '>>' + t + '<<' for j, t in token_list]))
                    if collapse_bert_tokens:
                        output_sentences[-1] = output_sentences[-1].replace(" ##", "")
    return np.array(vectors), output_sentences


def read_sentences(filepath):
    with open(filepath, 'r') as f:
        for sentence in f:
            yield sentence.strip()


def embed_chunk(filepath, target_tokens, batch_size=32, total=None):
    """Generate embeddings for target tokens for a given file where every line is a plaintext sentence"""
    cachepath = filepath + '.cache'
    if os.path.isfile(cachepath):
        with open(cachepath, 'rb') as f:
            return pickle.load(f)

    vectors, sentences = batched_sent2vec(read_sentences(filepath), target_tokens=target_tokens, batch_size=batch_size,
                                          total=total)

    print("Writing cached results to " + cachepath)
    with open(cachepath, 'wb') as f:
        pickle.dump((vectors, sentences), f)

    return np.array(vectors), sentences


def embed_encow():
    for i in range(80):
        embed_chunk(f'encow/encow_sent.txt.{str(i).zfill(3)}', target_tokens=PREPS, batch_size=16, total=math.ceil(100000/16))


embed_encow()
