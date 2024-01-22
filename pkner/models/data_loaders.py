from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from tokenizers import Encoding
import torch
from pkner.models.utils import simplify_labels_and_tokens, get_entity_token_indices
from pkner.utils import read_jsonl, bio_to_entity_tokens
import matplotlib.pyplot as plt
import random
from collections import Counter


def get_val_dataloader(val_data_file: Path, tokenizer: BertTokenizerFast, max_len: int, batch_size: int,
                       n_workers: int, tag_type: str, print_tokens: bool, tag2id: Dict[str, int],
                       dataset_name: str) -> DataLoader:
    labels_key = tag_type + "_tags"
    raw_val_samples = list(read_jsonl(file_path=val_data_file))
    val_samples = [{"tokens": [t["text"] for t in sentence["tokens"]], "text": sentence["text"],
                    "labels": sentence[labels_key]} for sentence in raw_val_samples]

    val_dataloader = make_dataloader(inp_samples=val_samples, batch_size=batch_size, inp_tokenizer=tokenizer,
                                     max_len=max_len, shuffle=False, n_workers=n_workers,
                                     tag2id=tag2id, dataset_name=dataset_name, print_tokens=print_tokens)

    return val_dataloader


def get_training_dataloader(training_data_file: Path, tokenizer: BertTokenizerFast, max_len: int, batch_size: int,
                            n_workers: int, tag_type: str, print_tokens: bool, dataset_name: str = "training",
                            tag2id: Dict[str, int] = None, al_exp_train: bool = False
                            ) -> Tuple[DataLoader, Dict[str, int], Dict[int, str], Dict[str, float]]:
    """
    @param training_data_file: jsonl file with NER annotations in IOB format
    @param tokenizer: Pre-loaded BERT tokenizer in which all the special tokens (e.g.[unused0]) have been registered
    @param max_len: maximum length for the list of context/mention tokens
    @param batch_size: batch size
    @param n_workers: number of workers for the dataloader
    @param tag_type: either bio or biluo
    @param print_tokens: whether to print tokens
    @param dataset_name: name of the dataset (just for logging purposes)
    @param tag2id: optional to include if loading a pre-trained model that already had some tag2id
    @return: (1) dataloader for the training data, (2) tag2id converter, (3) id2tag converter, (4) scaling dictionary
    """
    # 1. Read data
    labels_key = tag_type + "_tags"
    raw_train_samples = list(read_jsonl(file_path=training_data_file))
    if al_exp_train:
        raw_train_samples = random.sample(raw_train_samples, 500)

    train_dataloader, tag2id, id2tag, scaling_dict = construct_dataloader_and_mappers(raw_samples=raw_train_samples,
                                                                                      tokenizer=tokenizer,
                                                                                      max_len=max_len,
                                                                                      batch_size=batch_size,
                                                                                      n_workers=n_workers,
                                                                                      labels_key=labels_key,
                                                                                      dataset_name=dataset_name,
                                                                                      print_tokens=print_tokens,
                                                                                      tag2id=tag2id)

    return train_dataloader, tag2id, id2tag, scaling_dict


def get_merged_dataloader(files_to_merge: List[str], tokenizer: BertTokenizerFast,
                          max_len: int, batch_size: int, n_workers: int, tag_type: str, print_tokens: bool,
                          dataset_name: str, tag2id: Dict[str, int] = None
                          ) -> Tuple[DataLoader, Dict[str, int], Dict[int, str], Dict[str, float]]:
    """
    Similar to get_training_dataloader but including all the instances from all_data_files list, which is a list of
    paths to jsonl files that will be merged
    """
    labels_key = tag_type + "_tags"
    all_raw_samples = [x for tmp_file in files_to_merge for x in read_jsonl(Path(tmp_file))]
    merged_dataloader, tag2id, id2tag, scaling_dict = construct_dataloader_and_mappers(raw_samples=all_raw_samples,
                                                                                       tokenizer=tokenizer,
                                                                                       max_len=max_len,
                                                                                       batch_size=batch_size,
                                                                                       n_workers=n_workers,
                                                                                       labels_key=labels_key,
                                                                                       dataset_name=dataset_name,
                                                                                       print_tokens=print_tokens,
                                                                                       tag2id=tag2id)
    return merged_dataloader, tag2id, id2tag, scaling_dict


def construct_dataloader_and_mappers(raw_samples: List[Dict], tokenizer: BertTokenizerFast, max_len: int,
                                     batch_size: int, n_workers: int, labels_key: str, dataset_name: str,
                                     print_tokens: bool,
                                     tag2id: Dict[str, int] = None) -> Tuple[DataLoader, Dict[str, int], Dict[int, str],
                                                                             Dict[str, float]]:
    samples_ready = [{"tokens": [t["text"] for t in sentence["tokens"]], "text": sentence["text"],
                      "labels": sentence[labels_key]} for sentence in raw_samples]
    # 2. Compute proportion of labels
    scaling_dict = compute_tag_scaling(inp_training_samples=raw_samples, labels_key=labels_key)
    # 3. Compute tag2id id2tag mappers
    unique_tags = set(tag for doc in raw_samples for tag in doc[labels_key])
    if tag2id is None:
        tag2id = {tag: tag_id for tag_id, tag in enumerate(unique_tags)}
        tag2id["PAD"] = -100  # add padding label to -100 so it can be converted
    id2tag = {tag_id: tag for tag, tag_id in tag2id.items()}
    # 4. Make dataloader
    dataloader_ready = make_dataloader(inp_samples=samples_ready, batch_size=batch_size, inp_tokenizer=tokenizer,
                                       max_len=max_len, shuffle=True, n_workers=n_workers,
                                       tag2id=tag2id, dataset_name=dataset_name, print_tokens=print_tokens)
    return dataloader_ready, tag2id, id2tag, scaling_dict


def make_dataloader(inp_samples: List[Dict], batch_size: int, inp_tokenizer: BertTokenizerFast, max_len: int,
                    shuffle: bool, n_workers: int, tag2id: Dict[str, int],
                    dataset_name: str, print_tokens: bool) -> DataLoader:

    torch_dataset = process_mention_data(inp_samples=inp_samples, inp_tokenizer=inp_tokenizer, max_len=max_len,
                                         tag2id=tag2id, dataset_name=dataset_name, print_tokens=print_tokens)
    if shuffle:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)

    else:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers)

    return loader


def process_mention_data(inp_samples: List[Dict], inp_tokenizer: BertTokenizerFast, max_len: int,
                         tag2id: Dict[str, int], dataset_name: str, print_tokens: bool) -> Dataset:
    """
    Generates a pytorch dataset containing encoded tokens and labels
    """

    print(f"\n==== {dataset_name.upper()} set ====")
    print(f"Number of sentences : {len(inp_samples)}")

    texts = [sample["text"] for sample in inp_samples]
    labels = [sample["labels"] for sample in inp_samples]
    original_tokens = [sample["tokens"] for sample in inp_samples]
    if print_tokens:
        print_token_stats(all_tokens=original_tokens, dataset_name=dataset_name,
                          max_len=max_len, plot_histogram=True)

    labels = reshape_labels(inp_labels=labels, max_len=max_len)
    doc_encodings = inp_tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                                  return_overflowing_tokens=True)

    all_tokens = extract_all_tokens_withoutpad(doc_encodings.encodings)

    # print(f"Number of sentences : {len(all_tokens)}")

    check_labels_tokens_alignment(tokens=all_tokens, subword_labels=labels)

    print_few_mentions(all_tokens=all_tokens, labels=labels, n=5)

    encoded_labels = pad_and_encode_labels(all_labels=labels,
                                           max_len=max_len,
                                           tag2id=tag2id)

    torch_dataset = PKDataset(encodings=doc_encodings, labels=encoded_labels)

    return torch_dataset


def pad_and_encode_labels(all_labels: List[List[str]], max_len: int,
                          tag2id: Dict[str, int]):
    """PADS and transforms labels"""
    all_padded_labels = []
    for seq_lab in all_labels:
        padding = ["PAD"] * (max_len - len(seq_lab))
        padded_labels = seq_lab + padding
        assert len(padded_labels) == max_len
        all_padded_labels.append(padded_labels)

    encoded_labels = [[tag2id[label] for label in pl] for pl in all_padded_labels]
    return encoded_labels


def print_few_mentions(all_tokens, labels, n):
    i = 0
    for tokens, l in zip(all_tokens, labels):
        if i > n:
            break
        entity_tokens = bio_to_entity_tokens(l)
        if entity_tokens:
            i += 1
            for span in entity_tokens:
                mention = make_seq(tokens[span["token_start"]:span["token_end"] + 1])
                print(mention)


def print_token_stats(all_tokens: List[List[str]], dataset_name: str, max_len: int,
                      plot_histogram: bool = False):
    n_tokens = []
    sentences_with_more_than_max = 0
    overlflowing_sentences = []
    for tokens in all_tokens:
        nt = len(tokens)
        n_tokens.append(nt)
        if nt > max_len:
            sentences_with_more_than_max += 1
            overlflowing_sentences.append(make_seq(bert_tokens=tokens))
    if plot_histogram:
        plt.hist(n_tokens, bins=50)
        plt.axvline(x=256, color="red")
        plt.title(f"Number of tokens per sentence in the {dataset_name} set")
        plt.xlabel("# tokens")
        plt.ylabel("# sentences")
        plt.show()
        plt.close()
    print(f"There were {sentences_with_more_than_max} sentences with more than {max_len} tokens"
          f" ({round(sentences_with_more_than_max * 100 / len(all_tokens), 2)}%): ")
    for s in overlflowing_sentences:
        print(s)


def get_token_stats(docs_encodings: BatchEncoding, dataset_name: str, print_truncated: bool,
                    plot_histogram: bool = False):
    """
    Print some dataset statistics
    @param plot_histogram: whether to plot a token histogram
    @param docs_encodings: documents after passing through bert fast tokenizer
    @param dataset_name: name of the dataset e.g. training, valid, test
    @param print_truncated: whether to print some sentences from the ones that have been truncated
    (purely for visual inspection)
    @return:
    """
    overflowing_sentences = [x for x in docs_encodings.encodings if x.overflowing]
    print(f"Number of sentences overflowing in {dataset_name} set: {len(overflowing_sentences)} from "
          f"{len(docs_encodings.encodings)} "
          f"({round(len(overflowing_sentences) * 100 / len(docs_encodings.encodings), 2)}%)")

    number_of_bert_tokens = [len([token for token in doc.tokens if token != '[PAD]']) for doc in
                             docs_encodings.encodings if
                             '[PAD]' in doc.tokens]
    if plot_histogram:
        plt.hist(number_of_bert_tokens)
        plt.title(f"Number of bert tokens in the {dataset_name} set")
        plt.xlabel("# tokens")
        plt.ylabel("# sentences")
        plt.show()
        plt.close()

    if print_truncated:
        max_print = 5
        if len(overflowing_sentences) < 5:
            max_print = len(overflowing_sentences)
        example_sentences = overflowing_sentences[0:max_print]
        for example in example_sentences:
            print(make_seq(example.tokens))


def make_seq(bert_tokens: List[str]):
    """Very simple function to return a sequence (not the original one) given input bert tokens"""
    seq = bert_tokens[0]
    for tok in bert_tokens[1:]:
        if "##" in tok:
            seq += tok.replace("##", "")
        else:
            seq += f" {tok}"
    return seq


class PKDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def inspect_entities_tokenized(docs_encoded: BatchEncoding, labels_encoded: List,
                               tag2id: Dict[str, int], n: int):
    id2tag = {tag_id: tag for tag, tag_id in tag2id.items()}
    n_docs = len(docs_encoded.encodings)
    assert n_docs == len(labels_encoded)
    irrelevant_label = "-"
    for _ in range(n):
        sample_id = random.randint(0, n_docs)
        sample_tokens = docs_encoded.tokens(sample_id)
        sample_labels = [id2tag[x] if x != -100 else irrelevant_label for x in labels_encoded[sample_id]]
        sample_tokens_clean, sample_labels_clean = simplify_labels_and_tokens(sample_tokens=sample_tokens,
                                                                              sample_labels=sample_labels,
                                                                              irrelevant_label=irrelevant_label)

        entities_token_ids = get_entity_token_indices(sample_labels_clean)
        for entity_start, entity_end in entities_token_ids:
            entity_str = " ".join(sample_tokens_clean[entity_start:entity_end + 1])
            entity_str = entity_str.replace(" ##", "")
            print(entity_str)


def read_dataset(data_dir_inp: str, dataset_name: str) -> List[Dict]:
    file_name = f"{dataset_name}.jsonl"
    file_path = Path(os.path.join(data_dir_inp, file_name))
    return read_jsonl(file_path=file_path)


def check_labels_tokens_alignment(tokens: List[List[str]], subword_labels: List[List[str]]):
    for toks, seq_label in zip(tokens, subword_labels):
        if len(toks) != len(seq_label):
            raise ValueError(f"The number of tokens and the number of labels do not correspond.")


def encode_and_align_labels(tokenized_inputs: BatchEncoding, original_labels: List[List[str]],
                            tag2id: Dict[str, int]) -> List[List[int]]:
    """
    Returns original labels encoded in a numerical form and aligned with bert word-pieces. Adapted from
    https://huggingface.co/transformers/custom_datasets.html#token-classification-with-w-nut-emerging-entities
    If the original label of the token "@HuggingFace" was 3 and this token gets split by BERT into sub-words
    e.g.  ['@', 'hugging', '##face']
    This function would return
    ['@', 'hugging', '##face']
    [3, -100, -100]
    Assigning only the label to the first token and ignoring the subsequent sub-word pieces at training time.

    @param tokenized_inputs: Inputs tokenized by huggingface BertTokenizerFast
    @param original_labels: labels in the string form [["B-PK", "I-PK", "O-"], ["O-","B-PK", "I-PK"]]
    @param tag2id: Dictionary to convert string labels to numerical integers e.g. {"B-PK": 0, "I-PK": 1}
    @return: the original labels encoded in a numerical form and aligned with bert word-pieces
    """
    labels = [[tag2id[tag] for tag in doc] for doc in original_labels]  # transform string tags to numerical labels
    assert len(labels) == len(tokenized_inputs.offset_mapping)

    encoded_labels = []
    for i, (doc_labels, doc_offset) in tqdm(enumerate(zip(labels, tokenized_inputs.offset_mapping))):
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        if len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)]) == len(doc_labels):
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        else:
            # most likely a case of truncation in which the labels were not truncated
            assert doc_offset[-2] != (0, 0)  # assert that this is a truncation case
            n_exp_labels = len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])
            doc_labels = doc_labels[0:n_exp_labels]  #
            assert len(doc_labels) == len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])

        encoded_labels.append(doc_enc_labels.tolist())

    assert len(encoded_labels) == len(tokenized_inputs.encodings)

    return encoded_labels


def extract_all_tokens_withoutpad(docencodings: List[Encoding]):
    return [[t for t in enc.tokens if t != '[PAD]'] for enc in docencodings]


def reshape_labels(inp_labels: List[List[str]], max_len: int):
    reshaped_labels = []
    for seq_labels in inp_labels:
        if len(seq_labels) > max_len:
            base = 0
            ending = max_len - 1
            remaining_seq = seq_labels
            while len(remaining_seq) > max_len:
                new_labels = remaining_seq[base:ending] + ['O']  # append a final O since there will be an extra sep
                # token
                assert len(new_labels) == max_len
                reshaped_labels.append(new_labels)
                remaining_seq = ['O'] + remaining_seq[ending:]  # add the label for the new CLS token
            reshaped_labels.append(remaining_seq)
        else:
            reshaped_labels.append(seq_labels)
    return reshaped_labels


def compute_tag_scaling(inp_training_samples: List[Dict], labels_key: str) -> Dict[str, float]:
    all_training_tags = [tag for doc in inp_training_samples for tag in doc[labels_key]]
    tag_freqs = Counter(all_training_tags).most_common()
    most_freq_tag_count = tag_freqs[0][1]
    return {tmp_key: most_freq_tag_count / tmp_freq for tmp_key, tmp_freq in tag_freqs}
