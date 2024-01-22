from typing import List, Tuple, Dict
import torch
import gc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pkner.utils import bio_to_entity_tokens


def simplify_labels_and_tokens(sample_tokens: List[str], sample_labels: List[str],
                               irrelevant_label: str) -> Tuple[List[str], List[str]]:
    """

    @param sample_tokens: ["[CLS]", "hi", "##ho","hey", "huu", ".", "[SEP]", "[PAD]", "[PAD]"]
    @param sample_labels: ["-", "B-PK", "-","I-PK", "O", "O", "[SEP]", "[PAD]", "[PAD]"]
    @param irrelevant_label: "-"
    @return: ["hi", "##ho","hey", "huu", "."], ["B-PK", "I-PK","I-PK", "O", "O"]
    """
    sample_tokens_clean = []
    sample_labels_clean = []
    prev_label = None
    for token, label in zip(sample_tokens, sample_labels):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            sample_tokens_clean.append(token)
            if label == irrelevant_label:
                if "B" in prev_label:
                    new_label = "I-" + prev_label.split("-")[1]
                else:
                    new_label = prev_label
                sample_labels_clean.append(new_label)
                prev_label = new_label
            else:
                sample_labels_clean.append(label)
                prev_label = label
    assert len(sample_tokens_clean) == len(sample_labels_clean)
    return sample_tokens_clean, sample_labels_clean


def get_entity_token_indices(inp_sequence: List[str]) -> List[Tuple[int, int]]:
    """
    Gets an input list of tag-bio strings in the form of e.g.
    input = ['O', 'B-PK', 'I-PK', 'I-PK', 'I-PK', 'I-PK', 'O', 'O', 'O', 'O', 'B-PK', 'I-PK', 'I-PK', 'I-PK',"O"]
    And returns the start and end indexes of spans in the form of tuples. In this case there is only 1 entity:
    output = [(1,5),(10,13)]
    Important: it assumes "[CLS]", "[SEP]" and "[PAD]" token labels have been removed and the sequence has been passed
    through simplify_labels_and_tokens
    """
    out_list = []
    inside_entity = False
    start_entity_idx = None
    end_entity_idx = None
    for i, tagg in enumerate(inp_sequence):
        if len(inp_sequence) - 1 == i:  # last token
            if "B" in tagg:
                out_list.append((i, i))
            if "I" in tagg:
                out_list.append((start_entity_idx, i))
            if "O" in tagg:
                if inside_entity:
                    out_list.append((start_entity_idx, end_entity_idx))
        else:
            if "B" in tagg:
                if inside_entity:
                    out_list.append((start_entity_idx, end_entity_idx))
                else:
                    inside_entity = True
                start_entity_idx = i
                end_entity_idx = i
            if "O" in tagg:
                if inside_entity:
                    out_list.append((start_entity_idx, end_entity_idx))
                inside_entity = False
            if "I" in tagg:
                end_entity_idx = i

    return out_list


def get_f1(p, r):
    if p + r == 0.:
        return 0.
    else:
        return (2 * p * r) / (p + r)


def get_metrics(inp_dict):
    p = inp_dict['precision']
    r = inp_dict['recall']
    if "f1" in inp_dict.keys():
        f1 = inp_dict['f1']
    else:
        f1 = get_f1(p=p, r=r)
    return torch.FloatTensor([p]), torch.FloatTensor([r]), torch.FloatTensor([f1])


def get_predicted_entity_offsets(model_logits: torch.Tensor, inp_batch: Dict[str, torch.Tensor],
                                 id2tag: Dict[int, str]):
    predictions = model_logits.argmax(dim=2)

    tag_predictions = [[id2tag[prediction.item()] for mask, prediction in zip(att_masks, id_preds) if mask.item() == 1]
                       for att_masks, id_preds in zip(inp_batch["attention_mask"], predictions)]

    entity_tokens = [bio_to_entity_tokens(tag_prediction) for tag_prediction in tag_predictions]

    outputs = []
    for offsets, entities in zip(inp_batch["offset_mapping"], entity_tokens):
        tmp_outputs = []
        for entity in entities:
            entity["start"] = int(offsets[entity["token_start"]][0])
            entity["end"] = int(offsets[entity["token_end"]][1])
            tmp_outputs.append(entity)
        outputs.append(tmp_outputs)

    return outputs


def predict_bio_tags(model_logits: torch.Tensor, inp_batch: Dict[str, torch.Tensor],
                     id2tag: Dict[int, str]):
    predictions = model_logits.argmax(dim=2)

    tag_predictions = [[id2tag[prediction.item()] for mask, prediction in zip(att_masks, id_preds) if mask.item() == 1]
                       for att_masks, id_preds in zip(inp_batch["attention_mask"], predictions)]

    return tag_predictions


def empty_cuda_cache(n_gpus: int):
    torch.cuda.empty_cache()
    gc.collect()
    for x in range(0, n_gpus):
        with torch.cuda.device(x):
            torch.cuda.empty_cache()


class PKDatasetInference(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["sentence_idx"] = idx
        return item

    def __len__(self):
        return len(self.encodings.encodings)


def predict_pl_bert_ner(inp_texts, inp_model, inp_tokenizer, batch_size, n_workers):
    encodings = inp_tokenizer(inp_texts, padding=True, truncation=True, max_length=inp_model.seq_len,
                              return_offsets_mapping=True, return_overflowing_tokens=True)
    predict_dataset = PKDatasetInference(encodings=encodings)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, num_workers=n_workers)
    inp_model.eval()
    predicted_entities = []
    overflow_to_sample = []
    all_seq_end = []

    for idx, batch in tqdm(enumerate(predict_loader)):
        with torch.no_grad():
            batch_logits = inp_model(input_ids=batch['input_ids'],
                                     attention_masks=batch['attention_mask']).to('cpu')

            batch_predicted_entities = predict_bio_tags(model_logits=batch_logits, inp_batch=batch,
                                                        id2tag=inp_model.id2tag)

        for seq_end, omap in zip(batch['offset_mapping'], batch['overflow_to_sample_mapping']):
            all_seq_end.append(seq_end.flatten().max().item())
            overflow_to_sample.append(omap.item())

        predicted_entities += batch_predicted_entities

    predicted_entities = remap_overflowing_entities(predicted_tags=predicted_entities, all_seq_end=all_seq_end,
                                                    overflow_to_sample=overflow_to_sample, original_texts=inp_texts,
                                                    offset_mappings=encodings["offset_mapping"]
                                                    )
    return predicted_entities


def remap_overflowing_entities(predicted_tags: List[List[str]], all_seq_end: List[int], overflow_to_sample: List[int],
                               original_texts: List[str], offset_mappings: List[List[Tuple[int, int]]]) -> List[
    List[Dict]
]:
    if len(set(overflow_to_sample)) == len(overflow_to_sample):  # case with no overflowing tokens
        assert len(set(overflow_to_sample)) == len(original_texts)
        tags_per_sentence = predicted_tags
        offset_mappings_rearranged = offset_mappings
    else:
        # Case in which we have overflowing indices
        assert len(all_seq_end) == len(predicted_tags)
        print("Remapping Overflowing")

        all_o_to_s = []
        tags_per_sentence = []
        offset_mappings_rearranged = []
        for i, (ents, send, o_to_s, offsets) in enumerate(zip(predicted_tags, all_seq_end, overflow_to_sample,
                                                              offset_mappings)):
            if o_to_s not in all_o_to_s:
                # tags original sentence
                all_o_to_s.append(o_to_s)
                offset_mappings_rearranged.append(offsets)
                tags_per_sentence.append(ents)
            else:
                # overflowing tags
                new_offsets = offset_mappings_rearranged[-1] + offsets
                new_entities = tags_per_sentence[-1] + ents

                tags_per_sentence = tags_per_sentence[:-1]  # remove last element
                offset_mappings_rearranged = offset_mappings_rearranged[:-1]

                offset_mappings_rearranged.append(new_offsets)
                tags_per_sentence.append(new_entities)  # re-append last element + new one

            assert len(all_o_to_s) == len(set(all_o_to_s))

    entity_tokens = [bio_to_entity_tokens(tag_prediction) for tag_prediction in tags_per_sentence]

    assert len(tags_per_sentence) == len(original_texts) == len(entity_tokens) == len(offset_mappings_rearranged)

    outputs = []
    for offsets, entities, tag in zip(offset_mappings_rearranged, entity_tokens, tags_per_sentence):
        tmp_outputs = []
        for entity in entities:
            entity["start"] = int(offsets[entity["token_start"]][0])
            entity["end"] = int(offsets[entity["token_end"]][1])
            entity["tags"] = tag
            tmp_outputs.append(entity)
        outputs.append(tmp_outputs)

    return outputs
