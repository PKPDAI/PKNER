import warnings
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
from tokenizers import Encoding
from pkner.utils import view_entities_terminal


def filter_duplicates_by_key(inp_list: List[Dict], inp_key: str) -> List[Dict]:
    """
    Filter dictionaries in the input list that have the same inp_key
    """
    out_list = []
    uq_keys = []
    for example in inp_list:
        if example[inp_key] not in uq_keys:
            out_list.append(example)
            uq_keys.append(example[inp_key])
    return out_list


def align_tokens_and_annotations_bilou(tokenized: Encoding, annotations: Dict, example: Dict):
    tokens = tokenized.tokens
    aligned_labels_bio = ["O"] * len(tokens)  # Make a list to store our labels the same length as our tokens
    aligned_labels_bilou = ["O"] * len(tokens)

    entity_tokens = []
    for anno in annotations:
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        for char_ix in range(anno["start"], anno["end"]):

            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)

        if not check_correct_alignment(tokenized=tokenized, entity_token_ids=annotation_token_ix_set, annotation=anno):
            example['_task_hash'] = example['_task_hash'] if 'task_hash' in example.keys() else ""
            warnings.warn(f"Careful, some character-level annotations did not align correctly with BERT tokenizer in "
                          f"example with task hash {example['_task_hash']}:"
                          f"\n{view_entities_terminal(example['text'], anno)}")
            print(example["_task_hash"])

        entity_tokens.append(
            dict(start=anno["start"],
                 end=anno["end"],
                 token_start=min(annotation_token_ix_set),
                 token_end=max(annotation_token_ix_set),
                 label=anno["label"]
                 )
        )

        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            # bilou
            prefix = "U"  # This annotation spans one token so is prefixed with U for unique
            aligned_labels_bilou[token_ix] = f"{prefix}-{anno['label']}"
            # bio
            prefix = "B"
            aligned_labels_bio[token_ix] = f"{prefix}-{anno['label']}"

        else:

            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_ix:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels_bilou[token_ix] = f"{prefix}-{anno['label']}"
                if prefix == "L":
                    prefix = "I"
                aligned_labels_bio[token_ix] = f"{prefix}-{anno['label']}"
    return aligned_labels_bilou, aligned_labels_bio, entity_tokens


def check_correct_alignment(tokenized: Encoding, entity_token_ids: set, annotation: Dict):
    """Checks that the original character-level annotations for an entity correspond to the start and end character
    of bert-tokens """

    orig_start_ent_char = annotation["start"]
    orig_end_ent_char = annotation["end"]

    start_char_bert_ent = tokenized.offsets[min(entity_token_ids)][0]
    end_char_bert_ent = tokenized.offsets[max(entity_token_ids)][1]

    if orig_start_ent_char == start_char_bert_ent and orig_end_ent_char == end_char_bert_ent:
        return True
    return False


def get_sort_key(tmp_span):
    return tmp_span['end'] - tmp_span['start'], -tmp_span['start']


def resolve_overlapping_spans(inp_spans: List[Dict]) -> List[Dict]:
    """Adapted from spacy.util.filter_spans"""
    sorted_spans = sorted(inp_spans, key=get_sort_key, reverse=True)  # sorts spans by length
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span["start"] not in seen_tokens and span["end"] - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span["start"], span["end"]))  # covers the range of characters occupied by that entity
    result = sorted(result, key=lambda tmp_span: tmp_span["start"])
    return result


def has_child_param(inp_el):
    if inp_el.has_attr('sem'):
        for ch in inp_el:
            if not isinstance(ch, str):
                if ch.has_attr('sem'):
                    if ch['sem'] == inp_el['sem']:
                        return True
    return False


def get_offsets_element(soup_element: BeautifulSoup, base_parent: BeautifulSoup) -> Tuple[int, int]:
    beginning = 0
    for subel in soup_element.previous_elements:
        if isinstance(subel, str):
            beginning += len(subel)
        else:
            if subel == base_parent:
                break
    end = beginning + len(soup_element.text)

    return beginning, end


def extract_pk_entities(inp_soup_obj: BeautifulSoup) -> List[Dict]:
    records = []
    for sentence_annot in inp_soup_obj.findAll("sentence"):
        out_record = dict(text=sentence_annot.text)
        spans = []
        for tag_name in ["G_IVIVPARA", "G_IVITPARA"]:
            for parameter in sentence_annot.findAll("cons", {"sem": tag_name}):
                if not has_child_param(inp_el=parameter):  # we found overlapping PK parameters annotated
                    begin, end = get_offsets_element(soup_element=parameter, base_parent=sentence_annot)
                    new_span = dict(start=begin, end=end, label="PK")
                    if new_span not in spans and begin != end:
                        spans.append(dict(start=begin, end=end, label="PK"))
        spans = resolve_overlapping_spans(inp_spans=spans)
        out_record["spans"] = spans
        records.append(out_record)
    return records
