import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import ujson
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from spacy import Language
from spacy.tokens.doc import Doc
from spacy.training import offsets_to_biluo_tags
from termcolor import colored
from sty import fg
from nervaluate import Evaluator
from nltk.stem import PorterStemmer


def read_jsonl(file_path: Path):
    # Taken from prodigy support
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def write_jsonl(file_path: Path, lines: Iterable):
    # Taken from prodigy
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))


def character_annotations_to_spacy_doc(inp_annotation: Dict, inp_model: Language) -> Tuple[Doc, bool]:
    """
    Converts an input sentence annotated at the character level for NER to a spaCy doc object
    It assumes that the inp_annotation has:
        1. "text" field
        2. "spans" field with a list of NER annotations in the form of  {"start": <ch_idx>, "end": <ch_idx>,
        "label": <NER label name>}
    """
    text = inp_annotation["text"]  # extra
    doc = inp_model.make_doc(text)  # extra
    ents = []  # extra
    misaligned = False
    if "spans" in inp_annotation.keys():
        for entities_sentence in inp_annotation["spans"]:
            start = entities_sentence["start"]
            end = entities_sentence["end"]
            label = entities_sentence["label"]
            span = doc.char_span(start, end, label=label)
            if span is None:
                misaligned = True
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character" \
                      f" span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n "
                warnings.warn(msg)
            else:
                ents.append(span)
    doc.ents = ents
    return doc, misaligned


def get_iob_labels(inp_doc: Doc) -> List[str]:
    return [token.ent_iob_ + "-" + token.ent_type_ if token.ent_type_ else token.ent_iob_ for token in inp_doc]


def get_biluo_labels(inp_doc: Doc) -> List[str]:
    ch_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in inp_doc.ents]
    return offsets_to_biluo_tags(inp_doc, ch_entities)


def check_predictions_vs_labels(inp_labels: List, inp_predictions: List):
    uq_labels = set(list(itertools.chain(*inp_labels)))
    uq_predictions = set(list(itertools.chain(*inp_predictions)))
    assert uq_labels == uq_predictions
    print(f"The token labels for this corpus are:\n{uq_labels}")
    for labels, predictions in zip(inp_labels, inp_predictions):
        assert len(labels) == len(predictions)


def print_ner_scores(inp_dict: Dict):
    """

    @param inp_dict: Dictionary with keys corresponding to entity types and subkeys to metrics
    e.g. {'PK': {'ent_type': {..},{'partial': {..},{'strict': {..} }}
    @return: Prints summary of metrics
    """
    for ent_type in inp_dict.keys():
        print(f"====== Stats for entity {ent_type} ======")
        for metric_type in inp_dict[ent_type].keys():
            if metric_type in ['partial', 'strict']:
                print(f" === {metric_type} match: === ")
                precision = inp_dict[ent_type][metric_type]['precision']
                recall = inp_dict[ent_type][metric_type]['recall']
                f1 = inp_dict[ent_type][metric_type]['f1']
                p = round(precision * 100, 2)
                r = round(recall * 100, 2)
                f1 = round(f1 * 100, 2)
                print(f" Precision:\t {p}%")
                print(f" Recall:\t {r}%")
                print(f" F1:\t\t {f1}%")


def get_ner_scores(pred_ents_ch: List[List[Dict]], true_ents_ch: List[List[Dict]], inp_tags: List[str],
                   original_annotations: List[Dict],
                   display_errors: bool, display_all: bool = False):
    """

    @param pred_ents_ch: entities predicted at the character level expressed as:
    [
    [{'start': 3, 'end': 12, 'label': 'PK'},
    {'start': 17, 'end': 39, 'label': 'PK'}].
    [],
    [{'start': 15, 'end': 18, 'label': 'PK'}]
    ] one list per sentence
    @param inp_tags: list of entities to consider. e.g., ["PK", "VALUE", "UNITS"]
    @param true_ents_ch: true entities in the same format
    @param original_annotations: Original annotations with _task_hash and text
    @param display_errors: whether to display prediction errors in the terminal window
    @return: prints the results
    """
    assert len(pred_ents_ch) == len(true_ents_ch)
    evaluator = Evaluator(true_ents_ch, pred_ents_ch, tags=inp_tags)
    _, results_agg = evaluator.evaluate()

    print("\n===== Printing discrepancies between annotations and model prediction =====")

    if display_errors or display_all:
        i = 0
        for instance, predicted_ent, true_ent in zip(original_annotations, pred_ents_ch, true_ents_ch):
            sentence_text = instance["text"]
            if predicted_ent != true_ent or display_all:
                i += 1
                instance["_task_hash"] = 8888 if "_task_hash" not in instance.keys() else instance["_task_hash"]
                print(10 * "=", f"Example with task hash {instance['_task_hash']} n={i}", 10 * "=")
                print("REAL LABELS:")
                print(view_all_entities_terminal(inp_text=sentence_text, character_annotations=true_ent))
                print("MODEL PREDICTIONS:")
                print(view_all_entities_terminal(inp_text=sentence_text, character_annotations=predicted_ent))

    print_ner_scores(inp_dict=results_agg)


def get_tensorboard_logger(log_dir: str, run_name: str) -> LightningLoggerBase:
    return TensorBoardLogger(save_dir=log_dir, name="tensorboard-logs-{}".format(run_name))


def clean_excluded_terms(inp_tokens, inp_iob, inp_biluo, excluded_terms):
    assert len(inp_tokens) == len(inp_iob) == len(inp_biluo)
    out_tokens = []
    out_iobs = []
    out_biluo = []
    for token, iob_lab, biluo_lab in zip(inp_tokens, inp_iob, inp_biluo):
        if token not in excluded_terms:
            out_tokens.append(token)
            out_iobs.append(iob_lab)
            out_biluo.append(biluo_lab)
    return out_tokens, out_iobs, out_biluo


def is_azure_location(filepath: str) -> bool:
    if "blob.core.windows.net" in filepath:
        return True
    return False


def view_entities_terminal(inp_text: str, character_annotation: Dict):
    text_left = inp_text[0:character_annotation['start']]
    mention_text = colored(inp_text[character_annotation['start']:character_annotation['end']],
                           'green', attrs=['reverse', 'bold'])
    text_right = inp_text[character_annotation['end']:]
    all_text = text_left + mention_text + text_right
    return all_text


def view_all_entities_terminal(inp_text: str, character_annotations: List[Dict]):
    if character_annotations:
        character_annotations = sorted(character_annotations, key=lambda anno: anno['start'])
        sentence_text = ""
        end_previous = 0
        for annotation in character_annotations:
            sentence_text += inp_text[end_previous:annotation["start"]]
            sentence_text += colored(inp_text[annotation["start"]:annotation["end"]],
                                     'green', attrs=['reverse', 'bold'])
            end_previous = annotation["end"]
        sentence_text += inp_text[end_previous:]
        return sentence_text
    return inp_text


def print_entities_terminal(original_sentence: str, span_list: List[Dict]):
    highlighted_text = ""
    end_previous = 0
    for ent in span_list:
        highlighted_text += original_sentence[end_previous:ent["start"]]
        new_ent_text = fg.orange + original_sentence[ent["start"]:ent["end"]] + fg.rs
        highlighted_text = highlighted_text + new_ent_text
        end_previous = ent["end"]
    highlighted_text += original_sentence[end_previous:]
    print(highlighted_text)


def bio_to_entity_tokens(inp_bio_seq: List[str]) -> List[Dict]:
    """
    Gets as an input a list of BIO tokens and returns the starting and end tokens of each span
    @return: The return should be a list of dictionary spans in the form of [{"token_start": x,"token_end":y,"label":""]
    """
    out_spans = []

    b_toks = sorted([i for i, t in enumerate(inp_bio_seq) if "B-" in t])  # Get the indexes of B tokens
    sequence_len = len(inp_bio_seq)
    for start_ent_tok_idx in b_toks:
        entity_type = inp_bio_seq[start_ent_tok_idx].split("-")[1]
        end_ent_tok_idx = start_ent_tok_idx + 1
        if start_ent_tok_idx + 1 < sequence_len:  # if it's not the last element in the sequence
            for next_token in inp_bio_seq[start_ent_tok_idx + 1:]:
                if next_token.split("-")[0] == "I" and next_token.split("-")[1] == entity_type:
                    end_ent_tok_idx += 1
                else:
                    break
        out_spans.append(dict(token_start=start_ent_tok_idx, token_end=end_ent_tok_idx - 1, label=entity_type))
    return out_spans


def clean_instance_span(instance_spans: Dict):
    return [dict(start=x['start'], end=x['end'], label=x['label']) for x in instance_spans]


def extract_main_section_std(inp_section_list: List[str]) -> str:
    """
    ['', '', '3. Results and Discussion'] -> 'results and discussion'
    """
    ps = PorterStemmer()
    r = ""
    for subsec in inp_section_list:
        if subsec and subsec.strip() != "":
            subsec = "".join([ch for ch in subsec if ch == " " or ch.isalpha()])
            subsec = subsec.strip().lower()
            subsec = " ".join([ps.stem(w) for w in subsec.split()])
            return subsec
    return r


def extract_main_section(inp_annotation: Dict) -> str:
    if 'metadata' in inp_annotation.keys():
        metadata_keys = inp_annotation['metadata'].keys()
        if 'istitle' in metadata_keys and inp_annotation['metadata']['istitle']:
            return "title"
        if "sections" in metadata_keys:
            if inp_annotation['metadata']['sections'] != ["", "", ""]:
                return extract_main_section_std(inp_section_list=inp_annotation['metadata']['sections'])
            else:
                return "fulltext"

        return "abstract"
    return "unknown"
