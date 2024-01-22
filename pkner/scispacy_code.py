from typing import Callable
import spacy
from spacy.language import Language
from scispacy.custom_tokenizer import combined_rule_tokenizer


@spacy.registry.callbacks("replace_tokenizer")
def replace_tokenizer_callback() -> Callable[[Language], Language]:
    def replace_tokenizer(nlp: Language) -> Language:
        nlp.tokenizer = combined_rule_tokenizer(nlp)
        return nlp

    return replace_tokenizer
