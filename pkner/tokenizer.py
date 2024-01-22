import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_suffix_regex
from spacy.lang import char_classes

"""
Sequence idea: 
1. Split by unicode characters
2. Split by non-alphanumeric
3. Separate numeric and non-numeric parts
"""


def combined_rule_suffixes():
    quotes = char_classes.LIST_QUOTES.copy() + ["’"]
    suffix_punct = char_classes.PUNCT.replace("|", " ")
    suffixes_list = (
            char_classes.split_chars(suffix_punct)
            + char_classes.LIST_ELLIPSES
            + quotes
            + char_classes.LIST_ICONS
            + ["'s", "'S", "’s", "’S", "’s", "’S"]
            + [
                "[", "]",
                ",",
                r"[^a-zA-Z\d\s:]",  # all non-alphanum
                r"(?<=[0-9])([a-zA-Z]+|[^a-zA-Z\d\s:])",
                # r"(?<=[0-9])\D+", # digit + any non digit (handling unit separation)
                r"(?<=[0-9])\+",
                r"(?<=°[FfCcKk])\.",
                r"(?<=[0-9])(?:{})".format(char_classes.CURRENCY),
                # this is another place where we used a variable width lookbehind
                # so now things like 'H3g' will be tokenized as ['H3', 'g']
                # previously the lookbehind was (^[0-9]+)
                r"(?<=[0-9])(?:{u})".format(u=char_classes.UNITS),
                r"(?<=[0-9{}{}(?:{})])\.".format(
                    char_classes.ALPHA_LOWER, r"%²\-\)\]\+", "|".join(quotes)
                ),
                # add |\d to split off the period of a sentence that ends with 1D.
                r"(?<=[{a}|\d][{a}])\.".format(a=char_classes.ALPHA_UPPER),
            ]
    )

    return suffixes_list


def get_compiled_rex_rules():
    regex_non_alpha = r"[^a-zA-Z\d\s]|[0-9]+|[a-zA-Z]+"
    regex_suffixes = combined_rule_suffixes()
    prefix_re = re.compile(regex_non_alpha)
    infix_re = re.compile(regex_non_alpha)
    suffix_re = compile_suffix_regex(regex_suffixes)
    return prefix_re, infix_re, suffix_re


def create_tokenizer(nlp):
    prefix_re, infix_re, suffix_re = get_compiled_rex_rules()
    return Tokenizer(nlp.vocab, rules={},
                     prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     )


@spacy.registry.tokenizers("pk_tokenizer")
def create_pk_tokenizer():
    def create_tokenizer(nlp):
        prefix_re, infix_re, suffix_re = get_compiled_rex_rules()
        return Tokenizer(nlp.vocab, rules={},
                         prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         )

    return create_tokenizer
