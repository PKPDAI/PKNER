import os
from pathlib import Path
import typer
from transformers import BertTokenizerFast
from pkner.models.bert import load_pretrained_model
from pkner.utils import read_jsonl, clean_instance_span, get_ner_scores
from pkner.models.utils import predict_pl_bert_ner


def main(
        model_checkpoint: Path = typer.Option(
            default="results/checkpoints/pkbert-epoch=0013-val_f1_strict=0.94.ckpt",
            help="Path to the input model"),

        predict_file_path: Path = typer.Option(default="data/gold/test.jsonl",
                                               help="Path to the jsonl file of the test/evaluation set"),

        display_errors: bool = typer.Option(default=True, help="Whether to display sentences with errors"),

        batch_size: int = typer.Option(default=64, help="Batch size"),

        gpu: bool = typer.Option(default=True, help="Whether to use GPU for inference"),

        n_workers: int = typer.Option(default=12, help="Number of workers to use for the dataloader"),

):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # ============== 1. Load model and tokenizer ========================= #
    pl_model = load_pretrained_model(model_checkpoint_path=model_checkpoint, gpu=gpu)
    tokenizer = BertTokenizerFast.from_pretrained(pl_model.bert.name_or_path)

    # ============= 2. Load corpus  ============================ #
    predict_sentences = list(read_jsonl(file_path=predict_file_path))
    true_entities = [clean_instance_span(x["spans"]) for x in predict_sentences]
    texts_to_predict = [sentence["text"] for sentence in predict_sentences]

    # ============= 4. Predict  ============================ #
    predicted_entities = predict_pl_bert_ner(inp_texts=texts_to_predict, inp_model=pl_model, inp_tokenizer=tokenizer,
                                             batch_size=batch_size, n_workers=n_workers)

    predicted_entities_offsets = [clean_instance_span(x) for x in predicted_entities]

    # ============= 5. Evaluate  ============================ #

    get_ner_scores(pred_ents_ch=predicted_entities_offsets, true_ents_ch=true_entities, inp_tags=["PK"],
                   original_annotations=predict_sentences, display_errors=display_errors)


if __name__ == "__main__":
    typer.run(main)
