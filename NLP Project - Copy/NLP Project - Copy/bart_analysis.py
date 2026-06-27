import json
import os
import time
import nltk
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


T5_MODEL_PATH = "./t5-cnn-dailymail-finetuned/"
BART_MODEL_NAME = "facebook/bart-large-cnn"
NUM_SAMPLES_FOR_ANALYSIS = 20

OUTPUT_FILE = "qualitative_analysis_data_with_bart.json"

T5_PREFIX = "summarize: "
T5_MAX_INPUT_LENGTH = 512
T5_MAX_TARGET_LENGTH = 150
T5_MIN_TARGET_LENGTH = 30

BART_MAX_TARGET_LENGTH = 150
BART_MIN_TARGET_LENGTH = 30

TEXTRANK_SENTENCE_COUNT = 3

INFERENCE_DEVICE_ID = -1


@torch.no_grad()
def ensure_nltk_data():
    """Downloads necessary NLTK data ('punkt' and 'punkt_tab') if not already present."""
    resources_downloaded = []
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("INFO: Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt', quiet=True)
        resources_downloaded.append('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("INFO: Downloading NLTK 'punkt_tab' data (required by TextRank)...")
        nltk.download('punkt_tab', quiet=True)
        resources_downloaded.append('punkt_tab')
    if resources_downloaded:
        print(f"INFO: NLTK resource(s) downloaded: {', '.join(resources_downloaded)}.")


@torch.no_grad()
def load_inference_model(model_dir):
    """Loads T5 model and tokenizer for inference, preferring CPU."""
    try:
        if not os.path.isdir(model_dir):
            print(f"ERROR: T5 Model directory not found: {model_dir}")
            return None, None, None
        device = torch.device("cpu" if INFERENCE_DEVICE_ID == -1 else f"cuda:{INFERENCE_DEVICE_ID}")
        print(f"INFO: Loading T5 model onto {device} for inference...")
        model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
        model.eval()
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        print(f"INFO: T5 model '{os.path.basename(os.path.normpath(model_dir))}' loaded successfully on {device}.")
        return model, tokenizer, device
    except Exception as e:
        print(f"ERROR: Failed to load T5 model from {model_dir}: {e}")
        return None, None, None


@torch.no_grad()
def summarize_textrank(text, sentences_count=3):
    """Generates summary using TextRank."""
    try:
        parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count=sentences_count)
        summary = " ".join([str(sentence) for sentence in summary_sentences])
        return summary
    except Exception as e:
        print(f"WARNING: TextRank failed for an article: {e}")
        return f"Error during TextRank: {e}"

@torch.no_grad()
def summarize_t5(text, model, tokenizer, device):
    """Generates summary using the loaded T5 model."""
    if model is None or tokenizer is None:
        return "T5 Model Not Available"
    input_text = T5_PREFIX + text
    try:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=T5_MAX_INPUT_LENGTH, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        summary_ids = model.generate(
            inputs['input_ids'], num_beams=4, max_length=T5_MAX_TARGET_LENGTH + 2,
            min_length=T5_MIN_TARGET_LENGTH, length_penalty=2.0, early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"WARNING: T5 inference failed for an article: {e}")
        return f"Error during T5 inference: {e}"

@torch.no_grad()
def summarize_bart(text, summarizer_pipeline):
    """Generates summary using the loaded BART pipeline."""
    if summarizer_pipeline is None:
        return "BART Pipeline Not Available"
    try:
        results = summarizer_pipeline(
            text,
            max_length=BART_MAX_TARGET_LENGTH,
            min_length=BART_MIN_TARGET_LENGTH,
            do_sample=False
        )
        if results and isinstance(results, list) and 'summary_text' in results[0]:
            return results[0]['summary_text']
        else:
            print("WARNING: Unexpected BART output format:", results)
            return "Error: Unexpected BART output format"
    except Exception as e:
        print(f"WARNING: BART inference failed for an article: {e}")
        return f"Error during BART inference: {e}"


if __name__ == "__main__":
    print("Starting script to generate data for qualitative analysis (including BART)...")

    ensure_nltk_data()

    t5_model, t5_tokenizer, t5_device = load_inference_model(T5_MODEL_PATH)
    t5_available = t5_model is not None

    bart_pipeline = None
    print(f"Loading BART summarization pipeline ({BART_MODEL_NAME})...")
    try:
        bart_pipeline = pipeline("summarization", model=BART_MODEL_NAME, device=INFERENCE_DEVICE_ID)
        print("BART Pipeline loaded successfully.")
        bart_available = True
    except Exception as e:
        print(f"ERROR: Failed to load BART pipeline: {e}")
        bart_available = False

    print(f"Loading CNN/Daily Mail dataset (test split)...")
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=f'test[:{NUM_SAMPLES_FOR_ANALYSIS}]')
        print(f"Loaded {len(dataset)} samples for analysis.")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        exit()

    qualitative_data = []
    print(f"\nProcessing {len(dataset)} articles...")
    start_time = time.time()

    for i, example in enumerate(dataset):
        print(f"  Processing article {i+1}/{len(dataset)} (ID: {example.get('id', 'N/A')})...")
        article_text = example.get('article', '')
        reference_summary = example.get('highlights', '')

        tr_summary = summarize_textrank(article_text, sentences_count=TEXTRANK_SENTENCE_COUNT)

        t5_summary = summarize_t5(article_text, t5_model, t5_tokenizer, t5_device) if t5_available else "T5 Model Not Loaded"

        bart_summary = summarize_bart(article_text, bart_pipeline) if bart_available else "BART Pipeline Not Loaded"

        qualitative_data.append({
            "id": example.get('id', f'sample_{i}'),
            "article": article_text,
            "reference_summary": reference_summary,
            "textrank_summary": tr_summary,
            "t5_summary": t5_summary,
            "bart_summary": bart_summary
        })

    end_time = time.time()
    print(f"\nFinished processing {len(dataset)} articles in {end_time - start_time:.2f} seconds.")

    print(f"Saving data to '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(qualitative_data, f, ensure_ascii=False, indent=4)
        print("Data saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save data to JSON: {e}")

    print("\nScript finished. You can now review the generated JSON file (including BART summaries) for qualitative analysis.")