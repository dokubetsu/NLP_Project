import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
import os
import time


st.set_page_config(layout="wide", page_title="Text Summarizer", page_icon="📄")


MODEL_PATH = "./t5-cnn-dailymail-finetuned/"


DEFAULT_ARTICLE = """
LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of young girls everywhere, Radcliffe recently declared he is no longer single. The London native turns 18 Monday, giving him access to funds estimated at £20 million ($41.1 million). Despite his access to the money, Radcliffe has said he has no plans to indulge in extravagance. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund. The final two movies in the series potentially mean additional millions for the young star. Radcliffe has an estimated £20 million fortune locked away until he turns 18 Monday. Publisher Bloomsbury reveals the book has sold more than 2.5 million copies in Britain in its first 24 hours. It has topped the charts in the United States, where it sold 8.3 million copies in the first day, distributors Scholastic said. Radcliffe has portrayed the schoolboy wizard in all five Potter films released so far and is set to star in the final two adaptations. He is currently filming "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, Potter mania continues to grip the world since the launch of the final book in the series on Saturday. It was revealed Monday that Radcliffe had donated an autographed pair of spectacles worn by his character during filming of "Harry Potter and the Chamber of Secrets" to an auction memorabilia site to raise money for charity. This is his fifth outing as the young wizard.
"""
T5_PREFIX = "summarize: "


@st.cache_data
def ensure_nltk_data():
    """Downloads necessary NLTK data ('punkt' and 'punkt_tab') if not already present."""
    resources_downloaded = []
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt', quiet=True)
        resources_downloaded.append('punkt')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        st.info("Downloading NLTK 'punkt_tab' data (required by TextRank)...")
        nltk.download('punkt_tab', quiet=True)
        resources_downloaded.append('punkt_tab')

    if resources_downloaded:
        st.success(f"NLTK resource(s) downloaded: {', '.join(resources_downloaded)}.")


ensure_nltk_data()
@st.cache_resource
def load_t5_model(model_dir):
    """Loads the T5 model and tokenizer."""
    start_load_time = time.time()
    try:
        if not os.path.isdir(model_dir):
            st.error(f"Model directory not found: {model_dir}")
            st.error("Please ensure the fine-tuned T5 model is saved in the correct location relative to app.py.")
            return None, None, None

        device = torch.device("cpu")
        st.info(f"Loading T5 model onto {device}... (This may take a moment on first run)")

        model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)

        end_load_time = time.time()
        load_duration = end_load_time - start_load_time
        st.success(f"T5 model '{os.path.basename(os.path.normpath(model_dir))}' loaded successfully on {device} in {load_duration:.2f}s.")
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading T5 model from {model_dir}: {e}")
        st.error("T5 summarization will be disabled.")
        return None, None, None


t5_model, t5_tokenizer, t5_device = load_t5_model(MODEL_PATH)
t5_available = t5_model is not None and t5_tokenizer is not None

def summarize_textrank_app(text, sentences_count=3):
    """Generates summary using TextRank."""
    try:
        parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count=sentences_count)
        summary = " ".join([str(sentence) for sentence in summary_sentences])
        return summary
    except Exception as e:
        st.error(f"Error during TextRank summarization: {e}")
        return "Failed to generate TextRank summary."


def summarize_t5_app(text, model, tokenizer, device, min_len=30, max_len=128):
    """Generates summary using the loaded T5 model."""
    if not t5_available:
        return "T5 model is not available."

    model.eval()
    input_text = T5_PREFIX + text

    try:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                num_beams=4,               
                max_length=max_len + 2,    
                min_length=min_len,
                length_penalty=2.0,        
                early_stopping=True        
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error during T5 summarization: {e}")
        return "Failed to generate T5 summary."



# --- Streamlit UI ---
st.title("📄 Comparative Text Summarization")
st.markdown("Explore and compare summaries generated by Extractive (**TextRank**) and Abstractive (**fine-tuned T5**) methods.")
st.markdown("---")

col1, col2 = st.columns([0.5, 0.5], gap="medium")


with col1:
    st.header("Input Article")
    if 'article_input' not in st.session_state:
        st.session_state.article_input = DEFAULT_ARTICLE
    article_text = st.text_area(
        "Paste your article here:",
        key="article_input",
        height=400
    )

    st.header("Settings")
    model_options = ["T5 (Fine-tuned)", "TextRank"]
    default_model_index = 0 if t5_available else 1
    model_choice = st.selectbox(
        "Choose Summarization Model:",
        model_options,
        key="model_select",
        index=default_model_index,
        disabled=(not t5_available and model_options.index("T5 (Fine-tuned)") == default_model_index)
    )

    if model_choice == "T5 (Fine-tuned)" and not t5_available:
         st.error("T5 model could not be loaded. Please select TextRank or check the `MODEL_PATH` in the script.")

    st.markdown(f"**{model_choice} Parameters:**")
    if model_choice == "T5 (Fine-tuned)":
        min_len = st.slider("Minimum Summary Length:", min_value=10, max_value=100, value=30, key="t5_minlen", disabled=not t5_available)
        max_len = st.slider("Maximum Summary Length:", min_value=50, max_value=250, value=150, key="t5_maxlen", disabled=not t5_available)
        st.caption("Adjust the desired length range for the abstractive T5 summary.")
    elif model_choice == "TextRank":
        sentence_count = st.slider("Number of sentences:", min_value=1, max_value=10, value=3, key="tr_sentences")
        st.caption("Select the exact number of sentences for the extractive TextRank summary.")

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        summarize_button = st.button(
            "✨ Summarize",
            key="summarize_btn",
            type="primary",
            use_container_width=True,
            disabled=(model_choice == "T5 (Fine-tuned)" and not t5_available)
        )
    with button_col2:
         if st.button("Clear All", key="clear_btn", use_container_width=True):
            st.session_state.article_input = ""
            if 'generated_summary' in st.session_state:
                st.session_state.generated_summary = ""
            st.rerun()


with col2:
    st.header("Generated Summary")
    summary_placeholder = st.empty()

    if 'generated_summary' not in st.session_state:
        st.session_state.generated_summary = ""

    if st.session_state.generated_summary:
        summary_placeholder.text_area(
            "Summary Output",
            value=st.session_state.generated_summary,
            height=350,
            key="summary_display_initial",
            disabled=True
            )
        st.caption("Select text and use Ctrl+C / Cmd+C to copy.")
    else:
         summary_placeholder.info("Summary will appear here after clicking 'Summarize'.")


if summarize_button:
    if not article_text or not article_text.strip():
        st.warning("Please paste some article text first.")
        st.session_state.generated_summary = ""
        st.rerun()
    else:
        start_gen_time = time.time()
        summary = ""
        with st.spinner(f"Generating {model_choice} summary..."):
            if model_choice == "T5 (Fine-tuned)" and t5_available:
                 summary = summarize_t5_app(article_text, t5_model, t5_tokenizer, t5_device, min_len=min_len, max_len=max_len)
            elif model_choice == "TextRank":
                 summary = summarize_textrank_app(article_text, sentences_count=sentence_count)

        end_gen_time = time.time()
        generation_time = end_gen_time - start_gen_time

        st.session_state.generated_summary = summary

        if summary and "Failed" not in summary and "not available" not in summary:
             summary_placeholder.text_area(
                 "Summary Output",
                 value=summary,
                 height=350,
                 key="summary_display_update",
                 disabled=True
                 )
             st.caption(f"Summary generated in {generation_time:.2f} seconds using {model_choice}.")
             st.caption("Select text and use Ctrl+C / Cmd+C to copy.")
        elif summary:
            summary_placeholder.error(summary)
        else:
            summary_placeholder.warning("Summary generation resulted in empty output.")





st.markdown("---")
st.markdown("Created for NLP Project: A Comparative Study of Text Summarization Techniques.")