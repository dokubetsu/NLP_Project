import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


RESULTS_FILE_PATH = "evaluation_results.json"
PLOT_OUTPUT_DIR = "analysis_plots"
SAVE_PLOTS = True


if SAVE_PLOTS:
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

def load_results(filepath):
    """Loads evaluation results from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        print(f"Successfully loaded results from: {filepath}")
        return results
    except FileNotFoundError:
        print(f"Error: Results file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading results: {e}")
        return None

def analyze_and_visualize(results):
    """Analyzes results and creates visualizations."""
    if results is None:
        print("Analysis skipped due to missing results.")
        return

    textrank_rouge = results.get('textrank_rouge', {})
    t5_rouge = results.get('t5_rouge', {})
    avg_textrank_time = results.get('avg_textrank_time_sec', 0)
    avg_t5_time = results.get('avg_t5_time_sec', 0)
    t5_model_name = os.path.basename(results.get('t5_model_used', 'T5 (Unknown)'))

    data = {
        'Metric': [
            'ROUGE-1',
            'ROUGE-2',
            'ROUGE-L',
            'Avg Gen Time (s)'
        ],
        'TextRank': [
            textrank_rouge.get('rouge1', 0),
            textrank_rouge.get('rouge2', 0),
            textrank_rouge.get('rougeL', 0),
            avg_textrank_time
        ],
        'T5': [
            t5_rouge.get('rouge1', 0),
            t5_rouge.get('rouge2', 0),
            t5_rouge.get('rougeL', 0),
            avg_t5_time
        ]
    }
    df = pd.DataFrame(data)
    df.set_index('Metric', inplace=True)

    print("\n--- Evaluation Results Summary ---")

    df_display = df.copy()
    for col in ['TextRank', 'T5']:
        if col in df_display.columns:
             df_display[col] = df_display[col].apply(
                 lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and x > 1 else f"{x:.4f}"
             )

    print(df_display)
    print("-" * 35)
    print(f"T5 Model Analyzed: {t5_model_name}")
    print(f"Test Samples: {results.get('num_test_samples', 'N/A')}")
    print(f"Device Used (Training): {results.get('training_device', 'N/A')}")
    print("-" * 35)

    df_rouge_scores = df.loc[['ROUGE-1', 'ROUGE-2', 'ROUGE-L']]

    plt.style.use('seaborn-v0_8-talk')

    fig_rouge, ax_rouge = plt.subplots(figsize=(10, 6))
    df_rouge_scores.plot(kind='bar', ax=ax_rouge, rot=0, width=0.8)

    ax_rouge.set_title(f'ROUGE Score Comparison\n(T5 Model: {t5_model_name})', fontsize=16)
    ax_rouge.set_ylabel('ROUGE Score (F1)', fontsize=12)
    ax_rouge.set_xlabel('Metric', fontsize=12)
    ax_rouge.legend(title='Model', title_fontsize='11', fontsize='10')
    ax_rouge.grid(axis='y', linestyle='--', alpha=0.7)
    ax_rouge.tick_params(axis='both', which='major', labelsize=11)


    for container in ax_rouge.containers:
        ax_rouge.bar_label(container, fmt='%.2f', label_type='edge', fontsize=9, padding=3)

    plt.tight_layout()
    if SAVE_PLOTS:
        plot_path = os.path.join(PLOT_OUTPUT_DIR, "rouge_score_comparison.png")
        plt.savefig(plot_path)
        print(f"ROUGE score plot saved to: {plot_path}")
    plt.show()


    df_times = df.loc[['Avg Gen Time (s)']].T

    fig_time, ax_time = plt.subplots(figsize=(7, 6))
    bars = df_times.plot(kind='bar', ax=ax_time, rot=0, legend=False, width=0.6)

    ax_time.set_title('Average Generation Time per Sample', fontsize=16)
    ax_time.set_ylabel('Time (seconds)', fontsize=12)
    ax_time.set_xlabel('Model', fontsize=12)
    ax_time.grid(axis='y', linestyle='--', alpha=0.7)
    ax_time.tick_params(axis='both', which='major', labelsize=11)


    for container in ax_time.containers:
        ax_time.bar_label(container, fmt='%.4f', label_type='edge', fontsize=9, padding=3)


    if avg_t5_time / avg_textrank_time > 20:
         ax_time.set_yscale('log')
         ax_time.set_ylabel('Time (seconds, log scale)', fontsize=12)
         ax_time.set_title('Average Generation Time per Sample (Log Scale)', fontsize=16)


    plt.tight_layout()
    if SAVE_PLOTS:
        plot_path = os.path.join(PLOT_OUTPUT_DIR, "generation_time_comparison.png")
        plt.savefig(plot_path)
        print(f"Generation time plot saved to: {plot_path}")
    plt.show()


    print("\n--- Interpretation ---")
    if 'T5' in df.columns and 'TextRank' in df.columns:
        t5_r1 = df.loc['ROUGE-1', 'T5']
        tr_r1 = df.loc['ROUGE-1', 'TextRank']
        t5_rl = df.loc['ROUGE-L', 'T5']
        tr_rl = df.loc['ROUGE-L', 'TextRank']

        if t5_r1 > tr_r1 and t5_rl > tr_rl:
            print(f"- The T5 model ({t5_model_name}) significantly outperforms TextRank across all ROUGE metrics.")
            print(f"  This indicates better overlap with reference summaries in terms of words (ROUGE-1: {t5_r1:.2f} vs {tr_r1:.2f})")
            print(f"  and sentence structure/longest common subsequences (ROUGE-L: {t5_rl:.2f} vs {tr_rl:.2f}).")
            print("- This is expected, as fine-tuned abstractive models can paraphrase and generate novel sentences")
            print("  that better match human references, while TextRank is limited to extracting existing sentences.")
        else:
            print("- Comparison suggests mixed results or potential issues in T5 performance/evaluation.")

        if avg_t5_time > avg_textrank_time:
            speed_diff = avg_t5_time / avg_textrank_time if avg_textrank_time > 0 else float('inf')
            print(f"\n- There is a clear speed vs. quality trade-off:")
            print(f"  - TextRank is extremely fast (Avg: {avg_textrank_time:.4f}s).")
            print(f"  - T5 inference is considerably slower (Avg: {avg_t5_time:.4f}s), approx. {speed_diff:.1f} times slower than TextRank.")
            print("- The choice between models depends on the application's priority: speed/low-resource (TextRank) or summary quality/fluency (T5).")
        else:
            print("\n- Generation time comparison shows unexpected results.")

    print("\n- Further qualitative analysis is recommended to assess fluency, coherence, and factuality beyond ROUGE scores.")
    print("--- End of Analysis ---")


if __name__ == "__main__":
    print("Starting Results Analysis...")
    results_data = load_results(RESULTS_FILE_PATH)
    analyze_and_visualize(results_data)
    print("Analysis finished.")