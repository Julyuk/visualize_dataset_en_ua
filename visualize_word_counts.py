#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
from typing import Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import json

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional import guard for environments without pandas
    pd = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and visualize sentence word-count distributions for English and Ukrainian. "
            "You can load data from a Hugging Face dataset or a local CSV."
        )
    )

    # Data source: default to HF dataset if nothing is passed
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="turuta/Multi30k-uk",
        help="Hugging Face dataset repo id. Default: 'turuta/Multi30k-uk'",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a local CSV file containing English and Ukrainian columns. If provided, overrides --hf-dataset",
    )

    # HF dataset options
    parser.add_argument(
        "--config",
        type=str,
        default="flickr_2016",
        help="Dataset configuration/name (e.g., 'flickr_2016', 'flickr_2017', 'flickr_2018'). Default: flickr_2016",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (default: train)",
    )

    # Column names (for CSV or when auto-detection fails for HF datasets)
    parser.add_argument(
        "--en-col",
        type=str,
        default=None,
        help="Name of the English text column (default: auto-detect)",
    )
    parser.add_argument(
        "--uk-col",
        type=str,
        default=None,
        help="Name of the Ukrainian text column (default: auto-detect)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally limit the number of samples used for faster runs",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save figures (default: outputs)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Figure DPI (default: 140)",
    )

    # Reporting and plotting options
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate metrics report files (JSON/CSV and top-tokens CSVs)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Top-N frequent tokens to include in the report (default: 30)",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=3,
        help="Threshold (<= value) to count as very short sentences (default: 3 words)",
    )
    parser.add_argument(
        "--no-diagrams",
        action="store_true",
        help="Disable histogram images (diagrams)",
    )

    return parser.parse_args()


# Treat apostrophes and hyphens as part of a word; support Unicode letters
TOKEN_PATTERN = re.compile(r"\b[\w’'-]+\b", flags=re.UNICODE)


def count_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    tokens = TOKEN_PATTERN.findall(text)
    return len(tokens)


def compute_counts(texts: Iterable[str]) -> List[int]:
    return [count_words(t) for t in texts]


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


def compute_language_report(
    texts: List[str],
    counts: List[int],
    top_n: int,
    short_threshold: int,
) -> dict:
    import collections

    num_sentences = len(texts)
    arr = np.array(counts, dtype=np.int64) if num_sentences > 0 else np.array([], dtype=np.int64)

    # Sentence length stats
    if arr.size > 0:
        mean_len = float(arr.mean())
        median_len = float(np.median(arr))
        std_len = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        min_len = int(arr.min())
        max_len = int(arr.max())
        empty_frac = float((arr == 0).sum() / arr.size)
        short_frac = float((arr <= short_threshold).sum() / arr.size)
    else:
        mean_len = median_len = std_len = 0.0
        min_len = max_len = 0
        empty_frac = short_frac = 0.0

    # Character-level stats
    if num_sentences > 0:
        chars_with = np.array([len(t) if isinstance(t, str) else 0 for t in texts], dtype=np.int64)
        chars_no = np.array([len((t or "").replace(" ", "")) if isinstance(t, str) else 0 for t in texts], dtype=np.int64)
        avg_chars_with = float(chars_with.mean())
        avg_chars_no = float(chars_no.mean())
    else:
        avg_chars_with = avg_chars_no = 0.0

    # Tokens and vocabulary
    all_tokens: List[str] = []
    total_tokens = 0
    for t in texts:
        toks = tokenize(t)
        total_tokens += len(toks)
        if toks:
            all_tokens.extend(toks)
    counter = collections.Counter(all_tokens)
    vocab_size = int(len(counter))
    ttr = float(vocab_size / total_tokens) if total_tokens > 0 else 0.0
    top = counter.most_common(int(top_n)) if top_n and top_n > 0 else []

    return {
        "num_sentences": int(num_sentences),
        "mean_len": mean_len,
        "median_len": median_len,
        "std_len": std_len,
        "min_len": min_len,
        "max_len": max_len,
        "empty_fraction": empty_frac,
        "short_fraction": short_frac,
        "avg_chars_with_spaces": avg_chars_with,
        "avg_chars_no_spaces": avg_chars_no,
        "vocab_size": vocab_size,
        "ttr": ttr,
        "top_tokens": top,
        "total_tokens": int(total_tokens),
    }


def write_reports(
    en_texts: List[str],
    uk_texts: List[str],
    en_counts: List[int],
    uk_counts: List[int],
    output_dir: str,
    top_n: int,
    short_threshold: int,
) -> None:
    en_rep = compute_language_report(en_texts, en_counts, top_n, short_threshold)
    uk_rep = compute_language_report(uk_texts, uk_counts, top_n, short_threshold)

    # Correlation of lengths
    pair_n = min(len(en_counts), len(uk_counts))
    if pair_n >= 2 and np.std(en_counts[:pair_n]) > 0 and np.std(uk_counts[:pair_n]) > 0:
        corr = float(np.corrcoef(np.array(en_counts[:pair_n]), np.array(uk_counts[:pair_n]))[0, 1])
    else:
        corr = None

    report = {"english": en_rep, "ukrainian": uk_rep, "pearson_r_en_uk": corr}

    # JSON
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved: {json_path}")

    # CSV (wide form for scalar metrics)
    csv_path = os.path.join(output_dir, "metrics.csv")
    lines = [
        "metric,english,ukrainian",
        f"num_sentences,{en_rep['num_sentences']},{uk_rep['num_sentences']}",
        f"mean_len,{en_rep['mean_len']:.3f},{uk_rep['mean_len']:.3f}",
        f"median_len,{en_rep['median_len']:.3f},{uk_rep['median_len']:.3f}",
        f"std_len,{en_rep['std_len']:.3f},{uk_rep['std_len']:.3f}",
        f"min_len,{en_rep['min_len']},{uk_rep['min_len']}",
        f"max_len,{en_rep['max_len']},{uk_rep['max_len']}",
        f"empty_fraction,{en_rep['empty_fraction']:.4f},{uk_rep['empty_fraction']:.4f}",
        f"short_fraction,{en_rep['short_fraction']:.4f},{uk_rep['short_fraction']:.4f}",
        f"avg_chars_with_spaces,{en_rep['avg_chars_with_spaces']:.3f},{uk_rep['avg_chars_with_spaces']:.3f}",
        f"avg_chars_no_spaces,{en_rep['avg_chars_no_spaces']:.3f},{uk_rep['avg_chars_no_spaces']:.3f}",
        f"total_tokens,{en_rep['total_tokens']},{uk_rep['total_tokens']}",
        f"vocab_size,{en_rep['vocab_size']},{uk_rep['vocab_size']}",
        f"ttr,{en_rep['ttr']:.6f},{uk_rep['ttr']:.6f}",
        f"pearson_r_en_uk,{corr if corr is not None else ''},",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {csv_path}")

    # Top tokens to separate CSVs
    top_en_path = os.path.join(output_dir, "top_tokens_en.csv")
    with open(top_en_path, "w", encoding="utf-8") as f:
        f.write("token,count\n")
        for tok, cnt in en_rep["top_tokens"]:
            f.write(f"{tok},{cnt}\n")
    print(f"Saved: {top_en_path}")

    top_uk_path = os.path.join(output_dir, "top_tokens_uk.csv")
    with open(top_uk_path, "w", encoding="utf-8") as f:
        f.write("token,count\n")
        for tok, cnt in uk_rep["top_tokens"]:
            f.write(f"{tok},{cnt}\n")
    print(f"Saved: {top_uk_path}")


def plot_histogram(
    counts: List[int], title: str, save_path: str, dpi: int = 140
) -> None:
    if not counts:
        print(f"[WARN] No counts to plot for {title} -> {save_path}")
        return

    max_len = max(counts)
    # Use 1-wide bins up to the max length
    bins = list(range(0, max_len + 2))

    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel("Words per sentence")
    plt.ylabel("Number of sentences")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Saved: {save_path}")


def plot_side_by_side(
    en_counts: List[int], uk_counts: List[int], save_path: str, dpi: int = 140
) -> None:
    if not en_counts and not uk_counts:
        print(f"[WARN] No counts to plot -> {save_path}")
        return

    en_max = max(en_counts) if en_counts else 0
    uk_max = max(uk_counts) if uk_counts else 0
    max_len = max(en_max, uk_max)
    bins = list(range(0, max_len + 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].hist(en_counts, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    axes[0].set_title("English: words per sentence")
    axes[0].set_xlabel("Words")
    axes[0].set_ylabel("Sentences")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].hist(uk_counts, bins=bins, color="#F58518", edgecolor="black", alpha=0.85)
    axes[1].set_title("Ukrainian: words per sentence")
    axes[1].set_xlabel("Words")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Sentence length distributions")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")



def plot_overlay_hist(
    en_counts: List[int], uk_counts: List[int], save_path: str, dpi: int = 140
) -> None:
    if not en_counts and not uk_counts:
        print(f"[WARN] No counts to plot -> {save_path}")
        return
    max_len = max(max(en_counts or [0]), max(uk_counts or [0]))
    bins = list(range(0, max_len + 2))
    plt.figure(figsize=(8, 5))
    plt.hist(en_counts, bins=bins, color="#4C78A8", alpha=0.55, label="English", density=True)
    plt.hist(uk_counts, bins=bins, color="#F58518", alpha=0.55, label="Ukrainian", density=True)
    plt.title("Overlayed histograms (density)")
    plt.xlabel("Words per sentence")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Saved: {save_path}")


def plot_ecdf(en_counts: List[int], uk_counts: List[int], save_path: str, dpi: int = 140) -> None:
    def ecdf(values: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        if len(values) == 0:
            return np.array([]), np.array([])
        x = np.sort(np.array(values))
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    x_en, y_en = ecdf(en_counts)
    x_uk, y_uk = ecdf(uk_counts)

    plt.figure(figsize=(8, 5))
    if x_en.size:
        plt.step(x_en, y_en, where="post", color="#4C78A8", label="English")
    if x_uk.size:
        plt.step(x_uk, y_uk, where="post", color="#F58518", label="Ukrainian")
    plt.title("ECDF of words per sentence")
    plt.xlabel("Words per sentence")
    plt.ylabel("Cumulative fraction")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Saved: {save_path}")


def plot_box_violin(en_counts: List[int], uk_counts: List[int], save_path: str, dpi: int = 140) -> None:
    if not en_counts and not uk_counts:
        print(f"[WARN] No counts to plot -> {save_path}")
        return
    data = [en_counts, uk_counts]
    labels = ["English", "Ukrainian"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].boxplot(data, tick_labels=labels, showmeans=True)
    axes[0].set_title("Box plot")
    axes[0].set_ylabel("Words per sentence")
    axes[0].grid(True, linestyle=":", alpha=0.4)
    axes[1].violinplot(data, showmeans=True, showmedians=True)
    axes[1].set_title("Violin plot")
    axes[1].set_xticks([1, 2], labels)
    axes[1].grid(True, linestyle=":", alpha=0.4)
    fig.suptitle("Sentence length summary")
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_scatter_pairs(
    en_counts: List[int], uk_counts: List[int], save_path: str, dpi: int = 140
) -> None:
    n = min(len(en_counts), len(uk_counts))
    if n == 0:
        print(f"[WARN] No counts to plot -> {save_path}")
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(en_counts[:n], uk_counts[:n], s=10, alpha=0.5, color="#6F4E7C")
    plt.title("EN vs UK sentence lengths")
    plt.xlabel("English words")
    plt.ylabel("Ukrainian words")
    lim = max(max(en_counts[:n]), max(uk_counts[:n])) + 1
    plt.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Saved: {save_path}")


# ----------------------------
# Lightweight OOP orchestration
# ----------------------------

class DatasetLoader:
    """Simple loader that abstracts HF vs CSV sources.

    Keeps logic minimal by delegating to the lightweight helper functions
    `load_from_hf` and `load_from_csv` defined below.
    """

    def __init__(
        self,
        hf_dataset: str,
        config: Optional[str],
        split: str,
        csv_path: Optional[str],
        en_col: Optional[str],
        uk_col: Optional[str],
    ) -> None:
        self.hf_dataset = hf_dataset
        self.config = config
        self.split = split
        self.csv_path = csv_path
        self.en_col = en_col
        self.uk_col = uk_col

    def load(self) -> Tuple[List[str], List[str]]:
        if self.csv_path:
            return load_from_csv(self.csv_path, self.en_col, self.uk_col)
        return load_from_hf(self.hf_dataset, self.config, self.split, self.en_col, self.uk_col)


class Plotter:
    """Groups plotting actions for clarity and reusability."""

    def __init__(self, output_dir: str, dpi: int) -> None:
        self.output_dir = output_dir
        self.dpi = dpi

    def render_basic(self, en_counts: List[int], uk_counts: List[int]) -> None:
        en_path = os.path.join(self.output_dir, "en_word_count_hist.png")
        uk_path = os.path.join(self.output_dir, "uk_word_count_hist.png")
        both_path = os.path.join(self.output_dir, "word_count_hist_side_by_side.png")
        plot_histogram(en_counts, "English: words per sentence", en_path, dpi=self.dpi)
        plot_histogram(uk_counts, "Ukrainian: words per sentence", uk_path, dpi=self.dpi)
        plot_side_by_side(en_counts, uk_counts, both_path, dpi=self.dpi)

    def render_extended(self, en_counts: List[int], uk_counts: List[int]) -> None:
        plot_overlay_hist(en_counts, uk_counts, os.path.join(self.output_dir, "word_count_hist_overlay.png"), dpi=self.dpi)
        plot_ecdf(en_counts, uk_counts, os.path.join(self.output_dir, "word_count_ecdf.png"), dpi=self.dpi)
        plot_box_violin(en_counts, uk_counts, os.path.join(self.output_dir, "word_count_box_violin.png"), dpi=self.dpi)
        plot_scatter_pairs(en_counts, uk_counts, os.path.join(self.output_dir, "en_vs_uk_scatter.png"), dpi=self.dpi)

    def render_all(self, en_counts: List[int], uk_counts: List[int], include_extended: bool = True) -> None:
        self.render_basic(en_counts, uk_counts)
        if include_extended:
            self.render_extended(en_counts, uk_counts)


class ReportWriter:
    """Computes and writes metrics related outputs."""

    def __init__(self, top_n: int, short_threshold: int) -> None:
        self.top_n = top_n
        self.short_threshold = short_threshold

    def write(self, en_texts: List[str], uk_texts: List[str], en_counts: List[int], uk_counts: List[int], output_dir: str) -> None:
        write_reports(
            en_texts=en_texts,
            uk_texts=uk_texts,
            en_counts=en_counts,
            uk_counts=uk_counts,
            output_dir=output_dir,
            top_n=self.top_n,
            short_threshold=self.short_threshold,
        )


def load_from_hf(
    repo_id: str, config: Optional[str], split: str, en_col: Optional[str], uk_col: Optional[str]
) -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        print(
            "[ERROR] The 'datasets' library is required for --hf-dataset. Install it via 'pip install datasets'.",
            file=sys.stderr,
        )
        raise e

    ds = load_dataset(repo_id, config, split=split)

    # Case 1: translation dict/list-style column
    col_names = list(ds.column_names)
    lowered = [c.lower() for c in col_names]

    if "translation" in lowered:
        tcol = col_names[lowered.index("translation")]

        def safe_get(example: dict, lang_key: str) -> str:
            tr = example.get(tcol)
            if tr is None:
                return ""
            if isinstance(tr, dict):
                return str(tr.get(lang_key, ""))
            if isinstance(tr, list):
                # Try to map positions: assume order contains en then uk
                # This is a best-effort fallback
                if lang_key.lower().startswith("en") and len(tr) >= 1:
                    return str(tr[0])
                if lang_key.lower().startswith("uk") and len(tr) >= 2:
                    return str(tr[1])
            return ""

        en_texts = [safe_get(ex, "en") for ex in ds]
        uk_texts = [safe_get(ex, "uk") for ex in ds]
        return en_texts, uk_texts

    # Case 2: two top-level columns. Allow explicit or auto-detect
    if en_col and uk_col:
        return [str(x) for x in ds[en_col]], [str(x) for x in ds[uk_col]]

    # Auto-detect columns by name hints
    def find_col(hints: List[str]) -> Optional[str]:
        for c in col_names:
            lc = c.lower()
            if any(h in lc for h in hints):
                return c
        return None

    en_guess = en_col or find_col(["en", "eng", "english"])  # type: ignore
    uk_guess = uk_col or find_col(["uk", "ukr", "ukrain"])  # type: ignore

    if en_guess and uk_guess:
        return [str(x) for x in ds[en_guess]], [str(x) for x in ds[uk_guess]]

    raise ValueError(
        f"Could not infer text columns. Available columns: {col_names}. "
        f"Please specify --en-col and --uk-col."
    )


def load_from_csv(csv_path: str, en_col: Optional[str], uk_col: Optional[str]) -> Tuple[List[str], List[str]]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for --csv usage. Install via 'pip install pandas'.")

    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    if en_col is None or uk_col is None:
        def find_col(hints: List[str]) -> Optional[str]:
            for c in cols:
                lc = str(c).lower()
                if any(h in lc for h in hints):
                    return c
            return None

        en_col = en_col or find_col(["en", "eng", "english"])  # type: ignore
        uk_col = uk_col or find_col(["uk", "ukr", "ukrain"])  # type: ignore

    if en_col is None or uk_col is None:
        raise ValueError(
            f"Could not infer column names from CSV. Columns present: {cols}. "
            f"Please pass --en-col and --uk-col."
        )

    return df[en_col].astype(str).tolist(), df[uk_col].astype(str).tolist()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    loader = DatasetLoader(
        hf_dataset=args.hf_dataset,
        config=args.config,
        split=args.split,
        csv_path=args.csv,
        en_col=args.en_col,
        uk_col=args.uk_col,
    )
    en_texts, uk_texts = loader.load()

    if args.max_samples is not None:
        en_texts = en_texts[: args.max_samples]
        uk_texts = uk_texts[: args.max_samples]

    en_counts = compute_counts(en_texts)
    uk_counts = compute_counts(uk_texts)

    # Diagrams (unless disabled). Always render extended diagrams by default.
    if not args.no_diagrams:
        Plotter(output_dir=args.output_dir, dpi=args.dpi).render_all(
            en_counts=en_counts,
            uk_counts=uk_counts,
            include_extended=True,
        )

    # Reports
    if args.report:
        ReportWriter(top_n=args.top_n, short_threshold=args.short_threshold).write(
            en_texts=en_texts,
            uk_texts=uk_texts,
            en_counts=en_counts,
            uk_counts=uk_counts,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()


