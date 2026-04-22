"""
Phase 6: Text Analysis Module

TF-IDF, topic modeling, and keyword extraction for bug summaries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans


class TextAnalyzer:
    """Text analysis for bug summaries and alert notes."""

    def __init__(
        self,
        max_features: int = 1000,
        min_df: int = 2,
        max_df: float = 0.95
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.tfidf_vectorizer_ = None
        self.tfidf_matrix_ = None
        self.feature_names_ = None

    def fit_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF matrix
        """
        # Clean texts
        texts = [self._preprocess_text(t) for t in texts]

        self.tfidf_vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            ngram_range=(1, 2)
        )

        self.tfidf_matrix_ = self.tfidf_vectorizer_.fit_transform(texts)
        self.feature_names_ = self.tfidf_vectorizer_.get_feature_names_out()

        print(f"TF-IDF matrix shape: {self.tfidf_matrix_.shape}")

        return self.tfidf_matrix_.toarray()

    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using fitted vectorizer."""
        if self.tfidf_vectorizer_ is None:
            raise ValueError("Must call fit_tfidf first")
        texts = [self._preprocess_text(t) for t in texts]
        return self.tfidf_vectorizer_.transform(texts).toarray()

    def get_top_terms(
        self,
        n_terms: int = 20,
        doc_idx: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF terms.

        Args:
            n_terms: Number of top terms
            doc_idx: Specific document index (None for corpus-wide)

        Returns:
            List of (term, score) tuples
        """
        if self.tfidf_matrix_ is None:
            return []

        if doc_idx is not None:
            scores = self.tfidf_matrix_[doc_idx].toarray().flatten()
        else:
            scores = np.asarray(self.tfidf_matrix_.mean(axis=0)).flatten()

        top_indices = np.argsort(scores)[::-1][:n_terms]

        return [(self.feature_names_[i], scores[i]) for i in top_indices]

    def cluster_texts(
        self,
        n_clusters: int = 10,
        method: str = 'kmeans'
    ) -> np.ndarray:
        """
        Cluster texts based on TF-IDF features.

        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans')

        Returns:
            Cluster labels
        """
        if self.tfidf_matrix_ is None:
            raise ValueError("Must call fit_tfidf first")

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(self.tfidf_matrix_)
            return labels
        else:
            raise ValueError(f"Unknown method: {method}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special chars but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text


def extract_keywords(
    texts: List[str],
    n_keywords: int = 10,
    method: str = 'tfidf'
) -> List[str]:
    """
    Extract keywords from texts.

    Args:
        texts: List of text documents
        n_keywords: Number of keywords to extract
        method: 'tfidf' or 'frequency'

    Returns:
        List of top keywords
    """
    # Clean texts
    clean_texts = []
    for t in texts:
        if pd.isna(t):
            continue
        t = str(t).lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        clean_texts.append(t)

    if not clean_texts:
        return []

    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_texts)
            scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()
            top_indices = np.argsort(scores)[::-1][:n_keywords]
            return [feature_names[i] for i in top_indices]
        except:
            return []

    elif method == 'frequency':
        all_words = ' '.join(clean_texts).split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'this', 'that'}
        words = [w for w in all_words if w not in stop_words and len(w) > 2]
        word_counts = Counter(words)
        return [w for w, _ in word_counts.most_common(n_keywords)]

    return []


def extract_topics_lda(
    texts: List[str],
    n_topics: int = 10,
    n_top_words: int = 10
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Extract topics using LDA.

    Args:
        texts: List of text documents
        n_topics: Number of topics
        n_top_words: Number of top words per topic

    Returns:
        Tuple of (document_topic_matrix, topic_words_list)
    """
    # Clean texts
    clean_texts = []
    for t in texts:
        if pd.isna(t):
            clean_texts.append("")
            continue
        t = str(t).lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        clean_texts.append(t)

    # Vectorize
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )

    try:
        doc_term_matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        # Not enough documents
        return np.zeros((len(texts), n_topics)), [[] for _ in range(n_topics)]

    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20
    )

    doc_topic_matrix = lda.fit_transform(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()

    # Get top words per topic
    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        topic_words.append(top_words)

    return doc_topic_matrix, topic_words


def extract_topics_nmf(
    texts: List[str],
    n_topics: int = 10,
    n_top_words: int = 10
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Extract topics using NMF (Non-negative Matrix Factorization).

    Args:
        texts: List of text documents
        n_topics: Number of topics
        n_top_words: Number of top words per topic

    Returns:
        Tuple of (document_topic_matrix, topic_words_list)
    """
    # Clean and vectorize
    clean_texts = []
    for t in texts:
        if pd.isna(t):
            clean_texts.append("")
            continue
        t = str(t).lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        clean_texts.append(t)

    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return np.zeros((len(texts), n_topics)), [[] for _ in range(n_topics)]

    # Fit NMF
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
    doc_topic_matrix = nmf.fit_transform(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()

    # Get top words
    topic_words = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        topic_words.append(top_words)

    return doc_topic_matrix, topic_words


def categorize_by_keywords(
    texts: List[str],
    categories: Dict[str, List[str]]
) -> List[str]:
    """
    Categorize texts by keyword matching.

    Args:
        texts: List of text documents
        categories: Dict mapping category name to keywords

    Returns:
        List of category labels
    """
    result = []

    for text in texts:
        if pd.isna(text):
            result.append('unknown')
            continue

        text = str(text).lower()
        best_cat = 'other'
        best_score = 0

        for cat_name, keywords in categories.items():
            score = sum(1 for kw in keywords if kw.lower() in text)
            if score > best_score:
                best_score = score
                best_cat = cat_name

        result.append(best_cat)

    return result


# Predefined Mozilla-related categories
MOZILLA_CATEGORIES = {
    'memory': ['memory', 'leak', 'oom', 'allocation', 'heap', 'gc', 'garbage'],
    'graphics': ['graphics', 'render', 'webrender', 'gpu', 'canvas', 'layout', 'paint'],
    'network': ['network', 'http', 'fetch', 'socket', 'request', 'response', 'cache'],
    'javascript': ['javascript', 'js', 'wasm', 'jit', 'script', 'async', 'promise'],
    'dom': ['dom', 'element', 'node', 'event', 'document', 'html', 'css'],
    'media': ['media', 'video', 'audio', 'codec', 'webrtc', 'streaming'],
    'security': ['security', 'sandbox', 'permission', 'csp', 'certificate', 'ssl'],
    'startup': ['startup', 'init', 'boot', 'launch', 'cold', 'warm'],
    'io': ['io', 'disk', 'file', 'storage', 'indexeddb', 'localstorage']
}
