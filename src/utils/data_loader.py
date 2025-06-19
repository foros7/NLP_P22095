"""
Data Loading and Management Utilities
Handles loading, saving and preprocessing of text data
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loading and management utility class
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"

        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)
        (self.data_dir / "reconstructed").mkdir(exist_ok=True)

    def load_original_texts(self) -> List[str]:
        """
        Load the original texts from the assignment

        Returns:
            List of original text strings
        """
        try:
            texts_file = self.data_dir / "original_texts.txt"

            if not texts_file.exists():
                logger.error(f"Original texts file not found: {texts_file}")
                return []

            with open(texts_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the texts
            texts = []
            lines = content.split('\n')
            current_text = ""

            for line in lines:
                line = line.strip()
                if line.startswith('TEXT ') and ':' in line:
                    if current_text:
                        # Clean and add previous text
                        clean_text = current_text.strip().strip('"')
                        if clean_text:
                            texts.append(clean_text)
                    current_text = ""
                elif line and not line.startswith('TEXT '):
                    current_text += line + " "

            # Add the last text
            if current_text:
                clean_text = current_text.strip().strip('"')
                if clean_text:
                    texts.append(clean_text)

            logger.info(f"✅ Loaded {len(texts)} original texts")
            return texts

        except Exception as e:
            logger.error(f"❌ Error loading original texts: {str(e)}")
            return []

    def load_greek_legal_texts(self) -> Dict[str, Any]:
        """
        Load Greek legal texts for the bonus task

        Returns:
            Dictionary containing texts and ground truth
        """
        try:
            greek_file = self.data_dir / "greek_legal_texts.txt"

            if not greek_file.exists():
                logger.error(f"Greek legal texts file not found: {greek_file}")
                return {}

            with open(greek_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the content
            lines = content.split('\n')
            texts = []
            ground_truth = {}

            current_article = None
            current_text = ""
            in_ground_truth = False

            for line in lines:
                line = line.strip()

                if line.startswith('Άρθρο '):
                    if current_text and current_article:
                        texts.append({
                            'article': current_article,
                            'text': current_text.strip()
                        })

                    # Extract article number
                    parts = line.split('.')
                    if len(parts) > 0:
                        article_part = parts[0].replace('Άρθρο ', '').strip()
                        current_article = f"Article_{article_part}"
                    current_text = line

                elif line.startswith('GROUND TRUTH'):
                    in_ground_truth = True
                    if current_text and current_article:
                        texts.append({
                            'article': current_article,
                            'text': current_text.strip()
                        })
                        current_text = ""

                elif in_ground_truth and ':' in line:
                    # Parse ground truth
                    parts = line.split(':')
                    if len(parts) == 2:
                        article_key = parts[0].strip().replace(
                            'Άρθρο ', 'Article_')
                        words = parts[1].strip()
                        # Extract words from brackets
                        import re
                        matches = re.findall(r'\[([^\]]+)\]', words)
                        ground_truth[article_key] = matches

                elif not in_ground_truth and current_text:
                    current_text += " " + line

            # Add last text if exists
            if current_text and current_article:
                texts.append({
                    'article': current_article,
                    'text': current_text.strip()
                })

            result = {
                'texts': texts,
                'ground_truth': ground_truth
            }

            logger.info(
                f"✅ Loaded {len(texts)} Greek legal texts with ground truth")
            return result

        except Exception as e:
            logger.error(f"❌ Error loading Greek legal texts: {str(e)}")
            return {}

    def save_reconstruction_results(self, custom_results: Dict,
                                    library_results: Dict,
                                    comparison_results: Dict):
        """
        Save text reconstruction results

        Args:
            custom_results: Results from custom pipeline
            library_results: Results from library pipelines
            comparison_results: Comparison analysis results
        """
        try:
            results_file = self.data_dir / "reconstructed" / "reconstruction_results.json"

            # Convert dataclass objects to dictionaries for JSON serialization
            def convert_dataclass_to_dict(obj):
                if hasattr(obj, '__dict__'):
                    return {k: convert_dataclass_to_dict(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, dict):
                    return {k: convert_dataclass_to_dict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_dataclass_to_dict(item) for item in obj]
                else:
                    return obj

            # Convert results to JSON-serializable format
            serializable_custom = convert_dataclass_to_dict(custom_results)
            serializable_library = convert_dataclass_to_dict(library_results)
            serializable_comparison = convert_dataclass_to_dict(
                comparison_results)

            all_results = {
                'custom_pipeline': serializable_custom,
                'library_pipelines': serializable_library,
                'comparison': serializable_comparison,
                'metadata': {
                    'timestamp': str(pd.Timestamp.now()),
                    'version': '1.0'
                }
            }

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Reconstruction results saved to {results_file}")

        except Exception as e:
            logger.error(f"❌ Error saving reconstruction results: {str(e)}")

    def save_embeddings(self, embeddings: Dict, filename: str):
        """
        Save word embeddings using pickle

        Args:
            embeddings: Dictionary of embeddings
            filename: Name of file to save
        """
        try:
            embeddings_dir = self.data_dir / "embeddings"
            embeddings_dir.mkdir(exist_ok=True)

            filepath = embeddings_dir / f"{filename}.pkl"

            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)

            logger.info(f"✅ Embeddings saved to {filepath}")

        except Exception as e:
            logger.error(f"❌ Error saving embeddings: {str(e)}")

    def load_embeddings(self, filename: str) -> Dict:
        """
        Load word embeddings from pickle file

        Args:
            filename: Name of file to load

        Returns:
            Dictionary of embeddings
        """
        try:
            embeddings_dir = self.data_dir / "embeddings"
            filepath = embeddings_dir / f"{filename}.pkl"

            if not filepath.exists():
                logger.warning(f"Embeddings file not found: {filepath}")
                return {}

            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)

            logger.info(f"✅ Embeddings loaded from {filepath}")
            return embeddings

        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {str(e)}")
            return {}

    def save_similarity_results(self, results: Dict, filename: str):
        """
        Save similarity analysis results

        Args:
            results: Similarity results dictionary
            filename: Name of file to save
        """
        try:
            results_file = self.results_dir / "reports" / f"{filename}.json"

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Similarity results saved to {results_file}")

        except Exception as e:
            logger.error(f"❌ Error saving similarity results: {str(e)}")

    def create_results_dataframe(self, results: Dict) -> pd.DataFrame:
        """
        Convert results dictionary to pandas DataFrame for analysis

        Args:
            results: Results dictionary

        Returns:
            Pandas DataFrame
        """
        try:
            # Flatten the results dictionary for DataFrame creation
            rows = []

            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, dict):
                                items.extend(flatten_dict(
                                    item, f"{new_key}_{i}", sep=sep).items())
                            else:
                                items.append((f"{new_key}_{i}", item))
                    else:
                        items.append((new_key, v))
                return dict(items)

            flattened = flatten_dict(results)
            df = pd.DataFrame([flattened])

            logger.info(f"✅ Created DataFrame with {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"❌ Error creating results DataFrame: {str(e)}")
            return pd.DataFrame()

    def get_file_paths(self) -> Dict[str, Path]:
        """
        Get all important file paths for the project

        Returns:
            Dictionary of file paths
        """
        return {
            'project_root': self.project_root,
            'data_dir': self.data_dir,
            'results_dir': self.results_dir,
            'figures_dir': self.results_dir / "figures",
            'reports_dir': self.results_dir / "reports",
            'reconstructed_dir': self.data_dir / "reconstructed",
            'original_texts': self.data_dir / "original_texts.txt",
            'greek_legal_texts': self.data_dir / "greek_legal_texts.txt"
        }
