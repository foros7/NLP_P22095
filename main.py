#!/usr/bin/env python3
"""
Main entry point for NLP Assignment 2025
Text Reconstruction and Semantic Analysis

This script runs all the deliverables:
1. Text Reconstruction (Custom + Library pipelines)
2. Semantic Analysis (Word embeddings, similarity, visualizations)
3. Report Generation
4. Bonus: Masked Language Modeling for Greek legal texts
"""

from src.utils.evaluation import EvaluationMetrics
from src.masked_language.greek_legal_text import GreekLegalTextProcessor
from src.semantic_analysis.visualization import EmbeddingVisualizer
from src.semantic_analysis.similarity import SimilarityAnalyzer
from src.semantic_analysis.embeddings import WordEmbeddingsAnalyzer
from src.text_reconstruction.library_pipelines import LibraryPipelines
from src.text_reconstruction.custom_pipeline import CustomTextReconstructor
from src.utils.data_loader import DataLoader
import os
import sys
import logging
from pathlib import Path
import argparse

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nlp_assignment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_deliverable_1():
    """
    Παραδοτέο 1: Ανακατασκευή Κειμένου
    A. Custom pipeline για 2 προτάσεις  
    B. 3 διαφορετικά Python pipelines για ολόκληρα κείμενα
    C. Σύγκριση αποτελεσμάτων
    """
    logger.info("🚀 Starting Deliverable 1: Text Reconstruction")

    # Load original texts
    data_loader = DataLoader()
    texts = data_loader.load_original_texts()

    # A. Custom pipeline για επιλεγμένες προτάσεις
    logger.info("📝 Running custom pipeline for selected sentences...")
    custom_reconstructor = CustomTextReconstructor()

    # Επιλογή 2 προτάσεων με προβλήματα
    selected_sentences = [
        "Thank your message to show our words to the doctor, as his next contract checking, to all of us.",
        "During our ﬁnal discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?"
    ]

    custom_results = custom_reconstructor.reconstruct_sentences(
        selected_sentences)

    # B. Library pipelines για ολόκληρα κείμενα
    logger.info("🔧 Running library pipelines for complete texts...")
    library_pipelines = LibraryPipelines()
    library_results = library_pipelines.reconstruct_all_texts(texts)

    # C. Σύγκριση αποτελεσμάτων
    logger.info("📊 Comparing reconstruction results...")
    evaluator = EvaluationMetrics()
    comparison_results = evaluator.compare_reconstruction_methods(
        custom_results, library_results
    )

    # Save results
    data_loader.save_reconstruction_results(
        custom_results, library_results, comparison_results)

    logger.info("✅ Deliverable 1 completed!")
    return custom_results, library_results, comparison_results


def run_deliverable_2(custom_results, library_results):
    """
    Παραδοτέο 2: Υπολογιστική Ανάλυση
    - Word embeddings (Word2Vec, GloVe, FastText, BERT)
    - Cosine similarity calculations  
    - PCA/t-SNE visualizations
    """
    logger.info("🚀 Starting Deliverable 2: Computational Analysis")

    # Generate word embeddings
    logger.info("🧠 Generating word embeddings...")
    embeddings_analyzer = WordEmbeddingsAnalyzer()

    # Train all embedding models
    embedding_results = embeddings_analyzer.train_all_models(
        custom_results, library_results
    )

    # Calculate semantic similarities
    logger.info("🔍 Calculating semantic similarities...")
    similarity_analyzer = SimilarityAnalyzer()
    similarity_results = similarity_analyzer.calculate_all_similarities(
        embedding_results
    )

    # Create visualizations
    logger.info("📈 Creating visualizations...")
    visualizer = EmbeddingVisualizer()

    # PCA visualizations
    pca_plots = visualizer.create_pca_visualizations(embedding_results)

    # t-SNE visualizations
    tsne_plots = visualizer.create_tsne_visualizations(embedding_results)

    # Similarity heatmaps
    similarity_plots = visualizer.create_similarity_heatmaps(
        similarity_results)

    logger.info("✅ Deliverable 2 completed!")
    return embedding_results, similarity_results, {
        'pca': pca_plots,
        'tsne': tsne_plots,
        'similarity': similarity_plots
    }


def run_bonus_deliverable():
    """
    Bonus: Masked Clause Input για ελληνικά νομικά κείμενα
    """
    logger.info(
        "🚀 Starting Bonus: Masked Language Modeling for Greek Legal Texts")

    processor = GreekLegalTextProcessor()

    # Load Greek legal texts
    greek_texts = processor.load_greek_legal_texts()

    # Perform masked language modeling
    masked_results = processor.fill_masked_tokens(greek_texts)

    # Evaluate against ground truth
    evaluation_results = processor.evaluate_against_ground_truth(
        masked_results)

    # Syntactic analysis
    syntactic_analysis = processor.perform_syntactic_analysis(masked_results)

    logger.info("✅ Bonus deliverable completed!")
    return masked_results, evaluation_results, syntactic_analysis


def generate_report(all_results):
    """
    Παραδοτέο 3: Δομημένη Αναφορά
    """
    logger.info("📝 Generating structured report...")

    from src.utils.report_generator import ReportGenerator

    report_generator = ReportGenerator()
    report_path = report_generator.generate_comprehensive_report(all_results)

    logger.info(f"📋 Report generated: {report_path}")
    return report_path


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="NLP Assignment 2025")
    parser.add_argument("--deliverable", type=int, choices=[1, 2, 3],
                        help="Run specific deliverable (1, 2, or 3)")
    parser.add_argument("--bonus", action="store_true",
                        help="Run bonus masked language modeling")
    parser.add_argument("--all", action="store_true",
                        help="Run all deliverables")

    args = parser.parse_args()

    logger.info("🎯 Starting NLP Assignment 2025")

    all_results = {}

    try:
        if args.all or args.deliverable == 1 or args.deliverable is None:
            # Run Deliverable 1
            custom_results, library_results, comparison_results = run_deliverable_1()
            all_results['deliverable_1'] = {
                'custom': custom_results,
                'library': library_results,
                'comparison': comparison_results
            }

        if args.all or args.deliverable == 2 or args.deliverable is None:
            # Run Deliverable 2
            if 'deliverable_1' not in all_results:
                logger.warning(
                    "Deliverable 1 results not available. Running Deliverable 1 first...")
                custom_results, library_results, comparison_results = run_deliverable_1()
                all_results['deliverable_1'] = {
                    'custom': custom_results,
                    'library': library_results,
                    'comparison': comparison_results
                }

            embedding_results, similarity_results, visualization_results = run_deliverable_2(
                all_results['deliverable_1']['custom'],
                all_results['deliverable_1']['library']
            )
            all_results['deliverable_2'] = {
                'embeddings': embedding_results,
                'similarities': similarity_results,
                'visualizations': visualization_results
            }

        if args.bonus:
            # Run Bonus
            masked_results, evaluation_results, syntactic_analysis = run_bonus_deliverable()
            all_results['bonus'] = {
                'masked': masked_results,
                'evaluation': evaluation_results,
                'syntactic': syntactic_analysis
            }

        if args.all or args.deliverable == 3 or args.deliverable is None:
            # Generate Report (Deliverable 3)
            report_path = generate_report(all_results)
            all_results['deliverable_3'] = {'report_path': report_path}

        logger.info("🎉 All deliverables completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error during execution: {str(e)}")
        raise

    return all_results


if __name__ == "__main__":
    main()
