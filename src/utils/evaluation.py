"""
Evaluation Metrics for Text Reconstruction
Handles comparison and evaluation of different reconstruction approaches
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass
import re
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """Data class for comparison metrics"""
    method_name: str
    total_corrections: int
    average_confidence: float
    processing_time: float
    text_quality_score: float
    readability_score: float
    grammar_improvement_score: float


class EvaluationMetrics:
    """
    Evaluation metrics class for comparing reconstruction methods
    """

    def __init__(self):
        self.metrics = {}

    def calculate_text_quality_score(self, original: str, reconstructed: str) -> float:
        """
        Calculate text quality improvement score

        Args:
            original: Original text
            reconstructed: Reconstructed text

        Returns:
            Quality score between 0 and 1
        """
        quality_score = 0.5  # Base score

        # Check for improvements in common problematic patterns
        improvements = 0
        total_checks = 0

        # Check character encoding fixes
        encoding_issues = ['ﬁ', 'eﬀ', 'coﬀ']
        for issue in encoding_issues:
            total_checks += 1
            if issue in original and issue not in reconstructed:
                improvements += 1

        # Check grammar improvements
        grammar_issues = [
            r'\bThank\s+your\s+message\b',
            r'\bI\s+am\s+very\s+appreciated\b',
            r'\bduring\s+our\s+.*\s+discuss\b',
            r'\bthe\s+updates\s+was\s+confusing\b'
        ]

        for pattern in grammar_issues:
            total_checks += 1
            if re.search(pattern, original, re.IGNORECASE) and not re.search(pattern, reconstructed, re.IGNORECASE):
                improvements += 1

        # Calculate improvement ratio
        if total_checks > 0:
            improvement_ratio = improvements / total_checks
            quality_score += improvement_ratio * 0.4

        # Check for sentence structure improvements
        original_sentences = len(re.findall(r'[.!?]+', original))
        reconstructed_sentences = len(re.findall(r'[.!?]+', reconstructed))

        if reconstructed_sentences >= original_sentences:
            quality_score += 0.1

        return min(quality_score, 1.0)

    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate readability score using simple heuristics

        Args:
            text: Text to analyze

        Returns:
            Readability score between 0 and 1
        """
        score = 0.5  # Base score

        # Check average sentence length
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_sentence_length = sum(len(s.split())
                                      for s in sentences) / len(sentences)

            # Optimal sentence length is around 15-20 words
            if 10 <= avg_sentence_length <= 25:
                score += 0.2
            elif avg_sentence_length > 30:
                score -= 0.1

        # Check for proper capitalization
        if sentences:
            properly_capitalized = sum(
                1 for s in sentences if s and s[0].isupper())
            capitalization_ratio = properly_capitalized / len(sentences)
            score += capitalization_ratio * 0.15

        # Check for proper punctuation
        has_punctuation = bool(re.search(r'[.!?]$', text.strip()))
        if has_punctuation:
            score += 0.15

        # Check for excessive repetition
        words = text.lower().split()
        if words:
            word_counts = Counter(words)
            max_repetition = max(word_counts.values()) if word_counts else 1
            repetition_penalty = min(max_repetition / len(words), 0.2)
            score -= repetition_penalty

        return max(min(score, 1.0), 0.0)

    def calculate_grammar_improvement_score(self, original: str, reconstructed: str) -> float:
        """
        Calculate grammar improvement score

        Args:
            original: Original text
            reconstructed: Reconstructed text

        Returns:
            Grammar improvement score between 0 and 1
        """
        # Count grammar issues in original vs reconstructed
        grammar_patterns = [
            r'\bI\s+am\s+very\s+appreciated\b',  # Wrong: "I am very appreciated"
            r'\bthank\s+your\s+message\b',      # Wrong: "thank your message"
            r'\bwere\s+waiting\s+since\b',      # Wrong: "were waiting since"
            r'\bupdates\s+was\b',               # Wrong: "updates was"
            # Wrong: "plan for" (should be "plans for")
            r'\bplan\s+for\b(?=\s+\w+\s+section)',
            r'\bhe\s+sending\b',                # Wrong: "he sending"
        ]

        original_issues = sum(1 for pattern in grammar_patterns
                              if re.search(pattern, original, re.IGNORECASE))
        reconstructed_issues = sum(1 for pattern in grammar_patterns
                                   if re.search(pattern, reconstructed, re.IGNORECASE))

        if original_issues == 0:
            return 1.0  # No issues to fix

        improvement_ratio = (
            original_issues - reconstructed_issues) / original_issues
        return max(improvement_ratio, 0.0)

    def evaluate_reconstruction_method(self, method_name: str, results: Dict) -> ComparisonMetrics:
        """
        Evaluate a single reconstruction method

        Args:
            method_name: Name of the method
            results: Results dictionary from the method

        Returns:
            ComparisonMetrics object
        """
        total_corrections = 0
        total_confidence = 0
        total_time = 0
        quality_scores = []
        readability_scores = []
        grammar_scores = []

        count = 0

        # Process results based on structure
        if isinstance(results, dict):
            for key, result in results.items():
                if hasattr(result, 'corrections'):
                    total_corrections += len(result.corrections)
                    total_confidence += getattr(result, 'confidence_score', 0)
                    total_time += getattr(result, 'processing_time', 0)

                    # Handle different attribute names for different result types
                    if hasattr(result, 'original_text'):
                        # LibraryPipelineResult
                        original_text = result.original_text
                        reconstructed_text = result.reconstructed_text
                    else:
                        # ReconstructionResult
                        original_text = result.original
                        reconstructed_text = result.reconstructed

                    quality_score = self.calculate_text_quality_score(
                        original_text, reconstructed_text
                    )
                    readability_score = self.calculate_readability_score(
                        reconstructed_text
                    )
                    grammar_score = self.calculate_grammar_improvement_score(
                        original_text, reconstructed_text
                    )

                    quality_scores.append(quality_score)
                    readability_scores.append(readability_score)
                    grammar_scores.append(grammar_score)
                    count += 1

                elif isinstance(result, dict):
                    # Library pipeline results
                    for pipeline_key, pipeline_result in result.items():
                        if hasattr(pipeline_result, 'corrections'):
                            total_corrections += len(
                                pipeline_result.corrections)
                            total_confidence += pipeline_result.confidence_score
                            total_time += pipeline_result.processing_time

                            quality_score = self.calculate_text_quality_score(
                                pipeline_result.original_text, pipeline_result.reconstructed_text
                            )
                            readability_score = self.calculate_readability_score(
                                pipeline_result.reconstructed_text
                            )
                            grammar_score = self.calculate_grammar_improvement_score(
                                pipeline_result.original_text, pipeline_result.reconstructed_text
                            )

                            quality_scores.append(quality_score)
                            readability_scores.append(readability_score)
                            grammar_scores.append(grammar_score)
                            count += 1

        # Calculate averages
        avg_confidence = total_confidence / count if count > 0 else 0
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        avg_readability = np.mean(
            readability_scores) if readability_scores else 0
        avg_grammar = np.mean(grammar_scores) if grammar_scores else 0

        return ComparisonMetrics(
            method_name=method_name,
            total_corrections=total_corrections,
            average_confidence=avg_confidence,
            processing_time=total_time,
            text_quality_score=avg_quality,
            readability_score=avg_readability,
            grammar_improvement_score=avg_grammar
        )

    def compare_reconstruction_methods(self, custom_results: Dict, library_results: Dict) -> Dict:
        """
        Compare custom and library reconstruction methods

        Args:
            custom_results: Results from custom pipeline
            library_results: Results from library pipelines

        Returns:
            Comprehensive comparison dictionary
        """
        logger.info("📊 Starting reconstruction method comparison...")

        # Evaluate custom method
        custom_metrics = self.evaluate_reconstruction_method(
            "Custom Pipeline", custom_results)

        # Evaluate library methods
        library_metrics = {}

        # Extract individual pipeline results from library results
        for text_key, text_results in library_results.items():
            for pipeline_key, pipeline_result in text_results.items():
                pipeline_name = getattr(
                    pipeline_result, 'pipeline_name', f'Library Pipeline {pipeline_key}')

                if pipeline_name not in library_metrics:
                    library_metrics[pipeline_name] = []

                # Create a temporary dict for evaluation
                temp_results = {f"{text_key}_{pipeline_key}": pipeline_result}
                metrics = self.evaluate_reconstruction_method(
                    pipeline_name, temp_results)
                library_metrics[pipeline_name].append(metrics)

        # Aggregate library metrics
        aggregated_library_metrics = {}
        for pipeline_name, metrics_list in library_metrics.items():
            if metrics_list:
                aggregated_metrics = ComparisonMetrics(
                    method_name=pipeline_name,
                    total_corrections=sum(
                        m.total_corrections for m in metrics_list),
                    average_confidence=np.mean(
                        [m.average_confidence for m in metrics_list]),
                    processing_time=sum(
                        m.processing_time for m in metrics_list),
                    text_quality_score=np.mean(
                        [m.text_quality_score for m in metrics_list]),
                    readability_score=np.mean(
                        [m.readability_score for m in metrics_list]),
                    grammar_improvement_score=np.mean(
                        [m.grammar_improvement_score for m in metrics_list])
                )
                aggregated_library_metrics[pipeline_name] = aggregated_metrics

        # Create comparison summary
        all_methods = [custom_metrics] + \
            list(aggregated_library_metrics.values())

        # Find best performing methods
        best_quality = max(all_methods, key=lambda x: x.text_quality_score)
        best_readability = max(all_methods, key=lambda x: x.readability_score)
        best_grammar = max(
            all_methods, key=lambda x: x.grammar_improvement_score)
        fastest = min(all_methods, key=lambda x: x.processing_time)
        most_corrections = max(all_methods, key=lambda x: x.total_corrections)

        comparison_results = {
            'custom_method_metrics': {
                'method_name': custom_metrics.method_name,
                'total_corrections': custom_metrics.total_corrections,
                'average_confidence': custom_metrics.average_confidence,
                'processing_time': custom_metrics.processing_time,
                'text_quality_score': custom_metrics.text_quality_score,
                'readability_score': custom_metrics.readability_score,
                'grammar_improvement_score': custom_metrics.grammar_improvement_score
            },
            'library_methods_metrics': {
                name: {
                    'method_name': metrics.method_name,
                    'total_corrections': metrics.total_corrections,
                    'average_confidence': metrics.average_confidence,
                    'processing_time': metrics.processing_time,
                    'text_quality_score': metrics.text_quality_score,
                    'readability_score': metrics.readability_score,
                    'grammar_improvement_score': metrics.grammar_improvement_score
                }
                for name, metrics in aggregated_library_metrics.items()
            },
            'best_performers': {
                'best_text_quality': best_quality.method_name,
                'best_readability': best_readability.method_name,
                'best_grammar_improvement': best_grammar.method_name,
                'fastest_processing': fastest.method_name,
                'most_corrections': most_corrections.method_name
            },
            'summary_statistics': {
                'total_methods_compared': len(all_methods),
                'average_quality_score': np.mean([m.text_quality_score for m in all_methods]),
                'average_readability_score': np.mean([m.readability_score for m in all_methods]),
                'average_grammar_score': np.mean([m.grammar_improvement_score for m in all_methods]),
                'total_processing_time': sum(m.processing_time for m in all_methods)
            },
            'comparison_insights': self._generate_comparison_insights(custom_metrics, aggregated_library_metrics)
        }

        logger.info("✅ Reconstruction method comparison completed")
        return comparison_results

    def _generate_comparison_insights(self, custom_metrics: ComparisonMetrics,
                                      library_metrics: Dict[str, ComparisonMetrics]) -> List[str]:
        """Generate insights from the comparison"""
        insights = []

        all_library_scores = [
            m.text_quality_score for m in library_metrics.values()]

        if all_library_scores:
            avg_library_quality = np.mean(all_library_scores)

            if custom_metrics.text_quality_score > avg_library_quality:
                insights.append(
                    "Custom pipeline shows superior text quality improvement compared to library methods")
            else:
                insights.append(
                    "Library pipelines demonstrate better text quality improvement on average")

        # Processing time insights
        all_library_times = [
            m.processing_time for m in library_metrics.values()]
        if all_library_times:
            avg_library_time = np.mean(all_library_times)

            if custom_metrics.processing_time < avg_library_time:
                insights.append(
                    "Custom pipeline is more efficient in terms of processing time")
            else:
                insights.append(
                    "Library pipelines are generally faster than the custom approach")

        # Grammar improvement insights
        all_library_grammar = [
            m.grammar_improvement_score for m in library_metrics.values()]
        if all_library_grammar:
            avg_library_grammar = np.mean(all_library_grammar)

            if custom_metrics.grammar_improvement_score > avg_library_grammar:
                insights.append(
                    "Custom pipeline excels at grammar correction compared to library methods")

        # Correction count insights
        if custom_metrics.total_corrections > 10:
            insights.append(
                "Custom pipeline applies extensive corrections, indicating thorough processing")

        return insights
