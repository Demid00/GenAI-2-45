"""
Unit tests for Paragraph QA Evaluator.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path and import directly from module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from paragraph_qa_evaluator import ParagraphQAEvaluator


class TestParagraphQAEvaluator(unittest.TestCase):
    """Test cases for ParagraphQAEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ParagraphQAEvaluator()
    
    def test_initialization(self):
        """Test class initialization."""
        self.assertEqual(self.evaluator.model_name, "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")
        self.assertEqual(self.evaluator.context, "")
        self.assertEqual(self.evaluator.paragraphs, [])
        self.assertIsNone(self.evaluator.qa_pipeline)
    
    def test_create_three_paragraph_document(self):
        """Test document creation with 3 paragraphs."""
        context, paragraphs = self.evaluator.create_three_paragraph_document()
        
        self.assertEqual(len(paragraphs), 3)
        self.assertIn("Artificial Intelligence", paragraphs[0])
        self.assertIn("Deep learning", paragraphs[1])
        self.assertIn("ethical implications", paragraphs[2])
        self.assertEqual(context, "\n\n".join(paragraphs))
    
    def test_create_custom_document(self):
        """Test custom document creation."""
        custom_paragraphs = [
            "First paragraph content.",
            "Second paragraph with different content.",
            "Third paragraph here."
        ]
        
        context, paragraphs = self.evaluator.create_custom_document(custom_paragraphs)
        
        self.assertEqual(len(paragraphs), 3)
        self.assertEqual(paragraphs, custom_paragraphs)
        self.assertEqual(context, "\n\n".join(custom_paragraphs))
    
    def test_create_custom_document_empty(self):
        """Test custom document creation with empty input."""
        with self.assertRaises(ValueError):
            self.evaluator.create_custom_document([])
    
    def test_answer_correctness_check(self):
        """Test answer correctness validation."""
        # Exact match
        self.assertTrue(self.evaluator._ParagraphQAEvaluator__check_answer_correctness(
            "computational resources", "computational resources"
        ))
        
        # Substring match
        self.assertTrue(self.evaluator._ParagraphQAEvaluator__check_answer_correctness(
            "requires computational resources", "computational resources"
        ))
        
        # Word overlap
        self.assertTrue(self.evaluator._ParagraphQAEvaluator__check_answer_correctness(
            "resources and datasets", "computational resources"
        ))
        
        # No match
        self.assertFalse(self.evaluator._ParagraphQAEvaluator__check_answer_correctness(
            "completely different", "computational resources"
        ))


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        self.evaluator.create_three_paragraph_document()
    
    def test_validate_answer_without_model(self):
        """Test validation without loaded model."""
        with self.assertRaises(RuntimeError):
            self.evaluator.validate_paragraph_answer(
                "Test question", "Test answer", 1
            )
    
    def test_validate_answer_without_context(self):
        """Test validation without context."""
        evaluator = ParagraphQAEvaluator()
        
        # Mock that model is loaded but context is missing
        with patch.object(evaluator, 'qa_pipeline', Mock()):
            with self.assertRaises(ValueError):
                evaluator.validate_paragraph_answer(
                    "Test question", "Test answer", 1
                )
    
    def test_validate_answer_invalid_paragraph(self):
        """Test validation with invalid paragraph index."""
        # Mock that model is loaded
        with patch.object(self.evaluator, 'qa_pipeline', Mock()):
            with self.assertRaises(ValueError):
                self.evaluator.validate_paragraph_answer(
                    "Test question", "Test answer", 10
                )


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios with mocked model."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        self.evaluator.create_three_paragraph_document()
        
        # Create a mock pipeline that returns predictable results
        self.mock_pipeline = Mock()
        self.mock_pipeline.return_value = {
            'answer': 'substantial computational resources and large labeled datasets',
            'score': 0.92,
            'start': 450,
            'end': 512
        }
        self.evaluator.qa_pipeline = self.mock_pipeline
    
    def test_successful_validation(self):
        """Test successful answer validation."""
        result = self.evaluator.validate_paragraph_answer(
            question="What do deep learning models require for training?",
            expected_answer="substantial computational resources and large labeled datasets",
            target_paragraph_index=1
        )
        
        self.assertTrue(result['success'])
        self.assertTrue(result['answer_from_target_paragraph'])
        self.assertTrue(result['answer_correct'])
        self.assertAlmostEqual(result['score'], 0.92, places=2)
    
    def test_failed_paragraph_localization(self):
        """Test when answer is correct but from wrong paragraph."""
        # Mock pipeline to return answer from wrong position
        self.mock_pipeline.return_value = {
            'answer': 'machine learning and natural language processing',
            'score': 0.85,
            'start': 50,  # Position in first paragraph, not second
            'end': 90
        }
        
        result = self.evaluator.validate_paragraph_answer(
            question="What fields does AI encompass?",
            expected_answer="machine learning and natural language processing", 
            target_paragraph_index=1  # But answer comes from paragraph 0
        )
        
        self.assertFalse(result['success'])
        self.assertFalse(result['answer_from_target_paragraph'])
        self.assertTrue(result['answer_correct'])  # Content is correct
    
    def test_incorrect_answer_content(self):
        """Test when answer is from correct paragraph but content is wrong."""
        self.mock_pipeline.return_value = {
            'answer': 'completely wrong answer',
            'score': 0.45,
            'start': 460,  # Position in second paragraph
            'end': 485
        }
        
        result = self.evaluator.validate_paragraph_answer(
            question="What do models require?",
            expected_answer="computational resources",
            target_paragraph_index=1
        )
        
        self.assertFalse(result['success'])
        self.assertTrue(result['answer_from_target_paragraph'])  # Right paragraph
        self.assertFalse(result['answer_correct'])  # Wrong content


if __name__ == '__main__':
    unittest.main()
