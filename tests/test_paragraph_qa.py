"""
Unit tests for Paragraph QA Evaluator.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paragraph_qa_evaluator import ParagraphQAEvaluator


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
        self.assertTrue(self.evaluator._check_answer_correctness(
            "computational resources", "computational resources"
        ))
        
        # Substring match
        self.assertTrue(self.evaluator._check_answer_correctness(
            "requires computational resources", "computational resources"
        ))
        
        # Word overlap
        self.assertTrue(self.evaluator._check_answer_correctness(
            "resources and datasets", "computational resources"
        ))
        
        # No match
        self.assertFalse(self.evaluator._check_answer_correctness(
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
        with self.assertRaises(ValueError):
            evaluator.validate_paragraph_answer(
                "Test question", "Test answer", 1
            )
    
    def test_validate_answer_invalid_paragraph(self):
        """Test validation with invalid paragraph index."""
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer(
                "Test question", "Test answer", 10
            )


if __name__ == '__main__':
    unittest.main()
