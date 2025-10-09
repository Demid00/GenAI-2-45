"""
Simple tests to verify basic functionality.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from paragraph_qa_evaluator import ParagraphQAEvaluator


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests."""
    
    def test_initialization(self):
        """Test that evaluator initializes correctly."""
        evaluator = ParagraphQAEvaluator()
        self.assertEqual(evaluator.model_name, "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")
        self.assertEqual(evaluator.context, "")
        self.assertEqual(evaluator.paragraphs, [])
        self.assertIsNone(evaluator.qa_pipeline)
    
    def test_document_creation(self):
        """Test document creation methods."""
        evaluator = ParagraphQAEvaluator()
        
        # Test three paragraph document
        context, paragraphs = evaluator.create_three_paragraph_document()
        self.assertEqual(len(paragraphs), 3)
        self.assertIn("Artificial Intelligence", paragraphs[0])
        self.assertIn("Deep learning", paragraphs[1])
        self.assertEqual(context, "\n\n".join(paragraphs))
        
        # Test custom document
        custom_paras = ["Para 1", "Para 2", "Para 3"]
        context, paras = evaluator.create_custom_document(custom_paras)
        self.assertEqual(paras, custom_paras)
    
    def test_error_handling(self):
        """Test basic error handling."""
        evaluator = ParagraphQAEvaluator()
        
        # Should raise ValueError for empty paragraphs
        with self.assertRaises(ValueError):
            evaluator.create_custom_document([])
        
        # Should raise ValueError for None paragraphs
        with self.assertRaises(ValueError):
            evaluator.create_custom_document(None)


class TestAnswerValidation(unittest.TestCase):
    """Test answer validation logic."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        self.evaluator.create_three_paragraph_document()
    
    def test_answer_correctness(self):
        """Test answer correctness checking."""
        # Test exact match
        self.assertTrue(self.evaluator._check_answer_correctness(
            "same text", "same text"
        ))
        
        # Test substring match
        self.assertTrue(self.evaluator._check_answer_correctness(
            "this has computational resources inside", 
            "computational resources"
        ))
        
        # Test no match
        self.assertFalse(self.evaluator._check_answer_correctness(
            "completely different",
            "expected answer"
        ))
    
    def test_paragraph_detection(self):
        """Test paragraph boundary detection."""
        # Create a simple document for testing boundaries
        test_paragraphs = ["First paragraph", "Second paragraph", "Third paragraph"]
        self.evaluator.create_custom_document(test_paragraphs)
        
        context = self.evaluator.context
        
        # Test positions within first paragraph
        self.assertTrue(self.evaluator._is_answer_from_paragraph(
            "First", 0, 5, 0
        ))
        
        # Test positions within second paragraph
        second_para_start = len("First paragraph") + 2  # +2 for "\n\n"
        self.assertTrue(self.evaluator._is_answer_from_paragraph(
            "Second", second_para_start, second_para_start + 6, 1
        ))


class TestIntegration(unittest.TestCase):
    """Integration tests with mocked model."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        self.evaluator.create_three_paragraph_document()
        
        # Mock the pipeline
        self.mock_pipeline = Mock()
        self.evaluator.qa_pipeline = self.mock_pipeline
    
    def test_successful_validation(self):
        """Test successful validation scenario."""
        # Mock successful response
        self.mock_pipeline.return_value = {
            'answer': 'substantial computational resources',
            'score': 0.95,
            'start': 450,
            'end': 485
        }
        
        result = self.evaluator.validate_paragraph_answer(
            question="Test question",
            expected_answer="substantial computational resources",
            target_paragraph_index=1
        )
        
        self.assertTrue(result['success'])
        self.assertTrue(result['answer_from_target_paragraph'])
        self.assertTrue(result['answer_correct'])
    
    def test_error_handling_in_validation(self):
        """Test error handling during validation."""
        # Mock pipeline failure
        self.mock_pipeline.side_effect = Exception("Mock error")
        
        result = self.evaluator.validate_paragraph_answer(
            question="Test question",
            expected_answer="expected answer",
            target_paragraph_index=1
        )
        
        self.assertFalse(result['success'])
        self.assertEqual(result['model_answer'], 'ERROR')
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
