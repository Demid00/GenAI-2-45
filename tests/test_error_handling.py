"""
Error handling tests for Paragraph QA Evaluator.

Tests various error scenarios and exception handling to ensure
robust operation under different failure conditions.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paragraph_qa_evaluator import ParagraphQAEvaluator


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ParagraphQAEvaluator()
    
    def test_invalid_model_loading(self):
        """Test behavior with invalid model name."""
        # This should not crash but return False
        invalid_evaluator = ParagraphQAEvaluator(model_name="invalid/model/name")
        result = invalid_evaluator.load_model()
        self.assertFalse(result)
    
    def test_empty_question_validation(self):
        """Test validation with empty question."""
        self.evaluator.create_three_paragraph_document()
        
        # Mock the pipeline to avoid actual model loading
        self.evaluator.qa_pipeline = True  # Simulate loaded model
        
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer("", "test answer", 1)
    
    def test_none_question_validation(self):
        """Test validation with None question."""
        self.evaluator.create_three_paragraph_document()
        self.evaluator.qa_pipeline = True
        
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer(None, "test answer", 1)
    
    def test_negative_paragraph_index(self):
        """Test validation with negative paragraph index."""
        self.evaluator.create_three_paragraph_document()
        self.evaluator.qa_pipeline = True
        
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer("test question", "test answer", -1)
    
    def test_out_of_bounds_paragraph_index(self):
        """Test validation with out-of-bounds paragraph index."""
        self.evaluator.create_three_paragraph_document()
        self.evaluator.qa_pipeline = True
        
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer("test question", "test answer", 10)
    
    def test_empty_paragraphs_list(self):
        """Test creating document with empty paragraphs list."""
        with self.assertRaises(ValueError):
            self.evaluator.create_custom_document([])
    
    def test_none_paragraphs_list(self):
        """Test creating document with None paragraphs list."""
        with self.assertRaises(ValueError):
            self.evaluator.create_custom_document(None)
    
    def test_paragraphs_with_empty_strings(self):
        """Test creating document with paragraphs containing empty strings."""
        # Should filter out empty strings
        paragraphs = ["First paragraph", "", "Third paragraph"]
        context, processed_paragraphs = self.evaluator.create_custom_document(paragraphs)
        
        self.assertEqual(len(processed_paragraphs), 2)  # Empty string filtered out
        self.assertIn("First paragraph", processed_paragraphs)
        self.assertIn("Third paragraph", processed_paragraphs)
    
    def test_answer_correctness_edge_cases(self):
        """Test answer correctness with edge cases."""
        # Empty answers
        self.assertFalse(self.evaluator._check_answer_correctness("", "expected"))
        self.assertFalse(self.evaluator._check_answer_correctness("answer", ""))
        self.assertFalse(self.evaluator._check_answer_correctness("", ""))
        
        # None answers
        self.assertFalse(self.evaluator._check_answer_correctness(None, "expected"))
        self.assertFalse(self.evaluator._check_answer_correctness("answer", None))
        self.assertFalse(self.evaluator._check_answer_correctness(None, None))
    
    def test_whitespace_handling(self):
        """Test answer correctness with various whitespace scenarios."""
        # Different whitespace, same content
        self.assertTrue(self.evaluator._check_answer_correctness(
            "  computational resources  ",
            "computational resources"
        ))
        
        self.assertTrue(self.evaluator._check_answer_correctness(
            "computational\nresources",
            "computational resources"
        ))
    
    def test_case_insensitive_matching(self):
        """Test that answer matching is case insensitive."""
        self.assertTrue(self.evaluator._check_answer_correctness(
            "COMPUTATIONAL RESOURCES",
            "computational resources"
        ))
        
        self.assertTrue(self.evaluator._check_answer_correctness(
            "Computational Resources",
            "COMPUTATIONAL RESOURCES"
        ))


class TestParagraphBoundaryDetection(unittest.TestCase):
    """Test paragraph boundary detection functionality."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        
        # Create a simple test document
        self.test_paragraphs = [
            "First paragraph with some content.",
            "Second paragraph containing different information.",
            "Third paragraph with additional details."
        ]
        self.evaluator.create_custom_document(self.test_paragraphs)
    
    def test_paragraph_boundary_calculation(self):
        """Test that paragraph boundaries are calculated correctly."""
        # For our test document, boundaries should be:
        # Para 0: 0-35
        # Para 1: 37-78 (35 + 2 for "\n\n" + 41 chars)
        # Para 2: 80-118 (78 + 2 for "\n\n" + 38 chars)
        
        # This is a simplified test - in practice, we'd need to mock the actual positions
        # For now, just verify the context is constructed correctly
        expected_context = "\n\n".join(self.test_paragraphs)
        self.assertEqual(self.evaluator.context, expected_context)
    
    def test_answer_from_paragraph_logic(self):
        """Test the logic for detecting if answer comes from target paragraph."""
        # Mock answer positions - these would normally come from the model
        # For paragraph 1 (index 0)
        self.assertTrue(self.evaluator._is_answer_from_paragraph(
            "some content", 10, 22, 0
        ))
        
        # For paragraph 2 (index 1)
        self.assertTrue(self.evaluator._is_answer_from_paragraph(
            "different information", 50, 70, 1
        ))


class TestResultHandling(unittest.TestCase):
    """Test result handling and reporting."""
    
    def test_error_result_structure(self):
        """Test that error results have consistent structure."""
        evaluator = ParagraphQAEvaluator()
        evaluator.create_three_paragraph_document()
        
        # Simulate an error by not loading the model
        result = evaluator.validate_paragraph_answer("test", "answer", 1)
        
        self.assertIn('error', result)
        self.assertFalse(result['success'])
        self.assertEqual(result['model_answer'], 'ERROR')
        self.assertEqual(result['score'], 0.0)


if __name__ == '__main__':
    unittest.main()
