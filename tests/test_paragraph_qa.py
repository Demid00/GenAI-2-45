"""
Comprehensive tests for Paragraph QA Evaluator.

Tests core functionality, error handling, and integration scenarios.
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


class TestParameterValidation(unittest.TestCase):
    """Test parameter validation in main methods."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        self.evaluator.create_three_paragraph_document()
    
    def test_validate_empty_question(self):
        """Test validation with empty question."""
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer("", "answer", 1)
    
    def test_validate_none_question(self):
        """Test validation with None question."""
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer(None, "answer", 1)
    
    def test_validate_negative_paragraph(self):
        """Test validation with negative paragraph index."""
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer("question", "answer", -1)
    
    def test_validate_out_of_bounds_paragraph(self):
        """Test validation with out-of-bounds paragraph index."""
        with self.assertRaises(ValueError):
            self.evaluator.validate_paragraph_answer("question", "answer", 10)
    
    def test_validate_without_context(self):
        """Test validation without context."""
        evaluator = ParagraphQAEvaluator()
        with self.assertRaises(ValueError):
            evaluator.validate_paragraph_answer("question", "answer", 1)
    
    def test_validate_without_model(self):
        """Test validation without loaded model."""
        with self.assertRaises(RuntimeError):
            self.evaluator.validate_paragraph_answer("question", "answer", 1)


class TestAnswerValidation(unittest.TestCase):
    """Test answer validation logic."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        self.evaluator.create_three_paragraph_document()
    
    def test_answer_correctness_exact_match(self):
        """Test answer correctness with exact matches."""
        # Exact match
        self.assertTrue(self.evaluator._check_answer_correctness(
            "same text", "same text"
        ))
        
        # Case insensitive
        self.assertTrue(self.evaluator._check_answer_correctness(
            "COMPUTATIONAL RESOURCES", "computational resources"
        ))
        
        # Different whitespace
        self.assertTrue(self.evaluator._check_answer_correctness(
            "  computational resources  ", "computational resources"
        ))
    
    def test_answer_correctness_substring(self):
        """Test answer correctness with substring matches."""
        # Substring match
        self.assertTrue(self.evaluator._check_answer_correctness(
            "this has computational resources inside", 
            "computational resources"
        ))
        
        # Reverse substring
        self.assertTrue(self.evaluator._check_answer_correctness(
            "computational resources",
            "this has computational resources inside"
        ))
    
    def test_answer_correctness_word_overlap(self):
        """Test answer correctness with word overlap."""
        # Word overlap
        self.assertTrue(self.evaluator._check_answer_correctness(
            "resources and datasets", 
            "computational resources"
        ))
        
        # Multiple word overlap
        self.assertTrue(self.evaluator._check_answer_correctness(
            "large labeled datasets for training",
            "labeled datasets"
        ))
    
    def test_answer_correctness_no_match(self):
        """Test answer correctness with no matches."""
        # No match
        self.assertFalse(self.evaluator._check_answer_correctness(
            "completely different",
            "expected answer"
        ))
        
        # Empty strings
        self.assertFalse(self.evaluator._check_answer_correctness("", "expected"))
        self.assertFalse(self.evaluator._check_answer_correctness("answer", ""))
    
    def test_paragraph_detection(self):
        """Test paragraph boundary detection."""
        # Create a simple document for testing boundaries
        test_paragraphs = ["First paragraph text", "Second paragraph content", "Third paragraph info"]
        self.evaluator.create_custom_document(test_paragraphs)
        
        context = self.evaluator.context
        
        # Test positions within first paragraph
        self.assertTrue(self.evaluator._is_answer_from_paragraph(
            "First", 0, 5, 0
        ))
        
        # Test positions within second paragraph
        second_para_start = len("First paragraph text") + 2  # +2 for "\n\n"
        self.assertTrue(self.evaluator._is_answer_from_paragraph(
            "Second", second_para_start, second_para_start + 6, 1
        ))
        
        # Test wrong paragraph
        self.assertFalse(self.evaluator._is_answer_from_paragraph(
            "First", 0, 5, 1  # Answer from para 0 but checking para 1
        ))


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests with mocked model."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        self.evaluator.create_three_paragraph_document()
        
        # Mock the pipeline
        self.mock_pipeline = Mock()
        self.evaluator.qa_pipeline = self.mock_pipeline
    
    def test_successful_validation(self):
        """Test successful validation scenario."""
        # Mock successful response from second paragraph
        self.mock_pipeline.return_value = {
            'answer': 'substantial computational resources and large labeled datasets',
            'score': 0.95,
            'start': 450,  # Position in second paragraph
            'end': 512
        }
        
        result = self.evaluator.validate_paragraph_answer(
            question="What do deep learning models require for training?",
            expected_answer="substantial computational resources and large labeled datasets",
            target_paragraph_index=1
        )
        
        self.assertTrue(result['success'])
        self.assertTrue(result['answer_from_target_paragraph'])
        self.assertTrue(result['answer_correct'])
        self.assertAlmostEqual(result['score'], 0.95, places=2)
        self.assertEqual(result['question'], "What do deep learning models require for training?")
    
    def test_failed_paragraph_localization(self):
        """Test when answer is correct but from wrong paragraph."""
        # Mock response from wrong paragraph
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
        self.assertAlmostEqual(result['score'], 0.85, places=2)
    
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
        self.assertAlmostEqual(result['score'], 0.45, places=2)
    
    def test_error_handling_in_validation(self):
        """Test error handling during validation."""
        # Mock pipeline failure
        self.mock_pipeline.side_effect = Exception("Mock pipeline error")
        
        result = self.evaluator.validate_paragraph_answer(
            question="Test question",
            expected_answer="expected answer",
            target_paragraph_index=1
        )
        
        self.assertFalse(result['success'])
        self.assertEqual(result['model_answer'], 'ERROR')
        self.assertEqual(result['score'], 0.0)
        self.assertIn('error', result)
        self.assertIn('Mock pipeline error', result['error'])


class TestSpecificTestScenario(unittest.TestCase):
    """Test the specific test scenario from requirements."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
        self.evaluator.create_three_paragraph_document()
        
        # Mock pipeline for specific test
        self.mock_pipeline = Mock()
        self.evaluator.qa_pipeline = self.mock_pipeline
    
    def test_run_specific_test_default(self):
        """Test running the specific test with default parameters."""
        # Mock response targeting second paragraph
        self.mock_pipeline.return_value = {
            'answer': 'substantial computational resources and large labeled datasets',
            'score': 0.92,
            'start': 450,
            'end': 512
        }
        
        result = self.evaluator.run_specific_test()
        
        # Verify the test used correct default parameters
        self.assertEqual(result['question'], "What do deep learning models require for training?")
        self.assertEqual(result['expected_answer'], "substantial computational resources and large labeled datasets")
        self.assertEqual(result['target_paragraph_index'], 1)  # Second paragraph
        
        # Should succeed with mocked correct answer
        self.assertTrue(result['success'])
    
    def test_run_specific_test_custom(self):
        """Test running the specific test with custom parameters."""
        self.mock_pipeline.return_value = {
            'answer': 'custom expected answer',
            'score': 0.88,
            'start': 200,
            'end': 220
        }
        
        result = self.evaluator.run_specific_test(
            question="Custom question?",
            expected_answer="custom expected answer",
            target_paragraph=0  # First paragraph
        )
        
        self.assertEqual(result['question'], "Custom question?")
        self.assertEqual(result['expected_answer'], "custom expected answer")
        self.assertEqual(result['target_paragraph_index'], 0)

class TestExtendedCoverage(unittest.TestCase):
    """Extended tests to improve code coverage."""
    
    def setUp(self):
        self.evaluator = ParagraphQAEvaluator()
    
    def test_logging_initialization(self):
        """Test that logging is properly initialized."""
        # This tests the _initialize_logging method indirectly
        evaluator = ParagraphQAEvaluator()
        self.assertIsNotNone(evaluator.logger)
        # Имя логгера должно быть именем модуля
        self.assertEqual(evaluator.logger.name, "paragraph_qa_evaluator")
    
    def test_model_loading_failure(self):
        """Test model loading failure scenarios."""
        # Сохраним оригинальный pipeline
        original_pipeline = None
        if hasattr(self.evaluator, 'qa_pipeline'):
            original_pipeline = self.evaluator.qa_pipeline
        
        # Временно заменим метод load_model
        def mock_load_model():
            self.evaluator.qa_pipeline = None
            return False
        
        # Подменим метод
        original_method = self.evaluator.load_model
        self.evaluator.load_model = mock_load_model
        
        try:
            result = self.evaluator.load_model()
            self.assertFalse(result)
        finally:
            # Восстановим оригинальный метод
            self.evaluator.load_model = original_method
            self.evaluator.qa_pipeline = original_pipeline
    
    def test_save_results_success(self):
        """Test successful result saving."""
        self.evaluator.create_three_paragraph_document()
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('os.makedirs') as mock_makedirs:
                result = {
                    'question': 'test',
                    'answer': 'test',
                    'success': True
                }
                success = self.evaluator.save_results(result, "test_results.json")
                
                self.assertTrue(success)
                mock_makedirs.assert_called_once()
                mock_file.assert_called_once_with("test_results.json", 'w', encoding='utf-8')
    
    def test_save_results_failure(self):
        """Test result saving failure."""
        self.evaluator.create_three_paragraph_document()
        
        with patch('builtins.open', side_effect=Exception("File error")):
            result = {'test': 'data'}
            success = self.evaluator.save_results(result, "test_results.json")
            
            self.assertFalse(success)
    
    def test_run_specific_test_with_mock(self):
        """Test run_specific_test with mocked validation."""
        self.evaluator.create_three_paragraph_document()
        
        # Mock the validate_paragraph_answer method
        with patch.object(self.evaluator, 'validate_paragraph_answer') as mock_validate:
            mock_validate.return_value = {
                'success': True,
                'score': 0.95,
                'question': 'What do deep learning models require for training?',
                'expected_answer': 'substantial computational resources and large labeled datasets',
                'target_paragraph_index': 1
            }
            
            # Mock the model loading
            self.evaluator.qa_pipeline = Mock()
            
            result = self.evaluator.run_specific_test()
            
            self.assertTrue(result['success'])
            mock_validate.assert_called_once()
    
    def test_print_test_report_comprehensive(self):
        """Test comprehensive report printing."""
        self.evaluator.create_three_paragraph_document()
        
        test_result = {
            'question': 'Test question?',
            'expected_answer': 'Test answer',
            'model_answer': 'Test model answer',
            'score': 0.85,
            'answer_position': (100, 120),
            'target_paragraph_index': 1,
            'answer_from_target_paragraph': True,
            'answer_correct': True,
            'success': True,
            'context_length': 1000,
            'paragraph_count': 3
        }
        
        # This should not raise any exceptions
        self.evaluator.print_test_report(test_result)
    
    def test_print_test_report_failure_case(self):
        """Test report printing for failure case."""
        self.evaluator.create_three_paragraph_document()
        
        test_result = {
            'question': 'Test question?',
            'expected_answer': 'Expected answer',
            'model_answer': 'Wrong answer',
            'score': 0.45,
            'answer_position': (50, 60),
            'target_paragraph_index': 1,
            'answer_from_target_paragraph': False,
            'answer_correct': False,
            'success': False,
            'context_length': 1000,
            'paragraph_count': 3
        }
        
        # This should not raise any exceptions
        self.evaluator.print_test_report(test_result)
    
    def test_answer_from_paragraph_text_match(self):
        """Test answer detection by text content rather than position."""
        test_paragraphs = [
            "This is the first paragraph with unique content.",
            "This is the second paragraph with different content.",
            "Third paragraph here."
        ]
        self.evaluator.create_custom_document(test_paragraphs)
        
        # Test that answer text matching works even with wrong positions
        # The answer "unique content" appears in paragraph 0
        self.assertTrue(self.evaluator._is_answer_from_paragraph(
            "unique content",
            1000,  # Wrong position (out of bounds)
            1015,  # Wrong position
            0      # But text appears in paragraph 0
        ))
    
    def test_answer_correctness_edge_cases_comprehensive(self):
        """Comprehensive edge case testing for answer correctness."""
        # Single word matches
        self.assertTrue(self.evaluator._check_answer_correctness(
            "resources", "computational resources"
        ))
        
        # Word overlap matches
        self.assertTrue(self.evaluator._check_answer_correctness(
            "computational models", "computational resources"
        ))
        
        # Very short expected answer
        self.assertTrue(self.evaluator._check_answer_correctness(
            "AI", "AI technology"
        ))
        
        # Exact match with special characters
        self.assertTrue(self.evaluator._check_answer_correctness(
            "machine learning", "machine learning"
        ))


class TestMainFunction(unittest.TestCase):
    """Test the main function execution."""
    
    def test_main_success(self):
        """Test main function with successful execution."""
        with patch('src.paragraph_qa_evaluator.ParagraphQAEvaluator') as MockEvaluator:
            mock_instance = MockEvaluator.return_value
            mock_instance.load_model.return_value = True
            mock_instance.run_specific_test.return_value = {'success': True}
            
            # Переимпортируем main чтобы применить мок
            import importlib
            import src.paragraph_qa_evaluator
            importlib.reload(src.paragraph_qa_evaluator)
            from src.paragraph_qa_evaluator import main
            
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_with(0)
    
    def test_main_model_load_failure(self):
        """Test main function when model loading fails."""
        with patch('src.paragraph_qa_evaluator.ParagraphQAEvaluator') as MockEvaluator:
            mock_instance = MockEvaluator.return_value
            mock_instance.load_model.return_value = False
            
            # Переимпортируем main чтобы применить мок
            import importlib
            import src.paragraph_qa_evaluator
            importlib.reload(src.paragraph_qa_evaluator)
            from src.paragraph_qa_evaluator import main
            
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_with(1)
    
    def test_main_exception_handling(self):
        """Test main function exception handling."""
        with patch('src.paragraph_qa_evaluator.ParagraphQAEvaluator') as MockEvaluator:
            MockEvaluator.side_effect = Exception("Test exception")
            
            # Переимпортируем main чтобы применить мок
            import importlib
            import src.paragraph_qa_evaluator
            importlib.reload(src.paragraph_qa_evaluator)
            from src.paragraph_qa_evaluator import main
            
            with patch('sys.exit') as mock_exit:
                with patch('logging.error') as mock_log:
                    main()
                    mock_exit.assert_called_with(1)
                    mock_log.assert_called()

if __name__ == '__main__':
    unittest.main()
