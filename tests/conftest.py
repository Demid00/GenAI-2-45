"""
Pytest configuration and fixtures for Paragraph QA Evaluator tests.
"""

import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def evaluator():
    """Fixture providing ParagraphQAEvaluator instance."""
    from paragraph_qa_evaluator import ParagraphQAEvaluator
    evaluator = ParagraphQAEvaluator()
    evaluator.create_three_paragraph_document()
    return evaluator


@pytest.fixture
def evaluator_with_mock_model():
    """Fixture providing evaluator with mocked model."""
    from unittest.mock import Mock
    from paragraph_qa_evaluator import ParagraphQAEvaluator
    
    evaluator = ParagraphQAEvaluator()
    evaluator.create_three_paragraph_document()
    
    # Mock pipeline
    mock_pipeline = Mock()
    mock_pipeline.return_value = {
        'answer': 'test answer',
        'score': 0.9,
        'start': 100,
        'end': 110
    }
    evaluator.qa_pipeline = mock_pipeline
    
    return evaluator
