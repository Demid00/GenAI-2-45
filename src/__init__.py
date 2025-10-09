"""
Paragraph QA Evaluator package.

A specialized tool for evaluating question-answering models with multi-paragraph documents.
Tests the model's ability to find answers in specific paragraphs when given full document context.
"""

from .paragraph_qa_evaluator import ParagraphQAEvaluator

__version__ = "1.0.0"
__author__ = "Demid Babikov"
__email__ = "babikov.demid@gmail.com"

__all__ = [
    "ParagraphQAEvaluator",
]
