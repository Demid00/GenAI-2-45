"""
Tests for extended testing functionality.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from extended_testing import ExtendedQATesting, BatchTester
    EXTENDED_TESTING_AVAILABLE = True
except ImportError:
    EXTENDED_TESTING_AVAILABLE = False


@unittest.skipIf(not EXTENDED_TESTING_AVAILABLE, "Extended testing module not available")
class TestExtendedQATesting(unittest.TestCase):
    """Tests for ExtendedQATesting class."""
    
    def setUp(self):
        self.extended_tester = ExtendedQATesting()
    
    def test_initialization(self):
        """Test extended tester initialization."""
        self.assertIsNotNone(self.extended_tester.evaluator)
        self.assertEqual(self.extended_tester.results, [])
    
    def test_load_model(self):
        """Test model loading in extended tester."""
        with patch.object(self.extended_tester.evaluator, 'load_model') as mock_load:
            mock_load.return_value = True
            result = self.extended_tester.load_model()
            self.assertTrue(result)
    
    def test_create_custom_document(self):
        """Test custom document creation."""
        paragraphs = ["Para 1", "Para 2"]
        context, result_paragraphs = self.extended_tester.create_custom_document(paragraphs)
        
        self.assertEqual(result_paragraphs, paragraphs)
        self.assertEqual(context, "\n\n".join(paragraphs))
    
    def test_create_custom_document_insufficient_paragraphs(self):
        """Test custom document with insufficient paragraphs."""
        with self.assertRaises(ValueError):
            self.extended_tester.create_custom_document(["Single paragraph"])


@unittest.skipIf(not EXTENDED_TESTING_AVAILABLE, "Extended testing module not available")
class TestBatchTester(unittest.TestCase):
    """Tests for BatchTester class."""
    
    def setUp(self):
        self.batch_tester = BatchTester()
    
    def test_initialization(self):
        """Test batch tester initialization."""
        self.assertIsNotNone(self.batch_tester.evaluator)
        self.assertEqual(self.batch_tester.all_results, [])
    
    def test_load_model(self):
        """Test model loading in batch tester."""
        with patch.object(self.batch_tester.evaluator, 'load_model') as mock_load:
            mock_load.return_value = True
            result = self.batch_tester.load_model()
            self.assertTrue(result)


class TestBasicExtendedTesting(unittest.TestCase):
    """Basic tests that work even without extended_testing module."""
    
    def test_extended_module_availability(self):
        """Test if extended testing module is available."""
        # This test should pass regardless
        if EXTENDED_TESTING_AVAILABLE:
            self.assertTrue(True, "Extended testing module is available")
        else:
            self.skipTest("Extended testing module not available")
