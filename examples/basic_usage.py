"""
Basic usage example for Paragraph QA Evaluator.

Demonstrates the main functionality with different test scenarios.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paragraph_qa_evaluator import ParagraphQAEvaluator


def run_basic_test():
    """Run the basic 3-paragraph test."""
    print("=== BASIC 3-PARAGRAPH TEST ===")
    
    evaluator = ParagraphQAEvaluator()
    
    if evaluator.load_model():
        # Run standard test targeting second paragraph
        result = evaluator.run_specific_test()
        evaluator.print_test_report(result)
        
        return result
    else:
        print("Failed to load model")
        return None


def run_custom_test():
    """Run test with custom parameters."""
    print("\n=== CUSTOM PARAMETERS TEST ===")
    
    evaluator = ParagraphQAEvaluator()
    
    if evaluator.load_model():
        # Create document first
        evaluator.create_three_paragraph_document()
        
        # Test with custom question targeting first paragraph
        custom_question = "What fields does AI encompass?"
        expected_answer = "machine learning, natural language processing, computer vision, and robotics"
        
        result = evaluator.validate_paragraph_answer(
            question=custom_question,
            expected_answer=expected_answer,
            target_paragraph_index=0  # First paragraph
        )
        
        evaluator.print_test_report(result)
        return result
    else:
        print("Failed to load model")
        return None


def main():
    """Run all example tests."""
    print("Paragraph QA Evaluator - Basic Usage Examples")
    print("=" * 50)
    
    # Run basic test
    basic_result = run_basic_test()
    
    # Run custom test  
    custom_result = run_custom_test()
    
    # Summary
    print("\n=== SUMMARY ===")
    tests = [
        ("Basic Test", basic_result),
        ("Custom Test", custom_result)
    ]
    
    for test_name, result in tests:
        if result and result.get('success'):
            status = "PASS"
        else:
            status = "FAIL"
        print(f"{test_name}: {status}")


if __name__ == "__main__":
    main()
