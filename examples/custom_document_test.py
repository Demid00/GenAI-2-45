"""
Example: Testing with custom documents and questions.

Demonstrates how to use the ParagraphQAEvaluator with custom document content
and multiple test cases targeting different paragraphs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paragraph_qa_evaluator import ParagraphQAEvaluator


def test_custom_science_document():
    """Test with a custom science-themed document."""
    print("=== CUSTOM SCIENCE DOCUMENT TEST ===")
    
    evaluator = ParagraphQAEvaluator()
    
    if not evaluator.load_model():
        print("Failed to load model")
        return None
    
    # Create custom science document
    science_paragraphs = [
        """
        Quantum computing represents a revolutionary approach to computation that leverages 
        quantum mechanical phenomena like superposition and entanglement. Unlike classical 
        computers that use bits, quantum computers use quantum bits or qubits that can 
        exist in multiple states simultaneously.
        """,
        """
        The development of quantum algorithms has shown promising results for specific 
        problems. Shor's algorithm can factor large numbers exponentially faster than 
        classical algorithms, while Grover's algorithm provides quadratic speedup for 
        unstructured search problems. These advancements have significant implications 
        for cryptography and optimization.
        """,
        """
        Major challenges in quantum computing include maintaining qubit coherence, 
        error correction, and scaling systems to practical sizes. Companies like IBM, 
        Google, and Rigetti are developing quantum processors with increasing qubit 
        counts, but fault-tolerant quantum computing remains a long-term goal requiring 
        substantial engineering breakthroughs.
        """
    ]
    
    # Create document
    evaluator.create_custom_document(science_paragraphs)
    
    # Test cases targeting different paragraphs
    test_cases = [
        {
            'question': "What phenomena do quantum computers leverage?",
            'expected_answer': "superposition and entanglement",
            'target_paragraph': 0
        },
        {
            'question': "What is Shor's algorithm used for?",
            'expected_answer': "factor large numbers",
            'target_paragraph': 1
        },
        {
            'question': "What are the major challenges in quantum computing?",
            'expected_answer': "maintaining qubit coherence, error correction, and scaling",
            'target_paragraph': 2
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        result = evaluator.validate_paragraph_answer(
            question=test_case['question'],
            expected_answer=test_case['expected_answer'],
            target_paragraph_index=test_case['target_paragraph']
        )
        evaluator.print_test_report(result)
        results.append(result)
    
    return results


def test_custom_literature_document():
    """Test with a custom literature-themed document."""
    print("\n=== CUSTOM LITERATURE DOCUMENT TEST ===")
    
    evaluator = ParagraphQAEvaluator()
    
    if not evaluator.load_model():
        print("Failed to load model")
        return None
    
    # Create custom literature document
    literature_paragraphs = [
        """
        Russian literature of the 19th century is renowned for its psychological depth 
        and social commentary. Authors like Leo Tolstoy and Fyodor Dostoevsky explored 
        complex philosophical questions through their characters, creating works that 
        continue to influence modern thought and literature worldwide.
        """,
        """
        Tolstoy's "War and Peace" is considered one of the greatest novels ever written. 
        This epic work intertwines the lives of aristocratic families with historical 
        events of the Napoleonic Wars, exploring themes of free will, history, and the 
        search for meaning in life through characters like Pierre Bezukhov and Natasha Rostova.
        """,
        """
        Dostoevsky's contributions to literature include profound explorations of 
        human psychology and existential dilemmas. In "Crime and Punishment," he 
        examines morality through Raskolnikov's internal struggle, while "The Brothers 
        Karamazov" delves into faith, doubt, and family conflicts with unprecedented 
        psychological insight.
        """
    ]
    
    # Create document
    evaluator.create_custom_document(literature_paragraphs)
    
    # Test cases
    test_cases = [
        {
            'question': "Which authors are mentioned in Russian literature?",
            'expected_answer': "Leo Tolstoy and Fyodor Dostoevsky",
            'target_paragraph': 0
        },
        {
            'question': "What themes does War and Peace explore?",
            'expected_answer': "free will, history, and the search for meaning",
            'target_paragraph': 1
        },
        {
            'question': "What does Crime and Punishment examine?",
            'expected_answer': "morality through Raskolnikov's internal struggle",
            'target_paragraph': 2
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        result = evaluator.validate_paragraph_answer(
            question=test_case['question'],
            expected_answer=test_case['expected_answer'],
            target_paragraph_index=test_case['target_paragraph']
        )
        
        # Print simplified result
        status = "PASS" if result.get('success') else "FAIL"
        print(f"Result: {status} | Score: {result['score']:.4f} | "
              f"From Target: {result['answer_from_target_paragraph']} | "
              f"Correct: {result['answer_correct']}")
        
        results.append(result)
    
    return results


def analyze_results(results):
    """Analyze and print summary of test results."""
    if not results:
        return
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get('success', False))
    avg_score = sum(r.get('score', 0) for r in results) / total_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    print(f"Average Score: {avg_score:.4f}")
    
    # Paragraph performance
    paragraph_performance = {}
    for result in results:
        para_idx = result.get('target_paragraph_index', -1)
        if para_idx not in paragraph_performance:
            paragraph_performance[para_idx] = {'total': 0, 'success': 0}
        
        paragraph_performance[para_idx]['total'] += 1
        if result.get('success'):
            paragraph_performance[para_idx]['success'] += 1
    
    print(f"\nParagraph Performance:")
    for para_idx, stats in paragraph_performance.items():
        success_rate = (stats['success'] / stats['total']) * 100
        print(f"  Paragraph {para_idx + 1}: {success_rate:.1f}% "
              f"({stats['success']}/{stats['total']})")


def main():
    """Run all custom document tests."""
    print("Custom Document Testing - Paragraph QA Evaluator")
    print("=" * 50)
    
    # Run science document tests
    science_results = test_custom_science_document()
    
    # Run literature document tests
    literature_results = test_custom_literature_document()
    
    # Combine and analyze all results
    all_results = []
    if science_results:
        all_results.extend(science_results)
    if literature_results:
        all_results.extend(literature_results)
    
    if all_results:
        analyze_results(all_results)
        
        # Save combined results
        evaluator = ParagraphQAEvaluator()
        evaluator.save_results(
            {'test_cases': all_results, 'summary': analyze_results(all_results)}, 
            "../results/custom_tests_results.json"
        )


if __name__ == "__main__":
    main()
