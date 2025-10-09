"""
Example: Batch testing with multiple documents and test cases.

Demonstrates how to run comprehensive batch tests across different document types
and analyze performance metrics.
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paragraph_qa_evaluator import ParagraphQAEvaluator


class BatchTester:
    """Batch testing functionality for comprehensive evaluation."""
    
    def __init__(self, model_name: str = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"):
        self.evaluator = ParagraphQAEvaluator(model_name)
        self.all_results = []
    
    def load_model(self) -> bool:
        """Load the QA model."""
        return self.evaluator.load_model()
    
    def load_test_cases_from_file(self, filepath: str) -> dict:
        """
        Load test cases from JSON file.
        
        Args:
            filepath: Path to JSON file with test cases
            
        Returns:
            dict: Loaded test cases
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading test cases from {filepath}: {e}")
            return {}
    
    def run_batch_test(self, test_suite: dict) -> list:
        """
        Run a batch of tests from test suite definition.
        
        Args:
            test_suite: Dictionary containing test suite definition
            
        Returns:
            list: Results from all test cases
        """
        if not self.evaluator.qa_pipeline:
            print("Model not loaded")
            return []
        
        results = []
        
        for document_name, document_data in test_suite.get('documents', {}).items():
            print(f"\n📄 Testing Document: {document_name}")
            print("-" * 50)
            
            # Create document
            self.evaluator.create_custom_document(document_data['paragraphs'])
            
            # Run test cases for this document
            for i, test_case in enumerate(document_data.get('test_cases', [])):
                print(f"\n  Test {i+1}: {test_case['question']}")
                
                result = self.evaluator.validate_paragraph_answer(
                    question=test_case['question'],
                    expected_answer=test_case['expected_answer'],
                    target_paragraph_index=test_case['target_paragraph']
                )
                
                result['document_name'] = document_name
                result['test_case_id'] = f"{document_name}_{i+1}"
                results.append(result)
                
                # Print quick status
                status = "✅ PASS" if result.get('success') else "❌ FAIL"
                print(f"    {status} | Score: {result['score']:.4f}")
        
        self.all_results.extend(results)
        return results
    
    def generate_detailed_report(self, results: list) -> dict:
        """
        Generate detailed performance report.
        
        Args:
            results: List of test results
            
        Returns:
            dict: Comprehensive report
        """
        if not results:
            return {}
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get('success', False))
        avg_score = sum(r.get('score', 0) for r in results) / total_tests
        
        # Document-level statistics
        doc_stats = {}
        for result in results:
            doc_name = result.get('document_name', 'unknown')
            if doc_name not in doc_stats:
                doc_stats[doc_name] = {
                    'total': 0,
                    'successful': 0,
                    'total_score': 0,
                    'paragraph_stats': {}
                }
            
            doc_stats[doc_name]['total'] += 1
            doc_stats[doc_name]['total_score'] += result.get('score', 0)
            
            if result.get('success'):
                doc_stats[doc_name]['successful'] += 1
            
            # Paragraph-level stats
            para_idx = result.get('target_paragraph_index')
            if para_idx not in doc_stats[doc_name]['paragraph_stats']:
                doc_stats[doc_name]['paragraph_stats'][para_idx] = {
                    'total': 0,
                    'successful': 0
                }
            
            doc_stats[doc_name]['paragraph_stats'][para_idx]['total'] += 1
            if result.get('success'):
                doc_stats[doc_name]['paragraph_stats'][para_idx]['successful'] += 1
        
        # Calculate percentages
        for doc_name in doc_stats:
            doc_stats[doc_name]['success_rate'] = (
                doc_stats[doc_name]['successful'] / doc_stats[doc_name]['total'] * 100
            )
            doc_stats[doc_name]['avg_score'] = (
                doc_stats[doc_name]['total_score'] / doc_stats[doc_name]['total']
            )
            
            for para_idx in doc_stats[doc_name]['paragraph_stats']:
                para_stats = doc_stats[doc_name]['paragraph_stats'][para_idx]
                para_stats['success_rate'] = (
                    para_stats['successful'] / para_stats['total'] * 100
                )
        
        report = {
            'overall': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': (successful_tests / total_tests) * 100,
                'average_score': avg_score
            },
            'document_stats': doc_stats,
            'detailed_results': results
        }
        
        return report
    
    def print_batch_report(self, report: dict) -> None:
        """Print formatted batch test report."""
        if not report:
            return
        
        overall = report['overall']
        
        print("\n" + "="*80)
        print("BATCH TESTING REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Tests: {overall['total_tests']}")
        print(f"   Successful: {overall['successful_tests']}")
        print(f"   Success Rate: {overall['success_rate']:.1f}%")
        print(f"   Average Score: {overall['average_score']:.4f}")
        
        print(f"\n📈 DOCUMENT PERFORMANCE:")
        for doc_name, stats in report['document_stats'].items():
            print(f"\n   📄 {doc_name}:")
            print(f"      Success Rate: {stats['success_rate']:.1f}% "
                  f"({stats['successful']}/{stats['total']})")
            print(f"      Average Score: {stats['avg_score']:.4f}")
            
            print(f"      Paragraph Performance:")
            for para_idx, para_stats in stats['paragraph_stats'].items():
                print(f"        Paragraph {para_idx + 1}: {para_stats['success_rate']:.1f}% "
                      f"({para_stats['successful']}/{para_stats['total']})")
        
        # Failure analysis
        failures = [r for r in report['detailed_results'] if not r.get('success')]
        if failures:
            print(f"\n❌ FAILURE ANALYSIS ({len(failures)} failures):")
            for failure in failures[:5]:  # Show first 5 failures
                reasons = []
                if not failure.get('answer_from_target_paragraph'):
                    reasons.append("wrong paragraph")
                if not failure.get('answer_correct'):
                    reasons.append("incorrect answer")
                
                print(f"   - {failure['question']}")
                print(f"     Reasons: {', '.join(reasons)}")
                print(f"     Score: {failure['score']:.4f}")
    
    def save_batch_report(self, report: dict, filename: str = "batch_test_report.json") -> None:
        """Save batch report to file."""
        try:
            os.makedirs('../results', exist_ok=True)
            filepath = os.path.join('../results', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Report saved to: {filepath}")
        except Exception as e:
            print(f"Error saving report: {e}")


def create_sample_test_suite() -> dict:
    """Create a sample test suite for demonstration."""
    return {
        "documents": {
            "Science_Document": {
                "paragraphs": [
                    "Machine learning algorithms can be broadly categorized into three types: supervised learning, unsupervised learning, and reinforcement learning. Each type has distinct characteristics and applications in solving different kinds of problems.",
                    "Supervised learning requires labeled datasets where each training example is paired with the correct output. Common algorithms include linear regression for continuous values and classification algorithms like support vector machines for categorical outcomes.",
                    "Unsupervised learning discovers patterns in unlabeled data without predefined categories. Clustering algorithms group similar data points, while dimensionality reduction techniques like PCA help visualize high-dimensional data in lower dimensions."
                ],
                "test_cases": [
                    {
                        "question": "What are the three types of machine learning?",
                        "expected_answer": "supervised learning, unsupervised learning, and reinforcement learning",
                        "target_paragraph": 0
                    },
                    {
                        "question": "What does supervised learning require?",
                        "expected_answer": "labeled datasets",
                        "target_paragraph": 1
                    },
                    {
                        "question": "What does unsupervised learning discover?",
                        "expected_answer": "patterns in unlabeled data",
                        "target_paragraph": 2
                    }
                ]
            },
            "History_Document": {
                "paragraphs": [
                    "The Renaissance period marked a cultural rebirth in Europe from the 14th to 17th centuries. This era saw renewed interest in classical learning and values, with significant advancements in art, science, and literature.",
                    "Key figures of the Renaissance include Leonardo da Vinci, who exemplified the 'Renaissance Man' ideal with expertise in multiple fields. Michelangelo's artistic masterpieces and Galileo's scientific discoveries also characterized this transformative period.",
                    "The printing press invention by Johannes Gutenberg around 1440 revolutionized information dissemination. This technology enabled mass production of books, facilitating the spread of new ideas and contributing to the Protestant Reformation."
                ],
                "test_cases": [
                    {
                        "question": "When did the Renaissance occur?",
                        "expected_answer": "from the 14th to 17th centuries",
                        "target_paragraph": 0
                    },
                    {
                        "question": "Who exemplified the Renaissance Man ideal?",
                        "expected_answer": "Leonardo da Vinci",
                        "target_paragraph": 1
                    },
                    {
                        "question": "What did Johannes Gutenberg invent?",
                        "expected_answer": "printing press",
                        "target_paragraph": 2
                    }
                ]
            }
        }
    }


def main():
    """Run batch testing demonstration."""
    print("Batch Testing - Paragraph QA Evaluator")
    print("=" * 50)
    
    # Initialize batch tester
    batch_tester = BatchTester()
    
    # Load model
    if not batch_tester.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Create sample test suite
    test_suite = create_sample_test_suite()
    
    # Run batch tests
    print("Starting batch testing...")
    results = batch_tester.run_batch_test(test_suite)
    
    # Generate and display report
    if results:
        report = batch_tester.generate_detailed_report(results)
        batch_tester.print_batch_report(report)
        batch_tester.save_batch_report(report)
        
        print(f"\n🎉 Batch testing completed!")
        print(f"   Total test cases: {len(results)}")
        print(f"   Overall success rate: {report['overall']['success_rate']:.1f}%")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
