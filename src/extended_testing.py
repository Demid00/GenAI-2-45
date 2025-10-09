import json
from typing import List, Dict, Tuple
from paragraph_qa_evaluator import ParagraphQAEvaluator


class ExtendedQATesting:
    """
    Extended testing capabilities for comprehensive QA evaluation.
    """
    
    def __init__(self, model_name: str = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"):
        self.evaluator = ParagraphQAEvaluator(model_name)
        self.results = []
    
    def load_model(self) -> bool:
        """Load the QA model."""
        return self.evaluator.load_model()
    
    def create_custom_document(self, paragraphs: List[str]) -> Tuple[str, List[str]]:
        """
        Create custom multi-paragraph document.
        
        Args:
            paragraphs: List of paragraph texts
            
        Returns:
            Tuple[str, List[str]]: Full document and paragraphs
        """
        if len(paragraphs) < 2:
            raise ValueError("Need at least 2 paragraphs for meaningful testing")
        
        self.evaluator.paragraphs = [p.strip() for p in paragraphs]
        self.evaluator.context = "\n\n".join(self.evaluator.paragraphs)
        return self.evaluator.context, self.evaluator.paragraphs
    
    def run_multiple_tests(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Run multiple test cases.
        
        Args:
            test_cases: List of test case dictionaries with:
                - question: str
                - expected_answer: str  
                - target_paragraph: int (0-based)
                
        Returns:
            List[Dict]: Results for all test cases
        """
        if not self.evaluator.qa_pipeline:
            raise RuntimeError("Model not loaded")
        
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\nRunning test case {i+1}/{len(test_cases)}...")
            
            result = self.evaluator.validate_paragraph_answer(
                question=test_case['question'],
                expected_answer=test_case['expected_answer'],
                target_paragraph_index=test_case['target_paragraph']
            )
            
            result['test_case_id'] = i + 1
            results.append(result)
            
            # Print individual result
            status = "PASS" if result['success'] else "FAIL"
            print(f"  Result: {status} (Score: {result['score']:.4f})")
        
        self.results = results
        return results
    
    def calculate_statistics(self) -> Dict:
        """Calculate overall test statistics."""
        if not self.results:
            return {}
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        avg_score = sum(r['score'] for r in self.results) / total_tests
        
        paragraph_accuracy = {}
        for i in range(len(self.evaluator.paragraphs)):
            para_tests = [r for r in self.results if r['target_paragraph_index'] == i]
            if para_tests:
                para_success = sum(1 for r in para_tests if r['success'])
                paragraph_accuracy[i] = (para_success / len(para_tests)) * 100
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests) * 100,
            'average_score': avg_score,
            'paragraph_accuracy': paragraph_accuracy
        }
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive test report."""
        if not self.results:
            print("No test results available")
            return
        
        stats = self.calculate_statistics()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE QA TEST REPORT")
        print("="*80)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Total tests: {stats['total_tests']}")
        print(f"Successful tests: {stats['successful_tests']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Average score: {stats['average_score']:.4f}")
        
        print(f"\nPARAGRAPH ACCURACY:")
        for para_idx, accuracy in stats['paragraph_accuracy'].items():
            print(f"  Paragraph {para_idx + 1}: {accuracy:.1f}%")
        
        print(f"\nDETAILED RESULTS:")
        for result in self.results:
            status = "PASS" if result['success'] else "FAIL"
            print(f"\nTest {result['test_case_id']} [{status}]:")
            print(f"  Question: {result['question']}")
            print(f"  Answer: '{result['model_answer']}'")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Target Paragraph: {result['target_paragraph_index'] + 1}")
            print(f"  From Target: {result['answer_from_target_paragraph']}")
            print(f"  Correct: {result['answer_correct']}")


# Example usage
if __name__ == "__main__":
    # Create extended tester
    tester = ExtendedQATesting()
    
    if tester.load_model():
        # Create custom document
        paragraphs = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Supervised learning requires labeled training data where each example is paired with the correct output.",
            "Unsupervised learning finds patterns in unlabeled data without predefined categories or labels."
        ]
        
        tester.create_custom_document(paragraphs)
        
        # Define test cases
        test_cases = [
            {
                'question': "What does supervised learning require?",
                'expected_answer': "labeled training data",
                'target_paragraph': 1  # Second paragraph
            },
            {
                'question': "What is machine learning?",
                'expected_answer': "subset of artificial intelligence",
                'target_paragraph': 0  # First paragraph
            }
        ]
        
        # Run tests
        tester.run_multiple_tests(test_cases)
        
        # Generate report
        tester.generate_comprehensive_report()
