"""
Main module for paragraph-based QA evaluation.

This module provides functionality to test question-answering models
with multi-paragraph documents and validate answer locations.
"""

import logging
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any


class ParagraphQAEvaluator:
    """
    Evaluator for testing QA models with multi-paragraph documents.
    
    Specifically designed to test if model can find answers in specific paragraphs
    when given the entire document as context.
    
    Attributes:
        model_name (str): Name of the pre-trained model
        context (str): Full document text
        paragraphs (List[str]): List of individual paragraphs
        qa_pipeline: Loaded QA model pipeline
        logger: Configured logger instance
    """
    
    def __init__(self, model_name: str = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"):
        """
        Initialize the QA evaluator.
        
        Args:
            model_name: HuggingFace model name for question answering
        """
        self.model_name = model_name
        self.context = ""
        self.paragraphs = []
        self.qa_pipeline = None
        self._initialize_logging()
    
    def _initialize_logging(self) -> None:
        """Initialize logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> bool:
        """
        Load the QA model pipeline.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Raises:
            ImportError: If required packages are not installed
        """
        try:
            from transformers import pipeline
        except ImportError as e:
            self.logger.error("Transformers package not installed. Please install: pip install transformers")
            return False
        
        try:
            self.logger.info(f"Loading QA model: {self.model_name}")
            self.qa_pipeline = pipeline(
                "question-answering", 
                model=self.model_name,
                tokenizer=self.model_name
            )
            self.logger.info("Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def create_three_paragraph_document(self) -> Tuple[str, List[str]]:
        """
        Create a sample document with 3 paragraphs for testing.
        
        Returns:
            Tuple[str, List[str]]: Full document text and list of individual paragraphs
            
        Example:
            >>> evaluator = ParagraphQAEvaluator()
            >>> context, paragraphs = evaluator.create_three_paragraph_document()
            >>> len(paragraphs)
            3
        """
        paragraph1 = """
        Artificial Intelligence (AI) has become one of the most transformative technologies 
        of the 21st century. It is revolutionizing industries ranging from healthcare to 
        finance, and its impact continues to grow exponentially. The field of AI encompasses 
        machine learning, natural language processing, computer vision, and robotics.
        """
        
        paragraph2 = """
        Deep learning models, particularly neural networks with multiple hidden layers, 
        have achieved remarkable success in complex tasks. These models require substantial 
        computational resources and large labeled datasets for effective training. 
        The training process involves optimizing millions of parameters through backpropagation 
        and gradient descent algorithms to minimize prediction errors.
        """
        
        paragraph3 = """
        The ethical implications of artificial intelligence are increasingly important 
        as these systems become more pervasive. Key concerns include algorithmic bias, 
        data privacy, transparency in decision-making, and the potential impact on employment. 
        Researchers and policymakers are working to establish frameworks for responsible AI 
        development and deployment that prioritize fairness and accountability.
        """
        
        self.paragraphs = [p.strip() for p in [paragraph1, paragraph2, paragraph3]]
        self.context = "\n\n".join(self.paragraphs)
        
        self.logger.info(f"Created document with {len(self.paragraphs)} paragraphs")
        self.logger.info(f"Total document length: {len(self.context)} characters")
        
        return self.context, self.paragraphs
    
    def create_custom_document(self, paragraphs: List[str]) -> Tuple[str, List[str]]:
        """
        Create a custom document from provided paragraphs.
        
        Args:
            paragraphs: List of paragraph texts to form the document
            
        Returns:
            Tuple[str, List[str]]: Full document text and list of paragraphs
            
        Raises:
            ValueError: If paragraphs list is empty
            
        Example:
            >>> custom_paragraphs = ["First paragraph.", "Second paragraph."]
            >>> context, paragraphs = evaluator.create_custom_document(custom_paragraphs)
        """
        if not paragraphs:
            raise ValueError("Paragraphs list cannot be empty")
        
        self.paragraphs = [p.strip() for p in paragraphs if p.strip()]
        self.context = "\n\n".join(self.paragraphs)
        
        self.logger.info(f"Created custom document with {len(self.paragraphs)} paragraphs")
        self.logger.info(f"Total document length: {len(self.context)} characters")
        
        return self.context, self.paragraphs
    
    def validate_paragraph_answer(self, question: str, expected_answer: str, 
                                target_paragraph_index: int) -> Dict[str, Any]:
        """
        Ask a question and validate if answer comes from the target paragraph.
        
        Args:
            question: Question to ask the model
            expected_answer: The expected correct answer
            target_paragraph_index: Which paragraph should contain the answer (0-based index)
            
        Returns:
            Dict: Evaluation results including score and validation metrics
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If context not set or invalid paragraph index
            
        Example:
            >>> result = evaluator.validate_paragraph_answer(
            ...     question="What do models require?",
            ...     expected_answer="computational resources",
            ...     target_paragraph_index=1
            ... )
        """
        # First validate input parameters
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        if not self.context:
            raise ValueError("Context not set. Call create_three_paragraph_document() or create_custom_document() first.")
        
        if target_paragraph_index < 0 or target_paragraph_index >= len(self.paragraphs):
            raise ValueError(
                f"Invalid paragraph index: {target_paragraph_index}. "
                f"Must be between 0 and {len(self.paragraphs)-1}"
            )
        
        # Then check if model is loaded
        if not self.qa_pipeline:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Ask the question using the full context
            result = self.qa_pipeline(question=question, context=self.context)
            
            # Validate if answer comes from target paragraph
            answer_from_target = self._is_answer_from_paragraph(
                result['answer'], 
                result['start'], 
                result['end'], 
                target_paragraph_index
            )
            
            # Check if answer content matches expected
            answer_correct = self._check_answer_correctness(result['answer'], expected_answer)
            
            evaluation_result = {
                'question': question,
                'model_answer': result['answer'],
                'expected_answer': expected_answer,
                'score': result['score'],
                'answer_position': (result['start'], result['end']),
                'target_paragraph_index': target_paragraph_index,
                'answer_from_target_paragraph': answer_from_target,
                'answer_correct': answer_correct,
                'success': answer_from_target and answer_correct,
                'context_length': len(self.context),
                'paragraph_count': len(self.paragraphs)
            }
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error during question evaluation: {e}")
            return {
                'question': question,
                'model_answer': 'ERROR',
                'expected_answer': expected_answer,
                'score': 0.0,
                'answer_position': (0, 0),
                'target_paragraph_index': target_paragraph_index,
                'answer_from_target_paragraph': False,
                'answer_correct': False,
                'success': False,
                'error': str(e),
                'context_length': len(self.context),
                'paragraph_count': len(self.paragraphs)
            }
    
    def _is_answer_from_paragraph(self, answer: str, start_pos: int, end_pos: int, 
                                paragraph_index: int) -> bool:
        """
        Check if the answer comes from the specified paragraph.
        
        Args:
            answer: The answer text from model
            start_pos: Start position in context
            end_pos: End position in context  
            paragraph_index: Target paragraph index
            
        Returns:
            bool: True if answer is from target paragraph
        """
        # Calculate paragraph boundaries in the full context
        current_position = 0
        for i, paragraph in enumerate(self.paragraphs):
            paragraph_length = len(paragraph)
            # Add 2 for the "\n\n" between paragraphs (except after last paragraph)
            separator_length = 2 if i < len(self.paragraphs) - 1 else 0
            
            if i == paragraph_index:
                paragraph_end = current_position + paragraph_length
                
                # Check if answer position is within paragraph boundaries
                if start_pos >= current_position and end_pos <= paragraph_end:
                    return True
                
                # Additional check: see if answer text appears in target paragraph
                if answer.strip().lower() in paragraph.lower():
                    return True
            
            current_position += paragraph_length + separator_length
        
        return False
    
    def _check_answer_correctness(self, model_answer: str, expected_answer: str) -> bool:
        """
        Check if model answer matches expected answer using flexible matching.
        
        Args:
            model_answer: Answer from the model
            expected_answer: Expected correct answer
            
        Returns:
            bool: True if answers match based on flexible criteria
        """
        if not model_answer or not expected_answer:
            return False
        
        model_clean = model_answer.strip().lower()
        expected_clean = expected_answer.strip().lower()
        
        # Exact match
        if model_clean == expected_clean:
            return True
        
        # Substring match (either direction)
        if expected_clean in model_clean or model_clean in expected_clean:
            return True
        
        # Word overlap - check if key words from expected answer appear in model answer
        expected_words = set(expected_clean.split())
        model_words = set(model_clean.split())
        
        # Require at least 2 matching words or 50% of expected words
        overlap = expected_words.intersection(model_words)
        min_overlap = min(2, len(expected_words))
        
        return len(overlap) >= min_overlap or len(overlap) >= len(expected_words) * 0.5
    
    def run_specific_test(self, question: str = None, expected_answer: str = None, 
                         target_paragraph: int = 1) -> Dict[str, Any]:
        """
        Run the specific test as described in requirements.
        
        1. Create 3-paragraph document
        2. Ask question targeting specified paragraph  
        3. Use full text as context
        4. Validate correct answer
        5. Return score and results
        
        Args:
            question: Custom question (uses default if None)
            expected_answer: Custom expected answer (uses default if None)  
            target_paragraph: Target paragraph index (0-based, default=1 for second paragraph)
            
        Returns:
            Dict: Complete test results
        """
        self.logger.info("Starting specific paragraph QA test...")
        
        # 1. Create document with 3 paragraphs
        context, paragraphs = self.create_three_paragraph_document()
        
        # 2. Set up question and expected answer
        default_question = "What do deep learning models require for training?"
        default_answer = "substantial computational resources and large labeled datasets"
        
        question = question or default_question
        expected_answer = expected_answer or default_answer
        
        self.logger.info(f"Question: {question}")
        self.logger.info(f"Expected answer: {expected_answer}")
        self.logger.info(f"Target paragraph: {target_paragraph + 1}")
        
        # 3-5. Evaluate and validate
        result = self.validate_paragraph_answer(question, expected_answer, target_paragraph)
        
        return result
    
    def print_test_report(self, result: Dict[str, Any]) -> None:
        """
        Print comprehensive test report.
        
        Args:
            result: Test results dictionary
        """
        print("\n" + "="*80)
        print("PARAGRAPH QA TEST REPORT")
        print("="*80)
        
        print(f"\nMODEL: {self.model_name}")
        print(f"DOCUMENT: {result.get('paragraph_count', 0)} paragraphs, "
              f"{result.get('context_length', 0)} characters")
        
        print(f"\nQUESTION: {result['question']}")
        print(f"EXPECTED ANSWER: {result['expected_answer']}")
        print(f"TARGET PARAGRAPH: {result['target_paragraph_index'] + 1}")
        
        print(f"\nMODEL RESULTS:")
        print(f"Answer: '{result['model_answer']}'")
        print(f"Score: {result['score']:.4f}")
        print(f"Answer Position: {result['answer_position']}")
        
        print(f"\nVALIDATION RESULTS:")
        print(f"✓ Answer from target paragraph: {result['answer_from_target_paragraph']}")
        print(f"✓ Answer content correct: {result['answer_correct']}")
        print(f"✓ Overall test success: {result.get('success', False)}")
        
        if result.get('success', False):
            print(f"\n🎯 TEST PASSED: Model correctly found answer in target paragraph!")
        else:
            print(f"\n❌ TEST FAILED: Model did not meet all criteria")
            
            if not result['answer_from_target_paragraph']:
                print("   - Answer not from target paragraph")
            if not result['answer_correct']:
                print("   - Answer content incorrect")
        
        # Show paragraph overview
        if hasattr(self, 'paragraphs') and self.paragraphs:
            print(f"\nDOCUMENT OVERVIEW:")
            for i, paragraph in enumerate(self.paragraphs):
                status = "✓ TARGET" if i == result['target_paragraph_index'] else "  "
                preview = paragraph[:80].replace('\n', ' ') + "..." if len(paragraph) > 80 else paragraph
                print(f"{status} Paragraph {i+1}: {preview}")
    
    def save_results(self, result: Dict[str, Any], filepath: str = "qa_test_results.json") -> bool:
        """
        Save test results to JSON file.
        
        Args:
            result: Test results dictionary
            filepath: Path to save results
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', 
                       exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return False


def main():
    """
    Main execution function for paragraph QA testing.
    
    Demonstrates the complete workflow:
    1. Create 3-paragraph document
    2. Ask question targeting second paragraph  
    3. Use full text as context
    4. Validate correct answer
    5. Output score and results
    """
    try:
        # Initialize evaluator
        evaluator = ParagraphQAEvaluator()
        
        # Load model with error handling
        if not evaluator.load_model():
            print("❌ Failed to load model. Exiting.")
            sys.exit(1)
        
        # Run the specific test
        test_result = evaluator.run_specific_test()
        
        # Print comprehensive report
        evaluator.print_test_report(test_result)
        
        # Save results
        evaluator.save_results(test_result, "results/test_run.json")
        
        # Exit code based on test success
        if test_result.get('success', False):
            print("\n✅ Test completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️ Test completed with issues")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Critical error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
