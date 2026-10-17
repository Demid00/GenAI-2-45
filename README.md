# Paragraph QA Evaluator

A specialized evaluation framework for testing question-answering models with multi-paragraph documents. This tool specifically validates a model's ability to locate answers within targeted paragraphs when provided with full document context.

## Key Features

- **Multi-Paragraph Testing**: Create documents with multiple paragraphs to test contextual understanding
- **Targeted Questioning**: Ask questions designed to target specific paragraphs within documents
- **Answer Validation**: Comprehensive validation of both answer location and content correctness
- **Robust Error Handling**: Extensive error checking and validation throughout the pipeline
- **Flexible Configuration**: Support for custom documents, questions, and evaluation parameters
- **Detailed Analytics**: Comprehensive reporting with performance metrics and statistics
- **Batch Testing**: Run multiple test cases across different documents and domains
- **High Test Coverage**: 81% test coverage for core functionality with 39 comprehensive tests

## Problem Statement

This project addresses the specific challenge of evaluating whether QA models can correctly identify answers in designated paragraphs when given complete multi-paragraph documents as context. The evaluation criteria include:

1. Creating a document with 3 distinct paragraphs
2. Asking questions where answers are located in specific paragraphs (e.g., second paragraph)
3. Providing the entire document as context to the model
4. Validating that the model finds the correct answer
5. Reporting confidence scores and localization accuracy

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/paragraph_qa_evaluator.git
cd GenAI-2-45

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `transformers >= 4.20.0` - For pre-trained QA models
- `torch >= 1.9.0` - Deep learning framework
- `pytest >= 7.0.0` - For running tests (optional)

## Quick Start

### Basic Usage
```python
from src.paragraph_qa_evaluator import ParagraphQAEvaluator

# Initialize the evaluator with default model
evaluator = ParagraphQAEvaluator()

# Load the QA model
if evaluator.load_model():
    # Run the standard 3-paragraph test targeting the second paragraph
    result = evaluator.run_specific_test()
    
    # Generate comprehensive report
    evaluator.print_test_report(result)
    
    # Save results
    evaluator.save_results(result, "results/my_test.json")
```

### Expected Output
```
PARAGRAPH QA TEST REPORT
================================================================================

MODEL: AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru
DOCUMENT: 3 paragraphs, 1256 characters

QUESTION: What do deep learning models require for training?
EXPECTED ANSWER: substantial computational resources and large labeled datasets
TARGET PARAGRAPH: 2

MODEL RESULTS:
Answer: 'substantial computational resources and large labeled datasets'
Score: 0.9245
Answer Position: [456: 512]

VALIDATION RESULTS:
✓ Answer from target paragraph: True
✓ Answer content correct: True
✓ Overall test success: True

🎯 TEST PASSED: Model correctly found answer in target paragraph!
```

## Usage Examples

### Example 1: Basic Testing
```python
# examples/basic_usage.py
from src.paragraph_qa_evaluator import ParagraphQAEvaluator

evaluator = ParagraphQAEvaluator()
evaluator.load_model()

# Test with default parameters
result = evaluator.run_specific_test()
evaluator.print_test_report(result)
```

### Example 2: Custom Document
```python
# examples/custom_document_test.py
from src.paragraph_qa_evaluator import ParagraphQAEvaluator

evaluator = ParagraphQAEvaluator()
evaluator.load_model()

# Create custom document
custom_paragraphs = [
    "First paragraph about machine learning basics...",
    "Second paragraph discussing neural networks...", 
    "Third paragraph covering ethical considerations..."
]

evaluator.create_custom_document(custom_paragraphs)

# Test specific questions
result = evaluator.validate_paragraph_answer(
    question="What are neural networks?",
    expected_answer="computational models inspired by human brain",
    target_paragraph_index=1
)
```

### Example 3: Batch Testing
```python
# examples/batch_testing.py
from src.extended_testing import BatchTester

# Run comprehensive tests across multiple domains
tester = BatchTester()
tester.load_model()

# Load test cases from file
test_suite = tester.load_test_cases_from_file("data/test_cases.json")
results = tester.run_batch_test(test_suite)

# Generate detailed report
report = tester.generate_detailed_report(results)
tester.print_batch_report(report)
```

### Example 4: Extended Testing
```python
# Using ExtendedQATesting for multiple test cases
from src.extended_testing import ExtendedQATesting

tester = ExtendedQATesting()
tester.load_model()

test_cases = [
    {
        'question': 'What is AI?',
        'expected_answer': 'artificial intelligence',
        'target_paragraph': 0
    },
    {
        'question': 'What do models require?', 
        'expected_answer': 'computational resources',
        'target_paragraph': 1
    }
]

results = tester.run_multiple_tests(test_cases)
tester.generate_comprehensive_report()
```

## Project Structure

```
paragraph_qa_evaluator/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── paragraph_qa_evaluator.py   # Main evaluation class (81% coverage)
│   └── extended_testing.py         # Extended testing functionality
├── examples/
│   ├── basic_usage.py              # Basic usage examples
│   ├── custom_document_test.py     # Custom document testing  
│   └── batch_testing.py            # Batch testing examples
├── tests/
│   ├── __init__.py                 # Test package initialization
│   ├── test_paragraph_qa.py        # Core functionality tests (33 tests)
│   ├── test_extended_testing.py    # Extended testing tests (6 tests)
│   └── conftest.py                 # Test configuration
├── data/
│   ├── sample_document.json        # Sample test document
│   └── test_cases.json             # Comprehensive test cases
├── results/                        # Output directory for test results
├── requirements.txt                # Project dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Testing & Quality

### Running Tests
```bash
# Run all tests (39 tests total)
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Results
- **39 tests passing** - Comprehensive test suite
- **81% coverage** - Core module `paragraph_qa_evaluator.py`
- **60% overall coverage** - Including extended functionality
- **Fast execution** - Tests complete in under 1 second

### Test Categories
- **Unit Tests**: Basic functionality and initialization
- **Integration Tests**: Mocked model interactions
- **Error Handling**: Invalid inputs and edge cases
- **Validation Tests**: Answer correctness and paragraph detection

## Evaluation Metrics

The framework evaluates models based on:

- **Answer Localization**: Whether the answer comes from the target paragraph
- **Content Accuracy**: Whether the answer content matches expected results  
- **Confidence Score**: Model's confidence in the extracted answer (0.0-1.0)
- **Paragraph-specific Performance**: Accuracy per paragraph location
- **Success Rate**: Overall test success percentage

## Configuration

### Model Selection
```python
# Use different pre-trained models
evaluator = ParagraphQAEvaluator(
    model_name="deepset/roberta-base-squad2"  # Alternative model
)

# Or use the default multilingual model
evaluator = ParagraphQAEvaluator()  # Uses "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
```

### Custom Evaluation Parameters
```python
# Adjust confidence threshold
result = evaluator.validate_paragraph_answer(
    question="Your question",
    expected_answer="Expected answer",
    target_paragraph_index=1
)

# Custom test parameters
result = evaluator.run_specific_test(
    question="Custom question",
    expected_answer="Custom expected answer", 
    target_paragraph=0  # Target first paragraph
)
```

## Error Handling

The framework includes comprehensive error handling for:

- **Model loading failures** - Graceful degradation with clear error messages
- **Invalid paragraph indices** - Validation with descriptive ValueError
- **Missing context or questions** - Early validation with helpful messages
- **File I/O operations** - Safe file handling with fallbacks
- **Network connectivity issues** - Timeout handling and retry logic

All errors are logged with descriptive messages and appropriate fallback behavior.

## Performance

- **Fast model loading** with pre-trained transformers
- **Efficient text processing** with optimized pipelines
- **Minimal memory footprint** for large documents
- **Parallel-ready architecture** for batch processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Development Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/paragraph_qa_evaluator.git
cd paragraph_qa_evaluator
pip install -r requirements.txt

# Run tests to verify setup
python -m pytest tests/ -v
```
