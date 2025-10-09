# Paragraph QA Evaluator

A specialized tool for evaluating question-answering models with multi-paragraph documents. Tests the model's ability to find answers in specific paragraphs when given full document context.

## Features

- ✅ Create multi-paragraph documents for testing
- ✅ Ask questions targeting specific paragraphs  
- ✅ Validate answer location and correctness
- ✅ Comprehensive error handling and validation
- ✅ Flexible testing with custom documents
- ✅ Detailed reporting and statistics

## Installation

```bash
git clone https://github.com/yourusername/paragraph_qa_evaluator.git
cd paragraph_qa_evaluator
pip install -r requirements.txt
```

## Quick Start

```python
from src.paragraph_qa_evaluator import ParagraphQAEvaluator

# Initialize evaluator
evaluator = ParagraphQAEvaluator()

# Load model
if evaluator.load_model():
    # Run standard 3-paragraph test
    result = evaluator.run_specific_test()
    evaluator.print_test_report(result)
```

## Usage Examples

See examples/ directory for complete usage examples:

- basic_usage.py - Basic functionality

- custom_document_test.py - Custom document testing

- batch_testing.py - Multiple test cases

## Requirements

- Python 3.7+

- transformers >= 4.20.0

- torch >= 1.9.0

## Project Structure
- src/ - Main source code

- examples/ - Usage examples

- tests/ - Unit tests

- data/ - Sample data files

- results/ - Output results

## License
- MIT License
