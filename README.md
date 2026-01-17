# Open Domain Question Answering System

A comprehensive question answering system built with state-of-the-art transformer models and best practices. This system supports multiple models, interactive interfaces, batch processing, and comprehensive evaluation.

## Features

- **Multiple Models**: Support for various transformer models (DistilBERT, BERT, RoBERTa, XLM-RoBERTa)
- **Interactive Interfaces**: Streamlit web app and command-line interface
- **Batch Processing**: Process multiple questions at once from CSV/JSON files
- **Comprehensive Evaluation**: Multiple evaluation metrics and visualization
- **Synthetic Data Generation**: Generate synthetic datasets for testing
- **Configurable**: YAML-based configuration system
- **Well Tested**: Comprehensive test suite with pytest
- **Modern Architecture**: Type hints, docstrings, PEP8 compliance

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kryptologyst/Open-Domain-Question-Answering-System.git
   cd Open-Domain-Question-Answering-System
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system:**
   ```bash
   # Web interface
   streamlit run web_app/streamlit_app.py
   
   # Command line interface
   python src/cli.py --interactive
   
   # Direct usage
   python src/qa_system.py
   ```

### Basic Usage

#### Python API

```python
from src.qa_system import QuestionAnsweringSystem, QAConfig

# Initialize system
config = QAConfig()
qa_system = QuestionAnsweringSystem(config)

# Load model and dataset
qa_system.load_model()
qa_system.load_dataset()

# Answer a question
question = "What is artificial intelligence?"
context = "AI is a branch of computer science..."
answer = qa_system.answer_question(question, context)
print(f"Answer: {answer}")
```

#### Command Line Interface

```bash
# Interactive mode
python src/cli.py --interactive

# Single question
python src/cli.py --question "What is AI?" --context "AI is..."

# Batch processing
python src/cli.py --batch questions.json --output results.json

# Generate sample data
python src/cli.py --generate-sample sample.csv --samples 100
```

#### Web Interface

```bash
streamlit run web_app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
0542_Open_Domain_Question_Answering/
├── src/                          # Source code
│   ├── qa_system.py             # Main QA system
│   └── cli.py                   # Command line interface
├── web_app/                     # Web interfaces
│   └── streamlit_app.py         # Streamlit web app
├── config/                      # Configuration files
│   └── config.yaml              # Main configuration
├── tests/                       # Test suite
│   └── test_qa_system.py       # Comprehensive tests
├── data/                        # Data directory
├── models/                      # Model storage
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔧 Configuration

The system uses YAML configuration files for easy customization. Key configuration options:

```yaml
model:
  default_model: "distilbert-base-cased-distilled-squad"
  max_length: 512
  batch_size: 16

data:
  synthetic:
    num_samples: 1000
    context_length_range: [100, 500]

performance:
  use_gpu: true
  num_workers: 4
  cache_dir: "./cache"
```

## Supported Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| DistilBERT | Small | Fast | Good | Quick prototyping |
| BERT Large | Large | Slow | Excellent | High accuracy needs |
| RoBERTa | Medium | Medium | Very Good | Balanced performance |
| XLM-RoBERTa | Large | Slow | Excellent | Multilingual support |

## Evaluation Metrics

The system provides comprehensive evaluation including:

- **Exact Match**: Percentage of exactly correct answers
- **F1 Score**: Harmonic mean of precision and recall
- **Confidence Score**: Model's confidence in its answers
- **BLEU Score**: N-gram overlap with reference answers
- **ROUGE Score**: Recall-oriented evaluation

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_qa_system.py -v
```

## Performance Optimization

### GPU Acceleration

The system automatically detects and uses GPU when available:

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Batch Processing

For better performance with multiple questions:

```python
# Process multiple questions efficiently
questions = ["Q1", "Q2", "Q3"]
contexts = ["C1", "C2", "C3"]
results = qa_system.batch_answer(questions, contexts)
```

### Model Caching

Models are automatically cached to avoid re-downloading:

```yaml
performance:
  cache_dir: "./cache"  # Custom cache directory
```

## Advanced Features

### Zero-Shot Learning

The system supports zero-shot question answering without fine-tuning:

```python
# Use any model for zero-shot QA
qa_system.load_model("facebook/bart-large-cnn")
answer = qa_system.answer_question(question, context)
```

### Few-Shot Learning

Support for few-shot learning with examples:

```python
# Provide examples for better performance
examples = [
    {"question": "What is AI?", "context": "...", "answer": "..."},
    {"question": "How does ML work?", "context": "...", "answer": "..."}
]
# Implementation would use these examples for better performance
```

### Custom Dataset Integration

```python
# Load custom datasets
from datasets import load_dataset

# Load SQuAD dataset
squad_dataset = load_dataset("squad")
qa_system.dataset = squad_dataset["train"]
```

## Web Interface Features

The Streamlit web interface provides:

- **Single Question Mode**: Ask individual questions with custom contexts
- **Batch Processing**: Upload CSV/JSON files for batch processing
- **Model Evaluation**: Run comprehensive evaluations with visualizations
- **Dataset Explorer**: Explore and analyze datasets
- **Interactive Configuration**: Change models and settings on the fly

## API Reference

### QuestionAnsweringSystem

Main class for the QA system.

#### Methods

- `load_model(model_name=None)`: Load a transformer model
- `load_dataset(dataset_name=None)`: Load or generate a dataset
- `answer_question(question, context, return_confidence=False)`: Answer a single question
- `batch_answer(questions, contexts)`: Answer multiple questions
- `evaluate_model(test_data=None)`: Evaluate model performance
- `get_model_info()`: Get information about loaded model

### QAConfig

Configuration management class.

#### Methods

- `get(key, default=None)`: Get configuration value using dot notation
- `_load_config()`: Load configuration from YAML file

### SyntheticDatasetGenerator

Generate synthetic datasets for testing.

#### Methods

- `generate_qa_pairs(num_samples)`: Generate QA pairs
- `create_dataset()`: Create Hugging Face dataset
- `generate_context(subject)`: Generate context for a subject

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in config
   model:
     batch_size: 8  # or smaller
   ```

2. **Model Download Issues**
   ```python
   # Use custom cache directory
   performance:
     cache_dir: "/path/to/cache"
   ```

3. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

### Performance Issues

1. **Slow Model Loading**
   - Use smaller models for development
   - Enable model caching
   - Use GPU acceleration

2. **Memory Issues**
   - Reduce batch size
   - Use gradient checkpointing
   - Process data in smaller chunks

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Streamlit](https://streamlit.io/) for the web interface framework
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The open-source community for various contributions

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - BERT paper
- [SQuAD: 100,000+ Questions for Machine Reading Comprehension](https://arxiv.org/abs/1606.05250) - SQuAD dataset
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)

## Future Enhancements

- [ ] Support for more model architectures (GPT, T5, etc.)
- [ ] Real-time fine-tuning capabilities
- [ ] Advanced retrieval-augmented generation (RAG)
- [ ] Multi-modal question answering (text + images)
- [ ] API server with FastAPI
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Advanced visualization and explainability features
# Open-Domain-Question-Answering-System
