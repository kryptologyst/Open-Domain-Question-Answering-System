"""
Test suite for the Open Domain Question Answering System

Comprehensive tests covering all major components and functionality.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qa_system import (
    QAConfig, 
    SyntheticDatasetGenerator, 
    QuestionAnsweringSystem
)


class TestQAConfig:
    """Test cases for QAConfig class."""
    
    def test_config_initialization(self):
        """Test config initialization with default values."""
        config = QAConfig()
        assert config.config is not None
        assert isinstance(config.config, dict)
    
    def test_config_get_method(self):
        """Test config get method with dot notation."""
        config = QAConfig()
        
        # Test existing key
        default_model = config.get("model.default_model")
        assert default_model is not None
        
        # Test non-existing key with default
        non_existing = config.get("non.existing.key", "default_value")
        assert non_existing == "default_value"
    
    def test_config_get_method_nested(self):
        """Test config get method with nested keys."""
        config = QAConfig()
        
        # Test nested access
        max_length = config.get("model.max_length")
        assert max_length is not None
        assert isinstance(max_length, int)
    
    def test_config_file_not_found(self):
        """Test behavior when config file is not found."""
        config = QAConfig("non_existing_config.yaml")
        assert config.config is not None
        assert isinstance(config.config, dict)


class TestSyntheticDatasetGenerator:
    """Test cases for SyntheticDatasetGenerator class."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        
        assert generator.config is not None
        assert generator.question_templates is not None
        assert generator.subjects is not None
    
    def test_question_templates_structure(self):
        """Test question templates structure."""
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        
        # Check that all expected question types are present
        expected_types = ["what", "who", "when", "where", "why", "how"]
        for q_type in expected_types:
            assert q_type in generator.question_templates
            assert len(generator.question_templates[q_type]) > 0
    
    def test_subjects_list(self):
        """Test subjects list."""
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        
        assert len(generator.subjects) > 0
        assert all(isinstance(subject, str) for subject in generator.subjects)
    
    def test_generate_context(self):
        """Test context generation."""
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        
        # Test with known subject
        context = generator.generate_context("artificial intelligence")
        assert isinstance(context, str)
        assert len(context) > 0
        
        # Test with unknown subject
        context = generator.generate_context("unknown_subject")
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_generate_qa_pairs(self):
        """Test QA pairs generation."""
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        
        # Generate small number of pairs for testing
        qa_pairs = generator.generate_qa_pairs(5)
        
        assert len(qa_pairs) == 5
        assert all(isinstance(pair, dict) for pair in qa_pairs)
        
        # Check structure of each pair
        for pair in qa_pairs:
            assert "question" in pair
            assert "context" in pair
            assert "answer" in pair
            assert "subject" in pair
            assert "question_type" in pair
            
            assert isinstance(pair["question"], str)
            assert isinstance(pair["context"], str)
            assert isinstance(pair["answer"], str)
            assert isinstance(pair["subject"], str)
            assert isinstance(pair["question_type"], str)
    
    def test_create_dataset(self):
        """Test dataset creation."""
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        
        dataset = generator.create_dataset()
        
        assert dataset is not None
        assert len(dataset) > 0
        
        # Check dataset structure
        assert "question" in dataset.column_names
        assert "context" in dataset.column_names
        assert "answer" in dataset.column_names


class TestQuestionAnsweringSystem:
    """Test cases for QuestionAnsweringSystem class."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        
        assert qa_system.config is not None
        assert qa_system.model_name is not None
        assert qa_system.device is not None
    
    @patch('qa_system.AutoTokenizer')
    @patch('qa_system.AutoModelForQuestionAnswering')
    @patch('qa_system.pipeline')
    def test_load_model(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test model loading."""
        # Mock the transformers components
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        
        # Test model loading
        qa_system.load_model()
        
        # Verify that the components were called
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        mock_pipeline.assert_called_once()
    
    def test_load_dataset_synthetic(self):
        """Test synthetic dataset loading."""
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        
        # Mock the dataset generation
        with patch.object(qa_system, '_generate_synthetic_dataset') as mock_generate:
            mock_dataset = Mock()
            mock_generate.return_value = mock_dataset
            
            qa_system.load_dataset()
            
            assert qa_system.dataset is not None
            mock_generate.assert_called_once()
    
    @patch('qa_system.pipeline')
    def test_answer_question(self, mock_pipeline):
        """Test question answering."""
        # Mock pipeline response
        mock_response = {
            "answer": "Test answer",
            "score": 0.95,
            "start": 10,
            "end": 20
        }
        mock_pipeline.return_value = Mock(return_value=mock_response)
        
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        qa_system.pipeline = mock_pipeline.return_value
        
        # Test question answering
        result = qa_system.answer_question("Test question", "Test context", return_confidence=True)
        
        assert result["answer"] == "Test answer"
        assert result["confidence"] == 0.95
        assert result["start"] == 10
        assert result["end"] == 20
    
    @patch('qa_system.pipeline')
    def test_batch_answer(self, mock_pipeline):
        """Test batch question answering."""
        # Mock pipeline response
        mock_response = {
            "answer": "Test answer",
            "score": 0.95,
            "start": 10,
            "end": 20
        }
        mock_pipeline.return_value = Mock(return_value=mock_response)
        
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        qa_system.pipeline = mock_pipeline.return_value
        
        # Test batch answering
        questions = ["Question 1", "Question 2"]
        contexts = ["Context 1", "Context 2"]
        
        results = qa_system.batch_answer(questions, contexts)
        
        assert len(results) == 2
        for result in results:
            assert "question" in result
            assert "answer" in result
            assert "confidence" in result
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        
        # Test without loaded model
        info = qa_system.get_model_info()
        assert info["status"] == "No model loaded"
        
        # Test with mocked model
        qa_system.model = Mock()
        qa_system.tokenizer = Mock()
        qa_system.tokenizer.vocab_size = 1000
        qa_system.tokenizer.model_max_length = 512
        
        info = qa_system.get_model_info()
        assert "model_name" in info
        assert "device" in info
        assert "vocab_size" in info


class TestIntegration:
    """Integration tests for the complete system."""
    
    @patch('qa_system.AutoTokenizer')
    @patch('qa_system.AutoModelForQuestionAnswering')
    @patch('qa_system.pipeline')
    def test_end_to_end_workflow(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test complete end-to-end workflow."""
        # Mock all components
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Mock pipeline response
        mock_response = {
            "answer": "Artificial Intelligence",
            "score": 0.95,
            "start": 10,
            "end": 30
        }
        mock_pipeline.return_value.return_value = mock_response
        
        # Initialize system
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        
        # Load model and dataset
        qa_system.load_model()
        qa_system.load_dataset()
        
        # Answer a question
        question = "What is AI?"
        context = "Artificial Intelligence is a field of computer science."
        
        result = qa_system.answer_question(question, context, return_confidence=True)
        
        # Verify result
        assert result["answer"] == "Artificial Intelligence"
        assert result["confidence"] == 0.95
    
    def test_config_file_handling(self):
        """Test configuration file handling."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model:
  default_model: "test-model"
  max_length: 256
data:
  synthetic:
    num_samples: 100
            """)
            temp_config = f.name
        
        try:
            # Test loading from file
            config = QAConfig(temp_config)
            
            assert config.get("model.default_model") == "test-model"
            assert config.get("model.max_length") == 256
            assert config.get("data.synthetic.num_samples") == 100
            
        finally:
            # Clean up
            os.unlink(temp_config)


# Fixtures for pytest
@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return QAConfig()


@pytest.fixture
def sample_generator(sample_config):
    """Provide a sample dataset generator for testing."""
    return SyntheticDatasetGenerator(sample_config)


@pytest.fixture
def sample_qa_system(sample_config):
    """Provide a sample QA system for testing."""
    return QuestionAnsweringSystem(sample_config)


# Performance tests
class TestPerformance:
    """Performance-related tests."""
    
    def test_dataset_generation_performance(self):
        """Test dataset generation performance."""
        import time
        
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        
        start_time = time.time()
        qa_pairs = generator.generate_qa_pairs(100)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust as needed)
        assert end_time - start_time < 10.0
        assert len(qa_pairs) == 100
    
    def test_memory_usage(self):
        """Test memory usage during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        dataset = generator.create_dataset()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 100 * 1024 * 1024  # 100MB


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
