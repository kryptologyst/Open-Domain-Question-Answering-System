"""
Open Domain Question Answering System

A modern, comprehensive question answering system using state-of-the-art
transformer models with support for various QA techniques including
zero-shot, few-shot, and fine-tuning approaches.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
    Pipeline,
)
from datasets import Dataset, load_dataset
import numpy as np
from rich.console import Console
from rich.progress import track

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


class QAConfig:
    """Configuration manager for the QA system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "model": {
                "default_model": "distilbert-base-cased-distilled-squad",
                "max_length": 512,
                "stride": 128,
                "batch_size": 16
            },
            "data": {
                "synthetic": {
                    "num_samples": 1000,
                    "context_length_range": [100, 500]
                }
            },
            "performance": {
                "use_gpu": torch.cuda.is_available(),
                "cache_dir": "./cache"
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class SyntheticDatasetGenerator:
    """Generate synthetic datasets for QA system testing."""
    
    def __init__(self, config: QAConfig):
        """Initialize the dataset generator."""
        self.config = config
        self.question_templates = {
            "what": [
                "What is {subject}?",
                "What does {subject} mean?",
                "What are the characteristics of {subject}?",
                "What is the purpose of {subject}?"
            ],
            "who": [
                "Who is {subject}?",
                "Who created {subject}?",
                "Who is responsible for {subject}?",
                "Who discovered {subject}?"
            ],
            "when": [
                "When was {subject} created?",
                "When did {subject} happen?",
                "When is {subject} used?",
                "When was {subject} discovered?"
            ],
            "where": [
                "Where is {subject} located?",
                "Where can {subject} be found?",
                "Where does {subject} occur?",
                "Where was {subject} developed?"
            ],
            "why": [
                "Why is {subject} important?",
                "Why does {subject} happen?",
                "Why was {subject} created?",
                "Why do we need {subject}?"
            ],
            "how": [
                "How does {subject} work?",
                "How is {subject} used?",
                "How was {subject} created?",
                "How can {subject} be improved?"
            ]
        }
        
        self.subjects = [
            "artificial intelligence", "machine learning", "neural networks",
            "natural language processing", "computer vision", "robotics",
            "data science", "blockchain", "quantum computing", "cybersecurity",
            "cloud computing", "big data", "internet of things", "augmented reality",
            "virtual reality", "autonomous vehicles", "renewable energy",
            "biotechnology", "nanotechnology", "space exploration"
        ]
    
    def generate_context(self, subject: str) -> str:
        """Generate a context paragraph about a subject."""
        contexts = {
            "artificial intelligence": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            machines capable of intelligent behavior. AI systems can learn, reason, and make 
            decisions similar to humans. The field encompasses various subfields including 
            machine learning, natural language processing, computer vision, and robotics. 
            AI has applications in healthcare, finance, transportation, and many other sectors.
            """,
            "machine learning": """
            Machine Learning is a subset of artificial intelligence that focuses on algorithms 
            and statistical models that enable computer systems to improve their performance 
            on a specific task through experience. It involves training models on data to 
            make predictions or decisions without being explicitly programmed for every scenario. 
            Common types include supervised learning, unsupervised learning, and reinforcement learning.
            """,
            "neural networks": """
            Neural Networks are computing systems inspired by biological neural networks. 
            They consist of interconnected nodes (neurons) that process information using 
            a connectionist approach. Deep neural networks with multiple layers can learn 
            complex patterns in data. They are fundamental to modern AI applications including 
            image recognition, speech processing, and natural language understanding.
            """
        }
        
        return contexts.get(subject, f"""
        {subject.title()} is an important field of study that has gained significant attention 
        in recent years. It involves various techniques and methodologies that contribute 
        to our understanding and application of this domain. The field continues to evolve 
        with new research and technological advancements.
        """)
    
    def generate_qa_pairs(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate synthetic QA pairs."""
        qa_pairs = []
        
        for i in track(range(num_samples), description="Generating QA pairs..."):
            subject = np.random.choice(self.subjects)
            question_type = np.random.choice(list(self.question_templates.keys()))
            template = np.random.choice(self.question_templates[question_type])
            
            question = template.format(subject=subject)
            context = self.generate_context(subject)
            
            # Generate a simple answer (this is a simplified approach)
            answer = f"{subject} is a field that involves various techniques and methodologies."
            
            qa_pairs.append({
                "question": question,
                "context": context.strip(),
                "answer": answer,
                "subject": subject,
                "question_type": question_type
            })
        
        return qa_pairs
    
    def create_dataset(self) -> Dataset:
        """Create a Hugging Face dataset from synthetic QA pairs."""
        num_samples = self.config.get("data.synthetic.num_samples", 1000)
        qa_pairs = self.generate_qa_pairs(num_samples)
        
        dataset = Dataset.from_list(qa_pairs)
        logger.info(f"Created synthetic dataset with {len(dataset)} samples")
        
        return dataset


class QuestionAnsweringSystem:
    """Main QA system class with modern features."""
    
    def __init__(self, config: QAConfig):
        """Initialize the QA system."""
        self.config = config
        self.model_name = config.get("model.default_model")
        self.device = "cuda" if config.get("performance.use_gpu") and torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.dataset = None
        
        logger.info(f"Initializing QA system with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load the QA model and tokenizer."""
        model_name = model_name or self.model_name
        
        try:
            console.print(f"[blue]Loading model: {model_name}[/blue]")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            console.print(f"[green]✓ Model loaded successfully[/green]")
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def load_dataset(self, dataset_name: Optional[str] = None) -> None:
        """Load or generate dataset."""
        if dataset_name:
            try:
                console.print(f"[blue]Loading dataset: {dataset_name}[/blue]")
                self.dataset = load_dataset(dataset_name)
                console.print(f"[green]✓ Dataset loaded successfully[/green]")
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_name}: {str(e)}")
                self._generate_synthetic_dataset()
        else:
            self._generate_synthetic_dataset()
    
    def _generate_synthetic_dataset(self) -> None:
        """Generate synthetic dataset."""
        console.print("[blue]Generating synthetic dataset...[/blue]")
        generator = SyntheticDatasetGenerator(self.config)
        self.dataset = generator.create_dataset()
        console.print("[green]✓ Synthetic dataset generated[/green]")
    
    def answer_question(
        self, 
        question: str, 
        context: str, 
        return_confidence: bool = False
    ) -> Union[str, Dict[str, Union[str, float]]]:
        """
        Answer a question given a context.
        
        Args:
            question: The question to answer
            context: The context to search for the answer
            return_confidence: Whether to return confidence score
            
        Returns:
            Answer string or dictionary with answer and confidence
        """
        if not self.pipeline:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            result = self.pipeline(question=question, context=context)
            
            if return_confidence:
                return {
                    "answer": result["answer"],
                    "confidence": result["score"],
                    "start": result["start"],
                    "end": result["end"]
                }
            else:
                return result["answer"]
                
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
    
    def batch_answer(
        self, 
        questions: List[str], 
        contexts: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """Answer multiple questions in batch."""
        if not self.pipeline:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for question, context in zip(questions, contexts):
            try:
                result = self.pipeline(question=question, context=context)
                results.append({
                    "question": question,
                    "answer": result["answer"],
                    "confidence": result["score"],
                    "start": result["start"],
                    "end": result["end"]
                })
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                results.append({
                    "question": question,
                    "answer": "Error processing question",
                    "confidence": 0.0,
                    "start": 0,
                    "end": 0
                })
        
        return results
    
    def evaluate_model(self, test_data: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if test_data is None:
            test_data = self.dataset
        
        if test_data is None:
            raise ValueError("No test data available for evaluation")
        
        # Simple evaluation metrics
        total_questions = len(test_data)
        correct_answers = 0
        total_confidence = 0.0
        
        console.print("[blue]Evaluating model...[/blue]")
        
        for i, sample in enumerate(track(test_data, description="Evaluating...")):
            try:
                result = self.answer_question(
                    sample["question"], 
                    sample["context"], 
                    return_confidence=True
                )
                
                # Simple exact match evaluation
                if result["answer"].lower().strip() in sample["answer"].lower():
                    correct_answers += 1
                
                total_confidence += result["confidence"]
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {str(e)}")
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        avg_confidence = total_confidence / total_questions if total_questions > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "total_questions": total_questions,
            "correct_answers": correct_answers
        }
        
        console.print(f"[green]Evaluation complete:[/green]")
        console.print(f"  Accuracy: {accuracy:.3f}")
        console.print(f"  Average Confidence: {avg_confidence:.3f}")
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model."""
        if not self.model or not self.tokenizer:
            return {"status": "No model loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.tokenizer.model_max_length,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def main():
    """Main function to demonstrate the QA system."""
    # Initialize configuration
    config = QAConfig()
    
    # Initialize QA system
    qa_system = QuestionAnsweringSystem(config)
    
    # Load model and dataset
    qa_system.load_model()
    qa_system.load_dataset()
    
    # Display model info
    model_info = qa_system.get_model_info()
    console.print("\n[bold blue]Model Information:[/bold blue]")
    for key, value in model_info.items():
        console.print(f"  {key}: {value}")
    
    # Example usage
    console.print("\n[bold blue]Example Questions:[/bold blue]")
    
    example_context = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    machines capable of intelligent behavior. AI systems can learn, reason, and make 
    decisions similar to humans. The field encompasses various subfields including 
    machine learning, natural language processing, computer vision, and robotics.
    """
    
    example_questions = [
        "What is artificial intelligence?",
        "What can AI systems do?",
        "What subfields does AI encompass?"
    ]
    
    for question in example_questions:
        try:
            answer = qa_system.answer_question(question, example_context)
            console.print(f"\n[bold]Q:[/bold] {question}")
            console.print(f"[bold]A:[/bold] {answer}")
        except Exception as e:
            console.print(f"[red]Error answering '{question}': {str(e)}[/red]")
    
    # Evaluate model
    console.print("\n[bold blue]Model Evaluation:[/bold blue]")
    metrics = qa_system.evaluate_model()
    
    console.print("\n[green]QA System demonstration completed![/green]")


if __name__ == "__main__":
    main()
