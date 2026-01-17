#!/usr/bin/env python3
"""
Simple example script for the Open Domain Question Answering System

This script demonstrates basic usage of the QA system with a simple example.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qa_system import QuestionAnsweringSystem, QAConfig


def main():
    """Run a simple example of the QA system."""
    print("🤖 Open Domain Question Answering System - Simple Example")
    print("=" * 60)
    
    try:
        # Initialize configuration and system
        print("📋 Initializing QA system...")
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        
        # Load model
        print("🔄 Loading model...")
        qa_system.load_model()
        
        # Load dataset
        print("📊 Loading dataset...")
        qa_system.load_dataset()
        
        print("✅ System ready!")
        print()
        
        # Example questions and contexts
        examples = [
            {
                "question": "What is artificial intelligence?",
                "context": """
                Artificial Intelligence (AI) is a branch of computer science that aims to create 
                machines capable of intelligent behavior. AI systems can learn, reason, and make 
                decisions similar to humans. The field encompasses various subfields including 
                machine learning, natural language processing, computer vision, and robotics.
                """
            },
            {
                "question": "What can AI systems do?",
                "context": """
                AI systems can perform a wide range of tasks including pattern recognition, 
                decision making, language translation, image processing, and autonomous control. 
                They are used in healthcare for medical diagnosis, in finance for fraud detection, 
                in transportation for autonomous vehicles, and in many other sectors.
                """
            },
            {
                "question": "What are the main subfields of AI?",
                "context": """
                The main subfields of artificial intelligence include machine learning, which focuses 
                on algorithms that can learn from data; natural language processing, which deals with 
                human-computer interaction through language; computer vision, which enables machines 
                to interpret visual information; and robotics, which combines AI with mechanical systems.
                """
            }
        ]
        
        # Answer questions
        for i, example in enumerate(examples, 1):
            print(f"📝 Example {i}:")
            print(f"Question: {example['question']}")
            print(f"Context: {example['context'].strip()}")
            
            try:
                # Get answer with confidence
                result = qa_system.answer_question(
                    example['question'], 
                    example['context'], 
                    return_confidence=True
                )
                
                print(f"Answer: {result['answer']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Answer Position: {result['start']}-{result['end']}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
            
            print("-" * 40)
        
        # Show model information
        print("ℹ️ Model Information:")
        model_info = qa_system.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        print()
        print("🎉 Example completed successfully!")
        print()
        print("💡 Next steps:")
        print("  - Try the interactive CLI: python src/cli.py --interactive")
        print("  - Launch the web interface: streamlit run web_app/streamlit_app.py")
        print("  - Run the test suite: pytest tests/ -v")
        
    except Exception as e:
        print(f"❌ Error running example: {str(e)}")
        print("💡 Make sure you have installed all dependencies: pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
