#!/usr/bin/env python3
"""
Project 542: Open Domain Question Answering - Legacy Demo

This is the original simple implementation. For the modern, comprehensive system,
please use the files in the src/ directory or run:

- python example.py (simple example)
- python src/cli.py --interactive (command line interface)
- streamlit run web_app/streamlit_app.py (web interface)

The modern system includes:
- Multiple model support (DistilBERT, BERT, RoBERTa, XLM-RoBERTa)
- Interactive web interface with Streamlit
- Command line interface
- Batch processing capabilities
- Comprehensive evaluation metrics
- Synthetic dataset generation
- Type hints, docstrings, and modern Python practices
- Comprehensive test suite
- YAML configuration system
"""

from transformers import pipeline

def main():
    """Run the original simple QA demo."""
    print("Project 542: Open Domain Question Answering - Legacy Demo")
    print("=" * 60)
    print("Note: This is the original simple implementation.")
    print("For the modern system, run: python example.py")
    print("=" * 60)
    
    try:
        # 1. Load the pre-trained question answering model
        print("Loading question answering model...")
        qa_pipeline = pipeline("question-answering")
        
        # 2. Provide a context and a question
        context = """
        Open domain question answering systems are designed to answer any question, regardless of the subject matter. 
        They achieve this by retrieving relevant information from a large corpus of unstructured data. These systems use 
        techniques such as natural language processing (NLP) and machine learning to provide accurate and relevant answers.
        """
        question = "What are open domain question answering systems?"
        
        print(f"Question: {question}")
        print(f"Context: {context.strip()}")
        
        # 3. Use the pipeline to get the answer from the context
        print("\nProcessing question...")
        result = qa_pipeline(question=question, context=context)
        
        # 4. Display the result
        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['score']:.3f}")
        print(f"Answer Position: {result['start']}-{result['end']}")
        
        print("Legacy demo completed!")
        print("Try the modern system:")
        print("  - python example.py")
        print("  - python src/cli.py --interactive")
        print("  - streamlit run web_app/streamlit_app.py")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure you have installed transformers: pip install transformers")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
