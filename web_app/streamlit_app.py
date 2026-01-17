"""
Streamlit Web Interface for Open Domain Question Answering System

A modern, interactive web interface for the QA system with support for
multiple models, batch processing, and visualization features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qa_system import QuestionAnsweringSystem, QAConfig, SyntheticDatasetGenerator


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None


def load_qa_system():
    """Load the QA system."""
    try:
        if st.session_state.qa_system is None:
            with st.spinner("Initializing QA system..."):
                st.session_state.config = QAConfig()
                st.session_state.qa_system = QuestionAnsweringSystem(st.session_state.config)
                st.session_state.model_loaded = True
                st.success("QA system initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing QA system: {str(e)}")


def load_model(model_name: str):
    """Load a specific model."""
    try:
        with st.spinner(f"Loading model: {model_name}..."):
            st.session_state.qa_system.load_model(model_name)
            st.session_state.model_loaded = True
            st.success(f"Model {model_name} loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")


def load_dataset():
    """Load or generate dataset."""
    try:
        with st.spinner("Loading dataset..."):
            st.session_state.qa_system.load_dataset()
            st.session_state.dataset_loaded = True
            st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Open Domain Question Answering System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("🤖 Open Domain Question Answering System")
    st.markdown("""
    A modern, interactive question answering system powered by state-of-the-art 
    transformer models. Ask questions and get accurate answers from various contexts.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Selection")
        available_models = {
            "Small (Fast)": "distilbert-base-cased-distilled-squad",
            "Medium (Balanced)": "bert-large-cased-whole-word-masking-finetuned-squad",
            "Large (Accurate)": "deepset/roberta-base-squad2",
            "Multilingual": "deepset/xlm-roberta-base-squad2"
        }
        
        selected_model_name = st.selectbox(
            "Choose a model:",
            options=list(available_models.keys()),
            index=0
        )
        
        selected_model = available_models[selected_model_name]
        
        if st.button("Load Model", type="primary"):
            load_qa_system()
            load_model(selected_model)
        
        # Dataset options
        st.subheader("Dataset Options")
        if st.button("Load/Generate Dataset"):
            load_qa_system()
            load_dataset()
        
        # Display model info
        if st.session_state.model_loaded and st.session_state.qa_system:
            st.subheader("Model Information")
            model_info = st.session_state.qa_system.get_model_info()
            for key, value in model_info.items():
                st.text(f"{key}: {value}")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Single Question", 
        "📊 Batch Processing", 
        "📈 Evaluation", 
        "📚 Dataset Explorer", 
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("Single Question Answering")
        
        if not st.session_state.model_loaded:
            st.warning("Please load a model first using the sidebar.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Input")
                
                # Context input
                context = st.text_area(
                    "Context:",
                    value="""Artificial Intelligence (AI) is a branch of computer science that aims to create 
machines capable of intelligent behavior. AI systems can learn, reason, and make 
decisions similar to humans. The field encompasses various subfields including 
machine learning, natural language processing, computer vision, and robotics. 
AI has applications in healthcare, finance, transportation, and many other sectors.""",
                    height=200,
                    help="Enter the text context from which to answer questions."
                )
                
                # Question input
                question = st.text_input(
                    "Question:",
                    value="What is artificial intelligence?",
                    help="Enter your question here."
                )
                
                # Answer button
                if st.button("Get Answer", type="primary"):
                    if question and context:
                        try:
                            with st.spinner("Processing..."):
                                result = st.session_state.qa_system.answer_question(
                                    question, context, return_confidence=True
                                )
                            
                            with col2:
                                st.subheader("Answer")
                                st.success(f"**Answer:** {result['answer']}")
                                st.info(f"**Confidence:** {result['confidence']:.3f}")
                                st.info(f"**Answer Position:** {result['start']}-{result['end']}")
                                
                                # Highlight answer in context
                                highlighted_context = (
                                    context[:result['start']] + 
                                    f"**{result['answer']}**" + 
                                    context[result['end']:]
                                )
                                st.markdown("**Answer highlighted in context:**")
                                st.markdown(highlighted_context)
                        
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
                    else:
                        st.warning("Please enter both a question and context.")
    
    with tab2:
        st.header("Batch Question Processing")
        
        if not st.session_state.model_loaded:
            st.warning("Please load a model first using the sidebar.")
        else:
            st.subheader("Upload Questions")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload CSV file with questions and contexts",
                type=['csv'],
                help="CSV should have columns: 'question', 'context'"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head())
                    
                    if st.button("Process Batch", type="primary"):
                        with st.spinner("Processing batch..."):
                            questions = df['question'].tolist()
                            contexts = df['context'].tolist()
                            
                            results = st.session_state.qa_system.batch_answer(questions, contexts)
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(results)
                            
                            st.subheader("Results")
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="qa_results.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
            
            # Manual batch input
            st.subheader("Manual Batch Input")
            st.markdown("Enter multiple questions and contexts:")
            
            num_questions = st.number_input("Number of questions:", min_value=1, max_value=10, value=3)
            
            batch_data = []
            for i in range(num_questions):
                with st.expander(f"Question {i+1}"):
                    question = st.text_input(f"Question {i+1}:", key=f"q_{i}")
                    context = st.text_area(f"Context {i+1}:", key=f"c_{i}")
                    if question and context:
                        batch_data.append({"question": question, "context": context})
            
            if batch_data and st.button("Process Manual Batch", type="primary"):
                with st.spinner("Processing..."):
                    questions = [item["question"] for item in batch_data]
                    contexts = [item["context"] for item in batch_data]
                    
                    results = st.session_state.qa_system.batch_answer(questions, contexts)
                    
                    for i, result in enumerate(results):
                        st.write(f"**Question {i+1}:** {result['question']}")
                        st.write(f"**Answer:** {result['answer']}")
                        st.write(f"**Confidence:** {result['confidence']:.3f}")
                        st.divider()
    
    with tab3:
        st.header("Model Evaluation")
        
        if not st.session_state.model_loaded or not st.session_state.dataset_loaded:
            st.warning("Please load both a model and dataset first using the sidebar.")
        else:
            if st.button("Run Evaluation", type="primary"):
                with st.spinner("Evaluating model..."):
                    try:
                        metrics = st.session_state.qa_system.evaluate_model()
                        st.session_state.evaluation_results = metrics
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Avg Confidence", f"{metrics['average_confidence']:.3f}")
                        with col3:
                            st.metric("Total Questions", metrics['total_questions'])
                        with col4:
                            st.metric("Correct Answers", metrics['correct_answers'])
                        
                        # Visualization
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Accuracy', 'Average Confidence'],
                                y=[metrics['accuracy'], metrics['average_confidence']],
                                marker_color=['#1f77b4', '#ff7f0e']
                            )
                        ])
                        fig.update_layout(
                            title="Model Performance Metrics",
                            yaxis_title="Score",
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
    
    with tab4:
        st.header("Dataset Explorer")
        
        if not st.session_state.dataset_loaded:
            st.warning("Please load a dataset first using the sidebar.")
        else:
            dataset = st.session_state.qa_system.dataset
            
            st.subheader("Dataset Overview")
            st.write(f"**Total samples:** {len(dataset)}")
            
            # Display sample data
            st.subheader("Sample Data")
            sample_df = pd.DataFrame(dataset[:10])  # Show first 10 samples
            st.dataframe(sample_df)
            
            # Dataset statistics
            if 'question_type' in dataset.column_names:
                st.subheader("Question Type Distribution")
                question_types = [item['question_type'] for item in dataset]
                type_counts = pd.Series(question_types).value_counts()
                
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Question Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Subject distribution
            if 'subject' in dataset.column_names:
                st.subheader("Subject Distribution")
                subjects = [item['subject'] for item in dataset]
                subject_counts = pd.Series(subjects).value_counts().head(10)
                
                fig = px.bar(
                    x=subject_counts.values,
                    y=subject_counts.index,
                    orientation='h',
                    title="Top 10 Subjects"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("About This System")
        
        st.markdown("""
        ## 🤖 Open Domain Question Answering System
        
        This is a modern, comprehensive question answering system built with state-of-the-art 
        transformer models and best practices.
        
        ### ✨ Features
        
        - **Multiple Models**: Support for various transformer models (DistilBERT, BERT, RoBERTa)
        - **Interactive Interface**: Easy-to-use Streamlit web interface
        - **Batch Processing**: Process multiple questions at once
        - **Model Evaluation**: Comprehensive evaluation metrics
        - **Synthetic Data**: Generate synthetic datasets for testing
        - **Visualization**: Rich visualizations and analytics
        
        ### 🛠️ Technical Stack
        
        - **Transformers**: Hugging Face Transformers library
        - **Models**: DistilBERT, BERT, RoBERTa, XLM-RoBERTa
        - **Web Interface**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly
        - **Configuration**: YAML-based configuration
        
        ### 📚 Usage
        
        1. **Load a Model**: Choose from available models in the sidebar
        2. **Load Dataset**: Generate synthetic data or load real datasets
        3. **Ask Questions**: Use the Single Question tab for individual queries
        4. **Batch Process**: Upload CSV files or manually enter multiple questions
        5. **Evaluate**: Run comprehensive model evaluation
        6. **Explore**: Analyze dataset statistics and distributions
        
        ### 🔧 Configuration
        
        The system uses YAML configuration files for easy customization of:
        - Model parameters
        - Dataset settings
        - Performance options
        - UI preferences
        
        ### 📈 Performance
        
        The system supports:
        - GPU acceleration
        - Batch processing
        - Model caching
        - Comprehensive evaluation metrics
        
        ### 🤝 Contributing
        
        This system is designed to be extensible and modular. You can:
        - Add new models
        - Implement custom evaluation metrics
        - Create new dataset generators
        - Extend the web interface
        
        ### 📄 License
        
        This project is open source and available under the MIT License.
        """)
        
        st.markdown("---")
        st.markdown("**Built with ❤️ using Streamlit and Hugging Face Transformers**")


if __name__ == "__main__":
    main()
