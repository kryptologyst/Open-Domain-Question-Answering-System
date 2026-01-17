#!/usr/bin/env python3
"""
Quick test script to verify the QA system setup

This script tests the basic functionality without requiring heavy model downloads.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from qa_system import QAConfig, SyntheticDatasetGenerator, QuestionAnsweringSystem
        print("✅ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration system."""
    print("⚙️ Testing configuration...")
    
    try:
        from qa_system import QAConfig
        config = QAConfig()
        
        # Test basic config access
        default_model = config.get("model.default_model")
        assert default_model is not None
        
        # Test non-existing key
        non_existing = config.get("non.existing.key", "default")
        assert non_existing == "default"
        
        print("✅ Configuration system working")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_dataset_generator():
    """Test synthetic dataset generation."""
    print("📊 Testing dataset generator...")
    
    try:
        from qa_system import QAConfig, SyntheticDatasetGenerator
        
        config = QAConfig()
        generator = SyntheticDatasetGenerator(config)
        
        # Test generating a small number of samples
        qa_pairs = generator.generate_qa_pairs(3)
        
        assert len(qa_pairs) == 3
        assert all("question" in pair for pair in qa_pairs)
        assert all("context" in pair for pair in qa_pairs)
        assert all("answer" in pair for pair in qa_pairs)
        
        print("✅ Dataset generator working")
        return True
    except Exception as e:
        print(f"❌ Dataset generator error: {e}")
        return False

def test_qa_system_init():
    """Test QA system initialization."""
    print("🤖 Testing QA system initialization...")
    
    try:
        from qa_system import QAConfig, QuestionAnsweringSystem
        
        config = QAConfig()
        qa_system = QuestionAnsweringSystem(config)
        
        # Test basic properties
        assert qa_system.config is not None
        assert qa_system.model_name is not None
        assert qa_system.device is not None
        
        # Test model info without loaded model
        model_info = qa_system.get_model_info()
        assert model_info["status"] == "No model loaded"
        
        print("✅ QA system initialization working")
        return True
    except Exception as e:
        print(f"❌ QA system initialization error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("📁 Testing file structure...")
    
    required_files = [
        "src/qa_system.py",
        "src/cli.py",
        "web_app/streamlit_app.py",
        "config/config.yaml",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "example.py",
        "setup.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Run all tests."""
    print("🚀 Quick Test Suite for QA System")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_config,
        test_dataset_generator,
        test_qa_system_init
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📚 Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run example: python example.py")
        print("  3. Try CLI: python src/cli.py --interactive")
        print("  4. Launch web app: streamlit run web_app/streamlit_app.py")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
