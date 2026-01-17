#!/usr/bin/env python3
"""
Command Line Interface for Open Domain Question Answering System

A modern CLI tool for interacting with the QA system from the command line.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qa_system import QuestionAnsweringSystem, QAConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class QACLI:
    """Command Line Interface for the QA System."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.qa_system = None
        self.config = None
    
    def initialize_system(self, model_name: Optional[str] = None):
        """Initialize the QA system."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Initializing QA system...", total=None)
                
                self.config = QAConfig()
                self.qa_system = QuestionAnsweringSystem(self.config)
                
                if model_name:
                    self.qa_system.load_model(model_name)
                else:
                    self.qa_system.load_model()
                
                self.qa_system.load_dataset()
                
                progress.update(task, description="QA system initialized!")
            
            console.print("[green]✓ QA system initialized successfully![/green]")
            
        except Exception as e:
            console.print(f"[red]Error initializing QA system: {str(e)}[/red]")
            sys.exit(1)
    
    def interactive_mode(self):
        """Run interactive question answering mode."""
        console.print(Panel.fit(
            "[bold blue]Interactive Question Answering Mode[/bold blue]\n"
            "Type 'quit' or 'exit' to stop. Type 'help' for commands.",
            title="🤖 QA System"
        ))
        
        while True:
            try:
                # Get user input
                user_input = console.input("\n[bold cyan]Enter question (or command):[/bold cyan] ")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'model_info':
                    self.show_model_info()
                    continue
                
                if user_input.lower() == 'evaluate':
                    self.run_evaluation()
                    continue
                
                if user_input.lower().startswith('context:'):
                    # Multi-line context input
                    context = user_input[8:].strip()
                    console.print("[yellow]Enter your question:[/yellow]")
                    question = console.input()
                    
                    if question and context:
                        self.answer_question(question, context)
                    continue
                
                # Default: use built-in context
                if user_input:
                    self.answer_question(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    def answer_question(self, question: str, context: Optional[str] = None):
        """Answer a single question."""
        if not self.qa_system:
            console.print("[red]QA system not initialized![/red]")
            return
        
        # Default context if none provided
        if not context:
            context = """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            machines capable of intelligent behavior. AI systems can learn, reason, and make 
            decisions similar to humans. The field encompasses various subfields including 
            machine learning, natural language processing, computer vision, and robotics. 
            AI has applications in healthcare, finance, transportation, and many other sectors.
            """
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing question...", total=None)
                
                result = self.qa_system.answer_question(
                    question, context, return_confidence=True
                )
                
                progress.update(task, description="Question processed!")
            
            # Display result
            console.print(f"\n[bold]Question:[/bold] {question}")
            console.print(f"[bold green]Answer:[/bold green] {result['answer']}")
            console.print(f"[bold blue]Confidence:[/bold blue] {result['confidence']:.3f}")
            console.print(f"[bold blue]Position:[/bold blue] {result['start']}-{result['end']}")
            
        except Exception as e:
            console.print(f"[red]Error answering question: {str(e)}[/red]")
    
    def batch_process(self, input_file: str, output_file: Optional[str] = None):
        """Process a batch of questions from a file."""
        if not self.qa_system:
            console.print("[red]QA system not initialized![/red]")
            return
        
        try:
            # Read input file
            input_path = Path(input_file)
            if not input_path.exists():
                console.print(f"[red]Input file not found: {input_file}[/red]")
                return
            
            # Determine file format
            if input_path.suffix.lower() == '.json':
                with open(input_path, 'r') as f:
                    data = json.load(f)
            elif input_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(input_path)
                data = df.to_dict('records')
            else:
                console.print("[red]Unsupported file format. Use JSON or CSV.[/red]")
                return
            
            # Process questions
            results = []
            with Progress(console=console) as progress:
                task = progress.add_task("Processing questions...", total=len(data))
                
                for item in data:
                    if 'question' in item and 'context' in item:
                        result = self.qa_system.answer_question(
                            item['question'], item['context'], return_confidence=True
                        )
                        results.append({
                            'question': item['question'],
                            'context': item['context'],
                            'answer': result['answer'],
                            'confidence': result['confidence'],
                            'start': result['start'],
                            'end': result['end']
                        })
                    else:
                        console.print(f"[yellow]Skipping item missing 'question' or 'context': {item}[/yellow]")
                    
                    progress.advance(task)
            
            # Save results
            if output_file:
                output_path = Path(output_file)
                if output_path.suffix.lower() == '.json':
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)
                elif output_path.suffix.lower() == '.csv':
                    import pandas as pd
                    df = pd.DataFrame(results)
                    df.to_csv(output_path, index=False)
                else:
                    console.print("[red]Unsupported output format. Use JSON or CSV.[/red]")
                    return
                
                console.print(f"[green]Results saved to: {output_file}[/green]")
            else:
                # Display results
                table = Table(title="Batch Processing Results")
                table.add_column("Question", style="cyan")
                table.add_column("Answer", style="green")
                table.add_column("Confidence", style="blue")
                
                for result in results[:10]:  # Show first 10 results
                    table.add_row(
                        result['question'][:50] + "..." if len(result['question']) > 50 else result['question'],
                        result['answer'][:50] + "..." if len(result['answer']) > 50 else result['answer'],
                        f"{result['confidence']:.3f}"
                    )
                
                console.print(table)
                
                if len(results) > 10:
                    console.print(f"[yellow]Showing first 10 of {len(results)} results.[/yellow]")
        
        except Exception as e:
            console.print(f"[red]Error processing batch: {str(e)}[/red]")
    
    def run_evaluation(self):
        """Run model evaluation."""
        if not self.qa_system:
            console.print("[red]QA system not initialized![/red]")
            return
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Evaluating model...", total=None)
                
                metrics = self.qa_system.evaluate_model()
                
                progress.update(task, description="Evaluation complete!")
            
            # Display metrics
            table = Table(title="Model Evaluation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    table.add_row(key.replace('_', ' ').title(), f"{value:.3f}")
                else:
                    table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error during evaluation: {str(e)}[/red]")
    
    def show_model_info(self):
        """Display model information."""
        if not self.qa_system:
            console.print("[red]QA system not initialized![/red]")
            return
        
        try:
            model_info = self.qa_system.get_model_info()
            
            table = Table(title="Model Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in model_info.items():
                table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error getting model info: {str(e)}[/red]")
    
    def show_help(self):
        """Display help information."""
        help_text = """
[bold]Available Commands:[/bold]

[cyan]help[/cyan] - Show this help message
[cyan]model_info[/cyan] - Display model information
[cyan]evaluate[/cyan] - Run model evaluation
[cyan]context: <text>[/cyan] - Provide custom context for questions
[cyan]quit/exit/q[/cyan] - Exit the program

[bold]Examples:[/bold]
- Ask a question: "What is artificial intelligence?"
- Use custom context: "context: Machine learning is a subset of AI..."
- Then ask: "What is machine learning?"

[bold]Batch Processing:[/bold]
Use the --batch flag to process multiple questions from a file.
Supported formats: JSON, CSV
        """
        
        console.print(Panel.fit(help_text, title="Help"))
    
    def generate_sample_data(self, output_file: str, num_samples: int = 10):
        """Generate sample data for testing."""
        try:
            from qa_system import SyntheticDatasetGenerator
            
            generator = SyntheticDatasetGenerator(self.config)
            qa_pairs = generator.generate_qa_pairs(num_samples)
            
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(qa_pairs, f, indent=2)
            elif output_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.DataFrame(qa_pairs)
                df.to_csv(output_path, index=False)
            else:
                console.print("[red]Unsupported output format. Use JSON or CSV.[/red]")
                return
            
            console.print(f"[green]Sample data generated: {output_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error generating sample data: {str(e)}[/red]")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Open Domain Question Answering System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive
  %(prog)s --question "What is AI?" --context "AI is..."
  %(prog)s --batch input.json --output results.json
  %(prog)s --generate-sample sample.csv --samples 20
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    mode_group.add_argument(
        '--question', '-q',
        type=str,
        help='Ask a single question'
    )
    mode_group.add_argument(
        '--batch', '-b',
        type=str,
        help='Process batch of questions from file'
    )
    mode_group.add_argument(
        '--generate-sample', '-g',
        type=str,
        help='Generate sample data file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--context', '-c',
        type=str,
        help='Context for the question'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for batch processing'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model name to use'
    )
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=10,
        help='Number of samples to generate (default: 10)'
    )
    parser.add_argument(
        '--evaluate', '-e',
        action='store_true',
        help='Run evaluation after initialization'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = QACLI()
    
    # Handle different modes
    if args.interactive:
        cli.initialize_system(args.model)
        if args.evaluate:
            cli.run_evaluation()
        cli.interactive_mode()
    
    elif args.question:
        cli.initialize_system(args.model)
        cli.answer_question(args.question, args.context)
        if args.evaluate:
            cli.run_evaluation()
    
    elif args.batch:
        cli.initialize_system(args.model)
        cli.batch_process(args.batch, args.output)
        if args.evaluate:
            cli.run_evaluation()
    
    elif args.generate_sample:
        cli.config = QAConfig()
        cli.generate_sample_data(args.generate_sample, args.samples)


if __name__ == "__main__":
    main()
