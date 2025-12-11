#!/usr/bin/env python3
"""
EEW Transformer Evaluation - Simplified Entry Point

This script provides the simplest possible way to run the full evaluation.
Just run: python run_evaluation.py
"""

import os
import sys
from pathlib import Path

def main():
    """Main entry point."""
    
    print("=" * 80)
    print("EEW TRANSFORMER - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print()
    
    # Check if in correct directory
    if not Path('evaluation_suite.py').exists():
        print("❌ Error: evaluation_suite.py not found!")
        print("Please run this script from the eew_transformer directory")
        sys.exit(1)
    
    # Import after checking directory
    import subprocess
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run EEW Transformer evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default evaluation
  python run_evaluation.py
  
  # With custom sample size
  python run_evaluation.py --samples 1000
  
  # Use GPU if available
  python run_evaluation.py --device cuda
  
  # Skip baseline training (only evaluate EEW_Transformer)
  python run_evaluation.py --skip-baselines
  
  # Use Jupyter notebook instead
  python run_evaluation.py --notebook
        """
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=2000,
        help='Number of samples to use (default: 2000)'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='./results_improved_v3/checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./evaluation_results',
        help='Output directory (default: ./evaluation_results)'
    )
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip baseline training, only evaluate EEW_Transformer'
    )
    parser.add_argument(
        '--notebook',
        action='store_true',
        help='Launch Jupyter notebook instead of running script'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data (no STEAD required)'
    )
    
    args = parser.parse_args()
    
    # Launch Jupyter if requested
    if args.notebook:
        print("🚀 Launching Jupyter notebook...")
        print()
        notebook_path = Path('EEW_Comprehensive_Evaluation.ipynb')
        
        if not notebook_path.exists():
            print("❌ Error: EEW_Comprehensive_Evaluation.ipynb not found!")
            sys.exit(1)
        
        # Try to launch Jupyter
        try:
            subprocess.run([
                'jupyter', 'notebook',
                str(notebook_path)
            ], check=True)
        except FileNotFoundError:
            print("❌ Error: Jupyter not found!")
            print("Install with: pip install jupyter")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error launching Jupyter: {e}")
            sys.exit(1)
    
    else:
        # Run evaluation script
        print(f"Configuration:")
        print(f"  • Samples: {args.samples}")
        print(f"  • Checkpoint: {args.checkpoint}")
        print(f"  • Device: {args.device}")
        print(f"  • Output: {args.output}")
        print(f"  • Skip baselines: {args.skip_baselines}")
        print(f"  • Use synthetic: {args.synthetic}")
        print()
        
        # Build command
        cmd = [
            sys.executable, 'evaluation_suite.py',
            '--max-samples', str(args.samples),
            '--checkpoint', args.checkpoint,
            '--device', args.device,
            '--output', args.output,
        ]
        
        if args.skip_baselines:
            cmd.append('--skip-baselines')
        
        if args.synthetic:
            cmd.append('--use-synthetic')
        
        print("🚀 Starting evaluation...")
        print()
        
        try:
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                print()
                print("=" * 80)
                print("✓ EVALUATION COMPLETE")
                print("=" * 80)
                print()
                print(f"📁 Results saved to: {args.output}")
                print()
                print("📊 Generated files:")
                print("  • comparison_metrics.csv")
                print("  • confusion_matrices.png")
                print("  • roc_curves.png")
                print("  • metrics_comparison.png")
                print("  • speed_comparison.png")
                print("  • model_size_comparison.png")
                print("  • evaluation_report.txt")
                print("  • evaluation_report.md")
                print()
                print("📖 For more info:")
                print("  • Read EVALUATION_GUIDE.md")
                print("  • Read DELIVERY_SUMMARY.md")
                print()
            else:
                print()
                print("❌ Evaluation failed with return code:", result.returncode)
                sys.exit(1)
        
        except KeyboardInterrupt:
            print()
            print("\n⚠️  Evaluation interrupted by user")
            sys.exit(1)
        except Exception as e:
            print()
            print(f"❌ Error: {e}")
            sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
