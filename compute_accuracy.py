"""
Compute accuracy for model evaluation JSON files.
Computes accuracy per task and overall.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict


def extract_answer(response: str) -> str | None:
    """Extract the answer from model response, looking for \\boxed{X} and answer phrases."""
    # Try to find \boxed{X} pattern (handles various formats)
    patterns = [
        r'\$\\boxed\{<option>([A-Fa-f])>\}\$',  # $\boxed{<option>B>}$
        r'\\boxed\{<option>([A-Fa-f])>\}',     # \boxed{<option>B>}
        r'\$\\boxed\{<([A-Fa-f])>\}\$',         # $\boxed{<A>}$ placeholder
        r'\\boxed\{<([A-Fa-f])>\}',             # \boxed{<A>} placeholder
        r'\$\\boxed\{([A-Za-z])\}\$',           # $\boxed{A}$
        r'\\boxed\{([A-Za-z])\}',               # \boxed{A}
        r'\$\\boxed\{([A-Za-z])\s*\}\$',        # $\boxed{A }$
        r'\$\\boxed\{([A-Za-z])[.\s].*?\}\$',   # $\boxed{A. a6}$ or $\boxed{A. Left}$
        r'\\boxed\{([A-Za-z])[.\s].*?\}',       # \boxed{A. a6} without $ signs
        r'\$\\boxed\{\\text\{([A-Za-z])\}\}\$', # $\boxed{\text{A}}$
        r'\\boxed\{\\text\{([A-Za-z])\}\}',     # \boxed{\text{A}}
        # "the answer is X" / "answer is X"
        r'(?:the\s+)?answer\s+is\s+([A-Fa-f])\b',
        # "Answer: A" / "answer: A" / "Answer : A"
        r'answer\s*:\s*([A-Fa-f])\b',
        # "Answer is A" (capital A in Answer)
        r'Answer\s+is\s+([A-Fa-f])\b',
        # "I choose A" / "I select A"
        r'I\s+(?:choose|select)\s+([A-Fa-f])\b',
        # "Option A" / "choice A" at end or before punctuation
        r'(?:option|choice)\s+([A-Fa-f])\b',
        # "So the answer is A" / "Therefore, the answer is B"
        r'(?:so|therefore)[,\s]+(?:the\s+)?answer\s+is\s+([A-Fa-f])\b',
        # "(B) Middle." / "(A) Left." / "(C) option text"
        r'\(([A-Fa-f])\)',
    ]
    
    # Molmo2-SGCoT: "Answer: left/middle/right" -> A/B/C
    position_match = re.search(r'Answer:\s*(left|middle|right)', response, re.IGNORECASE)
    if position_match:
        position_map = {'left': 'A', 'middle': 'B', 'right': 'C'}
        return position_map[position_match.group(1).lower()]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()
    
    # Fallback: if response is just a single letter A/B/C/D/E/F (with optional whitespace/punctuation)
    stripped = response.strip()
    if len(stripped) == 1 and stripped.upper() in 'ABCDEF':
        return stripped.upper()
    # Handle "A." or "B." or "C." with optional punctuation
    if len(stripped) == 2 and stripped[0].upper() in 'ABCDEF' and stripped[1] in '.,':
        return stripped[0].upper()
    
    return None


def compute_accuracy(json_file: str | Path) -> dict:
    """
    Compute accuracy for a single JSON evaluation file.
    
    Returns:
        dict with per-task accuracy and overall accuracy
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Track correct/total per task
    task_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    overall_total = 0
    
    # Detailed results for debugging
    details = []
    print("\n--------------------------------")
    print("video | GT | Extracted")
    print("--------------------------------")
    
    for item in data:
        task = item.get('task', 'unknown')
        ground_truth = item.get('answer', '').upper()
        response = item.get('response', '')
        video = item.get('video', '')
        
        # Extract predicted answer
        predicted = extract_answer(response)

        # Debug: print ground truth and extracted answer
        pred_str = predicted if predicted else "N/A"
        print(f"{video} | GT: {ground_truth} | Extracted: {pred_str}")

        # Check if correct
        is_correct = predicted == ground_truth if predicted else False
        
        task_stats[task]['total'] += 1
        overall_total += 1
        
        if is_correct:
            task_stats[task]['correct'] += 1
            overall_correct += 1
        
        details.append({
            'video': video,
            'task': task,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': is_correct
        })
    
    # Calculate accuracies
    results = {
        'file': str(json_file),
        'per_task': {},
        'overall': {
            'correct': overall_correct,
            'total': overall_total,
            'accuracy': overall_correct / overall_total if overall_total > 0 else 0
        },
        'details': details
    }
    
    for task, stats in sorted(task_stats.items()):
        results['per_task'][task] = {
            'correct': stats['correct'],
            'total': stats['total'],
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        }
    
    return results


def print_results(results: dict, show_details: bool = False):
    """Print results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"File: {results['file']}")
    print(f"{'='*60}")
    
    # Per-task results
    print("Per-Task Accuracy:")
    print("-" * 40)
    for task, stats in results['per_task'].items():
        acc_pct = stats['accuracy'] * 100
        print(f"  {task:15} : {stats['correct']:3}/{stats['total']:3} ({acc_pct:6.2f}%)")
    
    # Overall results
    print("-" * 40)
    overall = results['overall']
    acc_pct = overall['accuracy'] * 100
    print(f"  {'OVERALL':15} : {overall['correct']:3}/{overall['total']:3} ({acc_pct:6.2f}%)")
    
    # Detailed results (optional)
    if show_details:
        print("\nDetailed Results:")
        print("-" * 60)
        for item in results['details']:
            status = "✓" if item['correct'] else "✗"
            pred = item['predicted'] if item['predicted'] else "N/A"
            print(f"  {status} {item['video']:20} [{item['task']:10}] GT:{item['ground_truth']} Pred:{pred}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute accuracy for model evaluation JSON files')
    parser.add_argument('files', nargs='*', help='JSON files to process (default: output/*.json)')
    parser.add_argument('-d', '--details', action='store_true', help='Show detailed per-item results')
    parser.add_argument('-o', '--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Get files to process
    if args.files:
        json_files = [Path(f) for f in args.files]
    else:
        # Default: look in output directory
        output_dir = Path(__file__).parent / 'results' / 'cup'
        if output_dir.exists():
            json_files = list(output_dir.glob('*.json'))
        else:
            json_files = list(Path(__file__).parent.glob('*_output.json'))
    
    if not json_files:
        print("No JSON files found to process.")
        print("Usage: python compute_accuracy.py [file1.json file2.json ...]")
        sys.exit(1)
    
    all_results = []
    
    for json_file in sorted(json_files):
        if not json_file.exists():
            print(f"Warning: File not found: {json_file}")
            continue
        
        try:
            results = compute_accuracy(json_file)
            print_results(results, show_details=args.details)
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Summary if multiple files
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL FILES")
        print(f"{'='*60}")
        
        for r in all_results:
            file_name = Path(r['file']).name
            overall = r['overall']
            acc_pct = overall['accuracy'] * 100
            print(f"  {file_name:40} : {overall['correct']:3}/{overall['total']:3} ({acc_pct:6.2f}%)")
    
    if args.output:
        clean_results = []
        for r in all_results:
            clean_r = {k: v for k, v in r.items() if k != 'details'}
            clean_results.append(clean_r)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

