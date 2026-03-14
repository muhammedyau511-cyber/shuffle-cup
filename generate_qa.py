#!/usr/bin/env python3
"""
Generate QA JSON for video LLM evaluation.
Reads all .json files from a directory and generates MCQ questions for each video.
"""

import json
import os
import argparse
import random
from pathlib import Path

# Answer format template appended to each question
# ANSWER_TEMPLATE = "Output your final answer (A, B, or C) in the format: $\\boxed{<option>}$."
ANSWER_TEMPLATE = "Output your final answer (A, B, C, etc.) in the format: $\\boxed{<option>}$."

def get_position_label(position: int, total: int) -> str:
    """Convert position number to human-readable label."""
    if total == 3:
        return ["Left", "Middle", "Right"][position - 1]
    elif total == 2:
        return ["Left", "Right"][position - 1]
    else:
        ordinals = ["First", "Second", "Third", "Fourth", "Fifth", 
                    "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
        if position <= len(ordinals):
            return f"{ordinals[position - 1]} from left"
        return f"Position {position} from left"


def generate_position_choices(total: int) -> list[str]:
    """Generate position choice labels."""
    if total == 3:
        return ["Left", "Middle", "Right"]
    elif total == 2:
        return ["Left", "Right"]
    else:
        ordinals = ["First", "Second", "Third", "Fourth", "Fifth",
                    "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
        return [f"{ordinals[i]} from left" if i < len(ordinals) else f"Position {i+1} from left" 
                for i in range(total)]


def generate_card_position_choices(total: int) -> list[str]:
    """Generate card position choice labels."""
    if total == 3:
        return ["Left", "Middle", "Right"]
    else:
        ordinals = ["First", "Second", "Third", "Fourth", "Fifth",
                    "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
        return [f"{ordinals[i]} card from left" if i < len(ordinals) else f"Card {i+1} from left" 
                for i in range(total)]


def format_mcq(question: str, choices: list[str], answer_idx: int) -> dict:
    """Format question as MCQ with lettered options."""
    letters = "ABCDEFGHIJ"
    options = [f"({letters[i]}) {choice}" for i, choice in enumerate(choices)]
    question_text = f"{question}\n" + "\n".join(options) + f"\n{ANSWER_TEMPLATE}"
    return {
        "question": question_text,
        "answer": letters[answer_idx]
    }


def get_piece_name(piece_type: str) -> str:
    """Convert piece type code to full name."""
    piece_names = {
        'k': 'King',
        'q': 'Queen', 
        'r': 'Rook',
        'b': 'Bishop',
        'n': 'Knight',
        'p': 'Pawn'
    }
    return piece_names.get(piece_type.lower(), piece_type)


def generate_chess_choices(final_positions: list[str], correct_position: str) -> tuple[list[str], int]:
    """Generate chess position choices from all final positions."""
    # Shuffle the choices
    choices = final_positions.copy()
    random.shuffle(choices)
    
    answer_idx = choices.index(correct_position)
    return choices, answer_idx


def get_card_name(card: str) -> str:
    """Convert card notation like '5♠' to 'Five of Spades'."""
    rank_names = {
        'A': 'Ace', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five',
        '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine', '10': 'Ten',
        'J': 'Jack', 'Q': 'Queen', 'K': 'King'
    }
    suit_names = {
        '♠': 'Spades', '♥': 'Hearts', '♦': 'Diamonds', '♣': 'Clubs'
    }
    
    # Parse rank and suit from card string
    if len(card) == 2:
        rank, suit = card[0], card[1]
    elif len(card) == 3:  # For "10♠"
        rank, suit = card[:2], card[2]
    else:
        return card
    
    rank_name = rank_names.get(rank, rank)
    suit_name = suit_names.get(suit, suit)
    return f"{rank_name} of {suit_name}"


def generate_card_qa(item: dict) -> dict:
    """Generate QA for card tracking game."""
    ground_truth = item["ground_truth"]
    num_cards = item["game_settings"]["num_cards"]
    
    # Handle both old format (single object) and new format (array)
    if isinstance(ground_truth, list):
        selected = random.choice(ground_truth)
        card = selected["card"]
        position = selected["position"]
    else:
        # Old format: single object {card, position}
        card = ground_truth["card"]
        position = ground_truth["position"]
    
    card_name = get_card_name(card)
    # question = f"Where is the {card_name} ({card}) at the end of the video?"
    question = f"Where is the {card_name} at the end of the video?"
    choices = generate_card_position_choices(num_cards)
    answer_idx = position - 1
    
    mcq = format_mcq(question, choices, answer_idx)
    
    return {
        "video": item["video"],
        "fps": item["fps"],
        "total_frames": item["total_frames"],
        "resolution": item["resolution"],
        "task": item["task"],
        "ground_truth": item["ground_truth"],
        **mcq
    }


def generate_cup_qa(item: dict) -> dict:
    """Generate QA for cup and ball game."""
    ground_truth = item["ground_truth"]
    num_cups = item["game_settings"]["num_cups"]
    
    # Handle different formats: array of {color, cup} or single {color, cup}
    if isinstance(ground_truth, list):
        # SOT/MOT format: array of {color, cup}
        # Pick a random ball to ask about
        selected = random.choice(ground_truth)
        color = selected["color"]
        cup = selected["cup"]
        question = f"Which cup is the {color} ball under at the end of the video?"
    elif isinstance(ground_truth, dict) and "cup" in ground_truth:
        # New SOT format: {color, cup}
        color = ground_truth.get("color", "")
        cup = ground_truth["cup"]
        if color:
            question = f"Which cup contains the ball at the end of the video?"
        else:
            question = "Which cup is the ball under at the end of the video?"
    else:
        print(f"Unknown ground truth format: {ground_truth}")
        return None

    
    choices = generate_position_choices(num_cups)
    answer_idx = cup - 1
    
    mcq = format_mcq(question, choices, answer_idx)
    
    return {
        "video": item["video"],
        "fps": item["fps"],
        "total_frames": item["total_frames"],
        "resolution": item["resolution"],
        "task": item["task"],
        "ground_truth": item["ground_truth"],
        **mcq
    }


def generate_chess_qa(item: dict) -> dict:
    """Generate QA for chess piece tracking."""
    piece_type = item["game_settings"]["piece_type"]
    ground_truth = item["ground_truth"]
    
    # Pick a random piece to ask about
    selected_piece = random.choice(ground_truth)
    
    original_pos = selected_piece["original_position"]
    final_pos = selected_piece["final_position"]
    
    # Get all final positions as choices
    all_final_positions = [p["final_position"] for p in ground_truth]
    
    piece_name = get_piece_name(piece_type)
    question = f"What is the final position of the {piece_name} originally at {original_pos}?"
    
    choices, answer_idx = generate_chess_choices(all_final_positions, final_pos)
    mcq = format_mcq(question, choices, answer_idx)
    
    return {
        "video": item["video"],
        "fps": item["fps"],
        "total_frames": item["total_frames"],
        "resolution": item["resolution"],
        "task": item["task"],
        "ground_truth": item["ground_truth"],
        **mcq
    }


def generate_pool_qa(item: dict) -> dict:
    """Generate QA for pool ball tracking."""
    num_balls = item["game_settings"]["num_balls"]
    ground_truth = item["ground_truth"]
    
    # Handle both old format (single object) and new format (array)
    if isinstance(ground_truth, list):
        # New format: array of {initial, final}
        # Pick a random ball to ask about
        selected = random.choice(ground_truth)
        initial_pos = selected["initial"]
        final_pos = selected["final"]
    else:
        # Old format: single object {initial, final}
        initial_pos = ground_truth["initial"]
        final_pos = ground_truth["final"]
    
    # Describe the initial position using "ball one", "ball two", etc.
    ball_names = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    ball_desc = f"ball {ball_names[initial_pos - 1]}" if initial_pos <= len(ball_names) else f"ball {initial_pos}"
    
    question = f"Which final position corresponds to {ball_desc}?"
    
    # Position labels A, B, C, etc. are shown in the video
    letters = "ABCDEFGHIJ"
    choices = [letters[i] for i in range(num_balls)]
    
    # Find the answer index based on the letter
    answer_idx = ord(final_pos) - ord('A')
    
    mcq = format_mcq(question, choices, answer_idx)
    
    return {
        "video": item["video"],
        "fps": item["fps"],
        "total_frames": item["total_frames"],
        "resolution": item["resolution"],
        "task": item["task"],
        "ground_truth": item["ground_truth"],
        **mcq
    }


def detect_game_type(filename: str) -> str:
    """Detect game type from JSON filename."""  
    name = filename.lower()
    if 'card' in name:
        return 'card'
    elif 'cup' in name:
        return 'cup'
    elif 'chess' in name:
        return 'chess'
    elif 'pool' in name:
        return 'pool'
    return 'unknown'


def process_json_file(filepath: Path) -> list[dict]:
    """Process a single JSON file and generate QA items."""
    game_type = detect_game_type(filepath.name)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    generators = {
        'card': generate_card_qa,
        'cup': generate_cup_qa,
        'chess': generate_chess_qa,
        'pool': generate_pool_qa
    }
    
    generator = generators.get(game_type)
    if generator is None:
        print(f"Warning: Unknown game type for {filepath.name}, skipping...")
        return []
    
    for item in data:
        qa_item = generator(item)
        results.append(qa_item)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA JSON for video LLM evaluation"
    )
    parser.add_argument(
        "input_dir",
        nargs='?',
        default="vetbench/card",
        type=str,
        help="Directory containing JSON files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="vetbench/card/QA.json",
        help="Output JSON file path (default: QA.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible choice generation (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' does not exist")
        return 1
    
    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in '{input_dir}'")
        return 1
    
    print(f"Found {len(json_files)} JSON file(s) in '{input_dir}'")
    
    # Process all JSON files
    all_qa_items = []
    for json_file in sorted(json_files):
        print(f"Processing: {json_file.name}")
        items = process_json_file(json_file)
        all_qa_items.extend(items)
        print(f"  Generated {len(items)} QA items")
    
    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa_items, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal: {len(all_qa_items)} QA items written to '{output_path}'")
    return 0


if __name__ == "__main__":
    exit(main())

