import argparse
import re
from pathlib import Path
from spellchecker import SpellChecker

# Function Definitions

def remove_unnecessary_data(text):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def remove_duplicates(lines):
    return list(dict.fromkeys(lines))

def handle_special_characters(text):
    text = text.replace('“', '"').replace('”', '"').replace('£', 'GBP')
    text = re.sub(r'[^\w\s,.!?\"\'-]', '', text)
    return text

def address_formatting_issues(text):
    text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', r'\3-\1-\2', text)  # Example for date format
    return text

def encode_properly(text):
    try:
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        print(f"Encoding error: {e}")
        return text

def normalize_text(text, to_lower=True):
    if to_lower:
        text = text.lower()
    return re.sub(r'[^\w\s]', '', text)

def validate_data(lines):
    valid_lines = [line for line in lines if line.strip()]
    return valid_lines

# Main Function to Process Files

def process_file(file_path, output_path, args):
    print(f"Processing file: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Default behavior: do not perform any cleanup unless specified
    if args.remove_whitespace:
        content = remove_unnecessary_data(content)

    # Apply additional processing based on flags
    if args.remove_duplicates:
        lines = content.splitlines()
        lines = validate_data(lines)
        lines = remove_duplicates(lines)
        cleaned_content = '\n'.join(lines)
    else:
        cleaned_content = content

    if args.handle_special_chars:
        cleaned_content = handle_special_characters(cleaned_content)

    if args.format_dates:
        cleaned_content = address_formatting_issues(cleaned_content)

    if args.spell_check:
        cleaned_content = spell_check(cleaned_content)

    # Determine output file path
    if output_path.is_dir():
        output_file_path = output_path / f"{file_path.stem}.clean.md"
    else:
        output_file_path = Path(output_path)  # Treat as a file name
    
    # Write cleaned content to output file (overrides if exists)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    
    print(f"Cleaned file saved as: {output_file_path}")

# Main Function to Handle Input/Output

def main(args):
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        return
    
    # Determine output path
    output_path = Path(args.output) if args.output else None
    
    # If input is a directory, process each markdown file
    if input_path.is_dir():
        for md_file in input_path.glob('*.md'):
            process_file(md_file, output_path or md_file.parent, args)
    
    # If input is a single file
    elif input_path.is_file() and input_path.suffix == '.md':
        process_file(input_path, output_path or input_path.parent, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleanup MD files in a directory or a single file.', allow_abbrev=False)
    
    # Input and output arguments accepting files or directories
    parser.add_argument('input_file', type=str, help='Input MD file or directory containing MD files')
    
    parser.add_argument('-o', '--output', type=str, help='Output MD file name or directory for cleaned MD files')

    # Cleanup options with shortened arguments
    parser.add_argument('-r', '--remove-whitespace', action='store_true', help='Remove unnecessary whitespace')
    
    parser.add_argument('-n', '--normalize-case', action='store_true', help='Normalize case (lowercase by default)')
    
    parser.add_argument('--to-lower', action='store_true', help='Convert text to lowercase (used with --normalize-case)')
    
    parser.add_argument('-d', '--remove-duplicates', action='store_true', help='Remove duplicate lines')
    
    parser.add_argument('-s', '--handle-special-chars', action='store_true', help='Handle special characters')
    
    parser.add_argument('-f', '--format-dates', action='store_true', help='Standardize date formats')
    
    parser.add_argument('-l', '--convert-links', action='store_true', help='Convert hyperlinks to Markdown format')
    
    parser.add_argument('-c', '--spell-check', action='store_true', help='Perform spell check on the text')

    args = parser.parse_args()
    
    main(args)
