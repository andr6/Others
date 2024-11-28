import argparse
import re
from spellchecker import SpellChecker

# Function Definitions

def remove_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def normalize_case(text, to_lower=True):
    return text.lower() if to_lower else text.title()

def replace_special_characters(text):
    return re.sub(r'[^\w\s,.!?]', '', text)

def ensure_utf8_encoding(text):
    return text.encode('utf-8', errors='ignore').decode('utf-8')

def format_paragraphs(text):
    return '\n\n'.join([f"# {line.strip()}" if line.startswith('#') else line for line in text.splitlines()])

def format_lists(text):
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        if line.startswith('* ') or line.startswith('- '):
            formatted_lines.append(f"* {line[2:]}")
        elif line[0].isdigit() and line[1] == '.':
            formatted_lines.append(f"{line}")
    return '\n'.join(formatted_lines)

def convert_tables_to_markdown(table_data):
    markdown_table = ""
    for row in table_data:
        markdown_table += "| " + " | ".join(row) + " |\n"
        markdown_table += "| " + " | ".join(['---'] * len(row)) + " |\n"
    return markdown_table.strip()

def extract_images_and_references(image_paths):
    return [f"![alt text]({path})" for path in image_paths]

def convert_hyperlinks(text):
    return re.sub(r'(https?://[^\s]+)', r'[\1](\1)', text)

def spell_check(text):
    spell = SpellChecker()
    words = text.split()
    corrected_text = ' '.join([spell.candidates(word).pop() if word not in spell else word for word in words])
    return corrected_text

def extract_metadata(metadata):
    title = metadata.get('title', 'Untitled')
    author = metadata.get('author', 'Unknown')
    date = metadata.get('date', 'Unknown Date')
    
    return f"# {title}\n\n*Author:* {author}\n*Date:* {date}\n"

# Main Function

def main(args):
    with open(args.input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    if args.remove_whitespace:
        content = remove_whitespace(content)
    
    if args.normalize_case:
        content = normalize_case(content, to_lower=args.to_lower)
    
    if args.replace_special_chars:
        content = replace_special_characters(content)

    # Additional processing based on user arguments
    if args.format_paragraphs:
        content = format_paragraphs(content)
        
    if args.format_lists:
        content = format_lists(content)

    if args.convert_links:
        content = convert_hyperlinks(content)

    if args.spell_check:
        content = spell_check(content)

    # Write cleaned content to output file
    with open(args.output_file, 'w', encoding='utf-8') as file:
        file.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleanup MD files.')
    
    # Input and output file arguments
    parser.add_argument('input_file', type=str, help='Input MD file path')
    parser.add_argument('output_file', type=str, help='Output MD file path')
    
    # Cleanup options
    parser.add_argument('--remove-whitespace', action='store_true', help='Remove unnecessary whitespace')
    parser.add_argument('--normalize-case', action='store_true', help='Normalize case (lowercase by default)')
    parser.add_argument('--to-lower', action='store_true', help='Convert text to lowercase (used with --normalize-case)')
    
    parser.add_argument('--replace-special-chars', action='store_true', help='Replace special characters')
    
    parser.add_argument('--format-paragraphs', action='store_true', help='Format paragraphs correctly')
    
    parser.add_argument('--format-lists', action='store_true', help='Format bulleted/numbered lists correctly')
    
    parser.add_argument('--convert-links', action='store_true', help='Convert hyperlinks to Markdown format')
    
    parser.add_argument('--spell-check', action='store_true', help='Perform spell check on the text')

    args = parser.parse_args()
    
    main(args)
