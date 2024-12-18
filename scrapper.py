import os
import subprocess
from pathlib import Path
import argparse
import pdfplumber
import re
import json
from datetime import datetime

def validate_url(url):
    return url.startswith(('http://', 'https://'))

def read_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def search_and_download_pdfs(target, max_results=100, download_dir='downloaded_pdfs'):
    command = ["ddgs", "text", "-k", f"'site:{target}' 'filetype:pdf'", "-d", "-m", str(max_results), "-dd", download_dir]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Error output: {e.stderr}")
        return False

def remove_non_pdf_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if not file.lower().endswith('.pdf'):
            os.remove(file_path)
            print(f"Removed non-PDF file: {file}")

def extract_text_from_pdfs(directory):
    extracted_texts = {}
    files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
                extracted_texts[file] = text
                print(f"Extracted text from {file}")
        except Exception as e:
            print(f"Error extracting text from {file}: {e}")
    return extracted_texts

def clean_text(text):
    text = re.sub(r'[^\x20-\x7E]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    lines = text.split('\n')
    unique_lines = list(dict.fromkeys(lines))
    cleaned_text = '\n'.join(unique_lines)
    return cleaned_text

def output_cleaned_text(cleaned_texts, output_format='json', output_dir='cleaned_texts'):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().isoformat()
    
    if output_format == 'json':
        for filename, text in cleaned_texts.items():
            output_path = os.path.join(output_dir, f'{Path(filename).stem}_cleaned.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({'text': text, 'source': filename, 'timestamp': timestamp}, f, ensure_ascii=False, indent=4)
            print(f"Output JSON for {filename} at {output_path}")
    elif output_format == 'txt':
        for filename, text in cleaned_texts.items():
            output_path = os.path.join(output_dir, f'{Path(filename).stem}_cleaned.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Output text for {filename} at {output_path}")

def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description="Search, download, and process PDF files from specified websites using DuckDuckGo search.",
        epilog="Example usage: python script.py -u https://www.example.com"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-u', '--url', type=str, help="Specify a single URL or domain to search.")
    group.add_argument('-f', '--file', type=str, help="Specify a text file containing a list of URLs to search.")
    
    parser.add_argument('-m', '--max', type=int, default=100, help="Maximum number of search results. Default is 100.")
    parser.add_argument('--parse', action='store_true', help="Parse and extract text from downloaded PDFs.")
    parser.add_argument('--clean', action='store_true', help="Clean extracted text from PDFs.")
    
    return parser

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    urls_to_process = []

    if args.url:
        if validate_url(args.url):
            urls_to_process = [args.url]
        else:
            print(f"Error: Invalid URL format. Please use http:// or https:// schema.")
            return
    elif args.file:
        try:
            urls_to_process = read_urls_from_file(args.file)
            invalid_urls = [url for url in urls_to_process if not validate_url(url)]
            if invalid_urls:
                print(f"Error: The following URLs in the file are invalid:")
                for url in invalid_urls:
                    print(f"  - {url}")
                return
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            return

    download_dir = 'downloaded_pdfs'
    os.makedirs(download_dir, exist_ok=True)

    for url in urls_to_process:
        print(f"Processing: {url}")
        target = url.split('//')[1].split('/')[0]
        
        if search_and_download_pdfs(target, args.max, download_dir):
            print(f"Search and download completed for {url}")
        else:
            print(f"CLI search failed for {url}.")

    remove_non_pdf_files(download_dir)

    if args.parse:
        print("Extracting text from PDFs...")
        extracted_texts = extract_text_from_pdfs(download_dir)
        
        if args.clean:
            print("Cleaning extracted text...")
            cleaned_texts = {filename: clean_text(text) for filename, text in extracted_texts.items()}
            output_cleaned_text(cleaned_texts, 'json', 'cleaned_texts')
        else:
            output_cleaned_text(extracted_texts, 'json', 'extracted_texts')

if __name__ == "__main__":
    main()
