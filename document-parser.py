#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import sys
import fitz  # PyMuPDF
import pymupdf4llm
import docx2txt
import openpyxl
import pytesseract
from PIL import Image
import csv
import xml.etree.ElementTree as ET
import xml.dom.minidom

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentParser:
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.xlsx'}
    SUPPORTED_OUTPUT_FORMATS = {'md', 'csv', 'xml'}

    def __init__(self, should_extract_images: bool = True, should_ocr_images: bool = False, output_formats: set = {'md'}, generate_llama_index: bool = False):
        self.should_extract_images = should_extract_images
        self.should_ocr_images = should_ocr_images
        self.output_formats = output_formats
        self.generate_llama_index = generate_llama_index

    def extract_text(self, file_path: Path) -> str:
        try:
            if file_path.suffix == '.pdf':
                return pymupdf4llm.to_markdown(str(file_path))
            elif file_path.suffix == '.txt':
                return file_path.read_text(encoding='utf-8')
            elif file_path.suffix == '.docx':
                return docx2txt.process(str(file_path))
            elif file_path.suffix == '.xlsx':
                workbook = openpyxl.load_workbook(file_path)
                text = []
                for sheet in workbook.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        text.append('\t'.join(str(cell) for cell in row if cell is not None))
                return '\n'.join(text)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    def extract_images_from_file(self, file_path: Path, output_dir: Path):
        if file_path.suffix != '.pdf':
            logger.warning(f"Image extraction not supported for {file_path.suffix} files")
            return

        try:
            doc = fitz.open(file_path)
            for i, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = output_dir / f"page{i+1}_img{img_index+1}.{image_ext}"
                    with open(image_filename, "wb") as image_file:
                        image_file.write(image_bytes)
                    logger.info(f"Extracted image: {image_filename}")

                    if self.should_ocr_images:
                        self.ocr_image(image_filename)
        except Exception as e:
            logger.error(f"Error extracting images from {file_path}: {str(e)}")

    def ocr_image(self, image_path: Path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            ocr_text_file = image_path.with_suffix('.txt')
            ocr_text_file.write_text(text)
            logger.info(f"OCR text saved to: {ocr_text_file}")
        except Exception as e:
            logger.error(f"Error performing OCR on {image_path}: {str(e)}")

    def process_file_or_directory(self, input_path: Path, output_dir: Path) -> list:
        processed_files = []
        if input_path.is_file():
            if input_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                processed_files.extend(self.process_document(input_path, output_dir))
        elif input_path.is_dir():
            for file_path in input_path.glob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    processed_files.extend(self.process_document(file_path, output_dir))
        return processed_files

    def process_document(self, input_path: Path, output_dir: Path) -> list:
        logger.info(f"Processing: {input_path}")
        text = self.extract_text(input_path)
        processed_files = []

        for format in self.output_formats:
            output_path = output_dir / f"{input_path.stem}.{format}"
            if format == 'md':
                output_path.write_text(text, encoding='utf-8')
            elif format == 'csv':
                self.save_as_csv(text, output_path)
            elif format == 'xml':
                self.save_as_xml(text, output_path)
            logger.info(f"Saved {format.upper()} to: {output_path}")
            processed_files.append(output_path)

        if self.should_extract_images:
            image_output_dir = output_dir / f"{input_path.stem}_images"
            image_output_dir.mkdir(exist_ok=True)
            self.extract_images_from_file(input_path, image_output_dir)

        # Generate LlamaIndex documents if requested
        if self.generate_llama_index and input_path.suffix == '.pdf':
            llama_docs = self.create_llama_index_documents(input_path)
            llama_output_file = output_dir / f"{input_path.stem}_llama.json"
            llama_docs_json_str = [doc.to_dict() for doc in llama_docs]
            with open(llama_output_file, 'w', encoding='utf-8') as f:
                json.dump(llama_docs_json_str, f, ensure_ascii=False, indent=4)
            logger.info(f"LlamaIndex documents saved to: {llama_output_file}")

        return processed_files

    def save_as_csv(self, text: str, output_path: Path):
        lines = text.split('\n')
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for line in lines:
                writer.writerow([line])

    def save_as_xml(self, text: str, output_path: Path):
        root = ET.Element("document")
        for line in text.split('\n'):
            sanitized_line = self.sanitize_for_xml(line)
            ET.SubElement(root, "line").text = sanitized_line
        
        xml_str = ET.tostring(root, encoding='unicode')
        pretty_xml = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ")
        output_path.write_text(pretty_xml, encoding='utf-8')

    def sanitize_for_xml(self, value: str) -> str:
        """Sanitize a string for use in XML."""
        return ''.join(c if c.isprintable() and c not in ('<', '>') else ' ' for c in value)

    def create_llama_index_documents(self, input_file: Path):
        """Create LlamaIndex documents from a PDF file."""
        llama_reader = pymupdf4llm.LlamaMarkdownReader()
        llama_docs = llama_reader.load_data(str(input_file))
        return llama_docs

def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Document Parsing and Text Extraction Tool for LLM Training",
        epilog="Example: python document_parser.py -i input.pdf -o output_dir --format md csv xml --ocr"
    )
    
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input file or directory path')
    parser.add_argument('-o', '--output', type=Path, default=Path.cwd() / 'extracted_documents',
                        help='Output directory for processed documents (default: ./extracted_documents)')
    parser.add_argument('--no-images', action='store_true', help='Disable image extraction')
    parser.add_argument('--ocr', action='store_true', help='Enable OCR for extracted images')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--show-supported-types', action='store_true', help='Display supported file types and exit')
    parser.add_argument('--format', nargs='+', choices=DocumentParser.SUPPORTED_OUTPUT_FORMATS,
                        default=['md'], help='Output format(s) (default: md)')
    parser.add_argument('--llama-index', action='store_true', help='Generate LlamaIndex documents from PDF files')

    return parser

def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.show_supported_types:
        print("Supported File Types:", ", ".join(DocumentParser.SUPPORTED_EXTENSIONS))
        print("Supported Output Formats:", ", ".join(DocumentParser.SUPPORTED_OUTPUT_FORMATS))
        return

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    args.output.mkdir(parents=True, exist_ok=True)

    doc_parser = DocumentParser(
        should_extract_images=not args.no_images,
        should_ocr_images=args.ocr,
        output_formats=set(args.format),
        generate_llama_index=args.llama_index
    )

    try:
        processed_files = doc_parser.process_file_or_directory(args.input, args.output)
        print(f"\nSuccessfully processed {len(processed_files)} output files.")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
