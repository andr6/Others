import argparse
import cv2
import fitz
import logging
import numpy as np
import os
import pytesseract
import sys
import hashlib
import redis
import requests
import spacy
from spacy.tokens import Span
from typing import Optional

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationModule:
    def __init__(self,
                 ollama_host: str = "http://localhost:11434",
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 chunk_size: int = 2000,
                 model_name: str = "mistral"):
        self.ollama_host = ollama_host
        self.redis = redis.Redis(host=redis_host, port=redis_port)
        self.chunk_size = chunk_size
        self.model_name = model_name

        # Load Russian NER model (need to install: python -m spacy download ru_core_news_lg)
        try:
            self.nlp = spacy.load("ru_core_news_lg")
        except OSError:
            raise RuntimeError("Russian spaCy model not found. Install with: python -m spacy download ru_core_news_lg")

    def _generate_cache_key(self, text: str) -> str:
        return f"translation:{hashlib.md5(text.encode()).hexdigest()}"

    def _preserve_entities(self, text: str) -> tuple[str, dict]:
        """Identify and replace entities with placeholders"""
        doc = self.nlp(text)
        replacements = {}
        entity_count = 0

        # Sort entities by start position in reverse order to avoid replacement conflicts
        entities = sorted(doc.ents, key=lambda x: x.start_char, reverse=True)

        for ent in entities:
            placeholder = f"ENT_{entity_count}_{ent.label_}"
            replacements[placeholder] = ent.text
            text = text[:ent.start_char] + placeholder + text[ent.end_char:]
            entity_count += 1

        return text, replacements

    def _restore_entities(self, text: str, replacements: dict) -> str:
        """Replace placeholders with original entities"""
        for placeholder, original in reversed(replacements.items()):
            text = text.replace(placeholder, original)
        return text

    def _translate_chunk(self, chunk: str) -> str:
        """Translate a single chunk with entity preservation"""
        # Check cache first
        cache_key = self._generate_cache_key(chunk)
        if cached := self.redis.get(cache_key):
            return cached.decode()

        # Process entities
        processed_text, replacements = self._preserve_entities(chunk)

        # Ollama API request
        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.model_name,
                "prompt": f"Translate this to English while maintaining formatting and special characters: {processed_text}",
                "stream": False
            }
        )

        if response.status_code != 200:
            raise Exception(f"Translation failed: {response.text}")

        translated = response.json()["response"]

        # Restore entities and cache result
        final_text = self._restore_entities(translated, replacements)
        self.redis.setex(cache_key, 3600, final_text)  # Cache for 1 hour

        return final_text

    def translate(self, text: str) -> str:
        """Batch translation with context-aware chunking"""
        # Split text into chunks preserving paragraphs
        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in text.split("\n"):
            if current_length + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(paragraph)
            current_length += len(paragraph)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # Process chunks in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._translate_chunk, chunk) for chunk in chunks]
            results = [f.result() for f in futures]

        return "\n".join(results)

class PDFProcessor:
    def __init__(self, enable_translation: bool = False):
        self.image_dir = "temp_images"
        self.enable_translation = enable_translation
        os.makedirs(self.image_dir, exist_ok=True)

        if enable_translation:
            self.translator = TranslationModule()

    def process_pdf(self, pdf_path: str) -> str:
        """Main processing method for a single PDF"""
        try:
            self.doc = fitz.open(pdf_path)
            full_text = []

            for page in self.doc:
                page_text = self._process_page(page, Path(pdf_path).stem)
                full_text.append(page_text)

            result = "\n".join(full_text)

            if self.enable_translation:
                result = self.translator.translate(result)

            return result
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}")
            raise
        finally:
            if hasattr(self, 'doc'):
                self.doc.close()

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """Image preprocessing pipeline"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=20)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(denoised)

    @staticmethod
    def process_image(image_data: Tuple[bytes, Tuple]) -> str:
        """Static method for parallel OCR processing"""
        try:
            img_bytes, bbox = image_data
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            processed = PDFProcessor.preprocess_image(img)
            return pytesseract.image_to_string(processed, lang='rus+eng', config='--psm 6').strip()
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""

    def process_pdf(self, pdf_path: str) -> str:
        """Main processing method for a single PDF"""
        try:
            # Open PDF document and store as instance attribute
            self.doc = fitz.open(pdf_path)
            full_text = []

            for page in self.doc:
                page_text = self._process_page(page, Path(pdf_path).stem)
                full_text.append(page_text)

            return "\n".join(full_text)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}")
            raise
        finally:
            # Clean up document resource
            if hasattr(self, 'doc'):
                self.doc.close()

    def _process_page(self, page, pdf_stem: str) -> str:
        """Process individual page"""
        page_text = []

        # Extract selectable text
        if text := page.get_text():
            page_text.append(text)

        # Process images using the document instance
        if images := page.get_images(full=True):
            page_text.extend(self._process_images(images, page.number, pdf_stem))

        return "\n".join(page_text)

    def _process_images(self, images, page_num: int, pdf_stem: str) -> List[str]:
        """Process images with parallel execution"""
        image_data = []
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                # Access document through instance attribute
                base_image = self.doc.extract_image(xref)
                image_data.append((
                    base_image["image"],
                    img[1:5]  # Bounding box
                ))
            except Exception as e:
                logger.error(f"Image extraction failed: {str(e)}")

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(PDFProcessor.process_image, image_data))

        return [text for text in results if text]

class BatchProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processor = PDFProcessor()
        os.makedirs(output_dir, exist_ok=True)

    def process_batch(self):
        """Process all PDFs in directory"""
        pdf_files = [
            f for f in Path(self.input_dir).iterdir()
            if f.suffix.lower() == ".pdf"
        ]

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.process_single, path): path.name
                for path in pdf_files
            }

            for future in as_completed(futures):
                try:
                    text = future.result()
                    path = futures[future]
                    output_path = Path(self.output_dir) / f"{path.stem}_output.txt"
                    output_path.write_text(text, encoding="utf-8")
                except Exception as e:
                    logger.error(f"Failed: {str(e)}")

    def process_single(self, pdf_path: Path) -> str:
        """Wrapper for single PDF processing"""
        return self.processor.process_pdf(str(pdf_path))

class CLI:
    @staticmethod
    def run():
        parser = argparse.ArgumentParser(
            description="PDF Processing Pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("-file", help="Single PDF file to process")
        parser.add_argument("-dir", help="Directory of PDFs to process")
        parser.add_argument("-output", default="./output", help="Output directory")
        parser.add_argument("-translate", action="store_true", help="Enable translation to English")
        parser.add_argument("-model", default="deepseek", help="Ollama model name")
        parser.add_argument("-redis-host", default="localhost", help="Redis server hostname")
        parser.add_argument("--chunk-size", type=int, default=2000, help="Translation chunk size in characters")


        args = parser.parse_args()

        if not args.file and not args.dir:
            logger.error("Specify -f or -d")
            sys.exit(1)

        output_dir = Path(args.output)  # Fixed line: removed extra parenthesis
        output_dir.mkdir(exist_ok=True)

        processor = PDFProcessor(enable_translation=args.translate)
        if args.translate:
            processor.translator = TranslationModule(model_name=args.model,redis_host=args.redis_host,chunk_size=args.chunk_size
            )
        if args.file:
            CLI().process_single(Path(args.file), output_dir)
        elif args.dir:
            BatchProcessor(args.dir, str(output_dir)).process_batch()

    def process_single(self, pdf_path: Path, output_dir: Path):
        try:
            text = PDFProcessor().process_pdf(str(pdf_path))
            output_file = output_dir / f"{pdf_path.stem}_output.txt"
            output_file.write_text(text, encoding="utf-8")
            logger.info(f"Successfully processed {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    CLI.run()
