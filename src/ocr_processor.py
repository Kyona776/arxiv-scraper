"""
OCR処理モジュール
複数のOCRモデルを統合し、高精度テキスト抽出を提供
"""

import os
import io
import time
import base64
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from dataclasses import dataclass
from loguru import logger

@dataclass
class OCRResult:
    """OCR処理結果を格納するデータクラス"""
    text: str
    confidence: float
    processing_time: float
    model_used: str
    page_count: int
    metadata: Dict[str, Any]
    errors: List[str]

class OCRProcessor(ABC):
    """OCR処理の抽象基底クラス"""
    
    @abstractmethod
    def process_pdf(self, pdf_path: str) -> OCRResult:
        """PDFファイルを処理してテキストを抽出"""
        pass
    
    @abstractmethod
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """画像からテキストを抽出"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """このOCRモデルが利用可能かチェック"""
        pass

class MistralOCRProcessor(OCRProcessor):
    """Mistral OCR処理クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.get('device', 'cuda')
        self.max_length = config.get('max_length', 2048)
        self.temperature = config.get('temperature', 0.1)
        
    def _load_model(self):
        """モデルを遅延ロード"""
        if self.model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                # NOTE: Mistral OCR is not a real model - using a fallback approach
                # For now, we'll use a basic vision-language model or disable this processor
                logger.warning("Mistral OCR is not available. This processor will be disabled.")
                raise ImportError("Mistral OCR model not available")
                
            except Exception as e:
                logger.error(f"Failed to load Mistral OCR model: {str(e)}")
                raise
    
    def process_pdf(self, pdf_path: str) -> OCRResult:
        """PDFファイルを処理"""
        start_time = time.time()
        errors = []
        
        try:
            self._load_model()
            
            # PDFを開く
            doc = fitz.open(pdf_path)
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # テキスト抽出を試行
                text = page.get_text()
                if len(text.strip()) > 100:  # 十分なテキストがある場合
                    page_texts.append(text)
                else:
                    # OCR処理が必要
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    ocr_text = self.process_image(image)
                    page_texts.append(ocr_text)
            
            doc.close()
            
            full_text = "\n\n".join(page_texts)
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=full_text,
                confidence=0.85,  # 推定信頼度
                processing_time=processing_time,
                model_used="mistral_ocr",
                page_count=len(doc),
                metadata={
                    "pdf_path": pdf_path,
                    "pages_processed": len(page_texts)
                },
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Mistral OCR processing failed: {str(e)}")
            errors.append(str(e))
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used="mistral_ocr",
                page_count=0,
                metadata={"error": str(e)},
                errors=errors
            )
    
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """画像からテキストを抽出"""
        try:
            if isinstance(image, str):
                image = Image.open(image)
            
            # Mistral OCR is not available, return empty string
            logger.warning("Mistral OCR not implemented - no text extracted from image")
            return ""
            
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            return ""
    
    def is_available(self) -> bool:
        """Mistral OCRが利用可能かチェック"""
        # Mistral OCR is not a real model, so always return False
        return False

class MistralOCRAPIProcessor(OCRProcessor):
    """Mistral OCR API処理クラス（直接API呼び出し）"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('api_key') or os.getenv('MISTRAL_API_KEY')
        self.base_url = config.get('base_url', 'https://api.mistral.ai')
        self.model = config.get('model', 'mistral-ocr-latest')
        self.include_image_base64 = config.get('include_image_base64', True)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.timeout = config.get('timeout', 120)
        
        if not self.api_key:
            logger.warning("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")
    
    def _encode_pdf_to_base64(self, pdf_path: str) -> str:
        """PDFファイルをBase64エンコード"""
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        return base64.b64encode(pdf_data).decode('utf-8')
    
    def _call_mistral_ocr_api(self, document: Dict) -> Dict:
        """Mistral OCR APIを呼び出し"""
        if not self.api_key:
            raise ValueError("Mistral API key is required")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'document': document,
            'include_image_base64': self.include_image_base64
        }
        
        url = f"{self.base_url}/v1/ocr"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        raise RuntimeError(f"Failed to call Mistral OCR API after {self.max_retries} attempts")
    
    def process_pdf(self, pdf_path: str) -> OCRResult:
        """PDFファイルを処理"""
        start_time = time.time()
        errors = []
        
        try:
            # PDFファイルサイズチェック
            file_size = os.path.getsize(pdf_path)
            max_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_size:
                raise ValueError(f"PDF file size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum limit (50MB)")
            
            # PDFをBase64エンコード
            base64_pdf = self._encode_pdf_to_base64(pdf_path)
            
            # API呼び出し
            document = {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            }
            
            response = self._call_mistral_ocr_api(document)
            
            # レスポンスからテキストを抽出
            text = response.get('text', '')
            confidence = response.get('confidence', 0.9)  # APIからの信頼度または推定値
            
            # ページ数を取得（PDFから）
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=text,
                confidence=confidence,
                processing_time=processing_time,
                model_used=f"mistral_ocr_api_{self.model}",
                page_count=page_count,
                metadata={
                    "pdf_path": pdf_path,
                    "file_size": file_size,
                    "api_response": response
                },
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Mistral OCR API processing failed: {str(e)}")
            errors.append(str(e))
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"mistral_ocr_api_{self.model}",
                page_count=0,
                metadata={"error": str(e)},
                errors=errors
            )
    
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """画像からテキストを抽出"""
        try:
            if isinstance(image, str):
                # 画像ファイルパス
                with open(image, 'rb') as f:
                    image_data = f.read()
                # 拡張子から形式を判定
                ext = Path(image).suffix.lower()
                if ext in ['.png']:
                    mime_type = 'image/png'
                elif ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext in ['.avif']:
                    mime_type = 'image/avif'
                else:
                    logger.warning(f"Unsupported image format: {ext}")
                    return ""
                
                base64_image = base64.b64encode(image_data).decode('utf-8')
                document = {
                    "type": "document_url",
                    "document_url": f"data:{mime_type};base64,{base64_image}"
                }
                
            elif isinstance(image, Image.Image):
                # PIL Image
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_data = buffer.getvalue()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                document = {
                    "type": "document_url",
                    "document_url": f"data:image/png;base64,{base64_image}"
                }
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return ""
            
            response = self._call_mistral_ocr_api(document)
            return response.get('text', '')
            
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            return ""
    
    def is_available(self) -> bool:
        """Mistral OCR APIが利用可能かチェック"""
        return bool(self.api_key)

class MistralOCROpenRouterProcessor(OCRProcessor):
    """Mistral OCR OpenRouter API処理クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('api_key') or os.getenv('OPENROUTER_API_KEY')
        self.base_url = config.get('base_url', 'https://openrouter.ai/api/v1')
        self.model = config.get('model', 'mistral/mistral-ocr-latest')
        self.site_url = config.get('site_url', 'https://arxiv-scraper.local')
        self.site_name = config.get('site_name', 'ArXiv Scraper')
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.timeout = config.get('timeout', 120)
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
    
    def _encode_pdf_to_base64(self, pdf_path: str) -> str:
        """PDFファイルをBase64エンコード"""
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        return base64.b64encode(pdf_data).decode('utf-8')
    
    def _call_openrouter_ocr_api(self, messages: List[Dict]) -> Dict:
        """OpenRouter APIを呼び出し"""
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': self.site_url,
            'X-Title': self.site_name
        }
        
        payload = {
            'model': self.model,
            'messages': messages,
            'stream': False
        }
        
        url = f"{self.base_url}/chat/completions"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"OpenRouter API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        raise RuntimeError(f"Failed to call OpenRouter API after {self.max_retries} attempts")
    
    def process_pdf(self, pdf_path: str) -> OCRResult:
        """PDFファイルを処理"""
        start_time = time.time()
        errors = []
        
        try:
            # PDFファイルサイズチェック
            file_size = os.path.getsize(pdf_path)
            max_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_size:
                raise ValueError(f"PDF file size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum limit (50MB)")
            
            # PDFをBase64エンコード
            base64_pdf = self._encode_pdf_to_base64(pdf_path)
            
            # OpenRouter APIメッセージ形式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this PDF document. Preserve the structure and formatting as much as possible. Return the text in markdown format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{base64_pdf}"
                            }
                        }
                    ]
                }
            ]
            
            response = self._call_openrouter_ocr_api(messages)
            
            # レスポンスからテキストを抽出
            text = ""
            if 'choices' in response and len(response['choices']) > 0:
                text = response['choices'][0]['message']['content']
            
            # ページ数を取得（PDFから）
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=text,
                confidence=0.85,  # OpenRouterでは信頼度が提供されないため推定値
                processing_time=processing_time,
                model_used=f"mistral_ocr_openrouter_{self.model}",
                page_count=page_count,
                metadata={
                    "pdf_path": pdf_path,
                    "file_size": file_size,
                    "api_response": response
                },
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Mistral OCR OpenRouter API processing failed: {str(e)}")
            errors.append(str(e))
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"mistral_ocr_openrouter_{self.model}",
                page_count=0,
                metadata={"error": str(e)},
                errors=errors
            )
    
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """画像からテキストを抽出"""
        try:
            if isinstance(image, str):
                # 画像ファイルパス
                with open(image, 'rb') as f:
                    image_data = f.read()
                # 拡張子から形式を判定
                ext = Path(image).suffix.lower()
                if ext in ['.png']:
                    mime_type = 'image/png'
                elif ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext in ['.avif']:
                    mime_type = 'image/avif'
                else:
                    logger.warning(f"Unsupported image format: {ext}")
                    return ""
                
                base64_image = base64.b64encode(image_data).decode('utf-8')
                image_url = f"data:{mime_type};base64,{base64_image}"
                
            elif isinstance(image, Image.Image):
                # PIL Image
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_data = buffer.getvalue()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                image_url = f"data:image/png;base64,{base64_image}"
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return ""
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Preserve the structure and formatting as much as possible."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ]
            
            response = self._call_openrouter_ocr_api(messages)
            
            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['message']['content']
            
            return ""
            
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            return ""
    
    def is_available(self) -> bool:
        """Mistral OCR OpenRouter APIが利用可能かチェック"""
        return bool(self.api_key)

class NougatOCRProcessor(OCRProcessor):
    """Nougat OCR処理クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.device = config.get('device', 'cuda')
        self.model_size = config.get('model_size', 'base')
        self.recompute = config.get('recompute', False)
    
    def _load_model(self):
        """モデルを遅延ロード"""
        if self.model is None:
            try:
                from nougat import NougatModel
                from nougat.utils.dataset import LazyDataset
                
                model_name = f"facebook/nougat-{self.model_size}"
                self.model = NougatModel.from_pretrained(model_name)
                self.model.to(self.device)
                
                logger.info(f"Nougat model loaded: {model_name}")
                
            except ImportError as e:
                logger.error(f"Nougat OCR not installed: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Failed to load Nougat model: {str(e)}")
                raise
    
    def process_pdf(self, pdf_path: str) -> OCRResult:
        """PDFファイルを処理"""
        start_time = time.time()
        errors = []
        
        try:
            self._load_model()
            
            # Nougat処理
            from nougat.utils.dataset import LazyDataset
            dataset = LazyDataset(
                pdf_path,
                partial=False,
                recompute=self.recompute
            )
            
            predictions = []
            for sample in dataset:
                prediction = self.model.inference(sample)
                predictions.append(prediction)
            
            full_text = "\n\n".join(predictions)
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=full_text,
                confidence=0.90,  # Nougatは学術論文に特化しているため高い信頼度
                processing_time=processing_time,
                model_used="nougat",
                page_count=len(predictions),
                metadata={
                    "pdf_path": pdf_path,
                    "model_size": self.model_size
                },
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Nougat OCR processing failed: {str(e)}")
            errors.append(str(e))
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used="nougat",
                page_count=0,
                metadata={"error": str(e)},
                errors=errors
            )
    
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """画像からテキストを抽出"""
        # NougatはPDF専用のため、画像処理は未実装
        logger.warning("Nougat OCR does not support direct image processing")
        return ""
    
    def is_available(self) -> bool:
        """Nougat OCRが利用可能かチェック"""
        try:
            import nougat
            return True
        except ImportError:
            return False

class UnstructuredOCRProcessor(OCRProcessor):
    """Unstructured OCR処理クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategy = config.get('strategy', 'hi_res')
        self.model_name = config.get('model_name', 'yolox')
    
    def process_pdf(self, pdf_path: str) -> OCRResult:
        """PDFファイルを処理"""
        start_time = time.time()
        errors = []
        
        try:
            from unstructured.partition.pdf import partition_pdf
            
            # Unstructured処理
            elements = partition_pdf(
                filename=pdf_path,
                strategy=self.strategy,
                model_name=self.model_name,
                extract_images_in_pdf=True,
                infer_table_structure=True
            )
            
            # テキストを結合
            texts = []
            for element in elements:
                if hasattr(element, 'text'):
                    texts.append(element.text)
            
            full_text = "\n\n".join(texts)
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=full_text,
                confidence=0.85,
                processing_time=processing_time,
                model_used="unstructured",
                page_count=len(texts),
                metadata={
                    "pdf_path": pdf_path,
                    "strategy": self.strategy,
                    "elements_count": len(elements)
                },
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Unstructured OCR processing failed: {str(e)}")
            errors.append(str(e))
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used="unstructured",
                page_count=0,
                metadata={"error": str(e)},
                errors=errors
            )
    
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """画像からテキストを抽出"""
        try:
            from unstructured.partition.image import partition_image
            
            if isinstance(image, Image.Image):
                # PIL ImageをBytesIOに変換
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                elements = partition_image(
                    file=img_bytes,
                    strategy=self.strategy,
                    model_name=self.model_name
                )
            else:
                elements = partition_image(
                    filename=image,
                    strategy=self.strategy,
                    model_name=self.model_name
                )
            
            texts = [element.text for element in elements if hasattr(element, 'text')]
            return "\n".join(texts)
            
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            return ""
    
    def is_available(self) -> bool:
        """Unstructured OCRが利用可能かチェック"""
        try:
            import unstructured
            return True
        except ImportError:
            return False

class SuryaOCRProcessor(OCRProcessor):
    """Surya OCR処理クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.device = config.get('device', 'cuda')
        self.batch_size = config.get('batch_size', 2)
        self.det_batch_size = config.get('det_batch_size', 2)
    
    def _load_model(self):
        """モデルを遅延ロード"""
        if self.model is None:
            try:
                from surya.ocr import run_ocr
                from surya.model.detection import segformer
                from surya.model.recognition import model as rec_model
                
                # Suryaモデルのロード
                self.det_model = segformer.load_model()
                self.rec_model = rec_model.load_model()
                
                logger.info("Surya OCR models loaded")
                
            except Exception as e:
                logger.error(f"Failed to load Surya OCR model: {str(e)}")
                raise
    
    def process_pdf(self, pdf_path: str) -> OCRResult:
        """PDFファイルを処理"""
        start_time = time.time()
        errors = []
        
        try:
            self._load_model()
            
            # PDFを画像に変換
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
            
            doc.close()
            
            # Surya OCR処理
            from surya.ocr import run_ocr
            
            predictions = run_ocr(
                images,
                [["en"]] * len(images),  # 言語設定
                self.det_model,
                self.rec_model,
                batch_size=self.batch_size,
                det_batch_size=self.det_batch_size
            )
            
            # テキストを結合
            page_texts = []
            for prediction in predictions:
                page_text = ""
                for line in prediction.text_lines:
                    page_text += line.text + "\n"
                page_texts.append(page_text)
            
            full_text = "\n\n".join(page_texts)
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=full_text,
                confidence=0.80,
                processing_time=processing_time,
                model_used="surya",
                page_count=len(images),
                metadata={
                    "pdf_path": pdf_path,
                    "pages_processed": len(images)
                },
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Surya OCR processing failed: {str(e)}")
            errors.append(str(e))
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used="surya",
                page_count=0,
                metadata={"error": str(e)},
                errors=errors
            )
    
    def process_image(self, image: Union[str, Image.Image]) -> str:
        """画像からテキストを抽出"""
        try:
            self._load_model()
            
            if isinstance(image, str):
                image = Image.open(image)
            
            from surya.ocr import run_ocr
            
            predictions = run_ocr(
                [image],
                [["en"]],
                self.det_model,
                self.rec_model,
                batch_size=1,
                det_batch_size=1
            )
            
            if predictions:
                text = ""
                for line in predictions[0].text_lines:
                    text += line.text + "\n"
                return text
            
            return ""
            
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            return ""
    
    def is_available(self) -> bool:
        """Surya OCRが利用可能かチェック"""
        try:
            import surya
            return True
        except ImportError:
            return False

class OCRManager:
    """OCR処理マネージャー"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.primary_model = config.get('model', 'mistral_ocr')
        self.fallback_models = config.get('fallback_models', [])
        self.processors = {}
        
        # プロセッサーを初期化
        self._init_processors()
    
    def _init_processors(self):
        """利用可能なプロセッサーを初期化"""
        processor_classes = {
            'mistral_ocr': MistralOCRProcessor,
            'mistral_ocr_api': MistralOCRAPIProcessor,
            'mistral_ocr_openrouter': MistralOCROpenRouterProcessor,
            'nougat': NougatOCRProcessor,
            'unstructured': UnstructuredOCRProcessor,
            'surya': SuryaOCRProcessor
        }
        
        for model_name, processor_class in processor_classes.items():
            try:
                model_config = self.config.get(model_name, {})
                model_config.update({
                    'device': self.config.get('device', 'cuda'),
                    'batch_size': self.config.get('batch_size', 4)
                })
                
                processor = processor_class(model_config)
                if processor.is_available():
                    self.processors[model_name] = processor
                    logger.info(f"OCR processor {model_name} initialized")
                else:
                    logger.warning(f"OCR processor {model_name} not available")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {str(e)}")
    
    def process_pdf(self, pdf_path: str) -> OCRResult:
        """PDFファイルを処理（フォールバック機能付き）"""
        
        # 処理順序を決定
        models_to_try = [self.primary_model] + self.fallback_models
        models_to_try = [m for m in models_to_try if m in self.processors]
        
        if not models_to_try:
            raise RuntimeError("No OCR processors available")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying OCR model: {model_name}")
                processor = self.processors[model_name]
                result = processor.process_pdf(pdf_path)
                
                if result.text and len(result.text.strip()) > 100:  # 最小品質チェック
                    logger.info(f"OCR successful with {model_name}")
                    return result
                else:
                    logger.warning(f"OCR result quality too low with {model_name}")
                    
            except Exception as e:
                logger.error(f"OCR failed with {model_name}: {str(e)}")
                continue
        
        # すべて失敗した場合
        logger.error("All OCR processors failed")
        return OCRResult(
            text="",
            confidence=0.0,
            processing_time=0.0,
            model_used="none",
            page_count=0,
            metadata={"error": "All OCR processors failed"},
            errors=["All OCR processors failed"]
        )
    
    def get_available_models(self) -> List[str]:
        """利用可能なモデルのリストを取得"""
        return list(self.processors.keys())

def create_ocr_manager(config: Dict) -> OCRManager:
    """
    設定からOCRマネージャーを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        OCRManager インスタンス
    """
    return OCRManager(config.get('ocr', {}))