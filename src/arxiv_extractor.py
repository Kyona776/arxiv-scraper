"""
arXiv論文抽出メインスクリプト
全パイプラインを統合した実行可能モジュール
"""

import argparse
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger

# 内部モジュールのインポート
from .arxiv_api import create_arxiv_client, ArxivPaper
from .ocr_processor import create_ocr_manager
from .reference_cleaner import create_reference_cleaner
from .text_processor import create_text_processor
from .llm_extractor import create_llm_extractor
from .csv_generator import create_csv_generator


class ArxivExtractor:
    """arXiv論文抽出パイプライン"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # 各コンポーネントを初期化
        self.arxiv_client = create_arxiv_client(self.config)
        self.ocr_manager = create_ocr_manager(self.config)
        self.reference_cleaner = create_reference_cleaner(self.config)
        self.text_processor = create_text_processor(self.config)
        self.llm_extractor = create_llm_extractor(self.config)
        self.csv_generator = create_csv_generator(self.config)
        
        logger.info("ArxivExtractor initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from: {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を返す"""
        return {
            'arxiv': {
                'base_url': 'http://export.arxiv.org/api/query',
                'rate_limit': {'requests_per_second': 1},
                'pdf': {'timeout': 30, 'max_retries': 3}
            },
            'ocr': {
                'model': 'unstructured',
                'fallback_models': ['surya'],
                'device': 'cpu'
            },
            'text_processing': {
                'remove_references': True,
                'citation_cleanup': True,
                'min_text_length': 1000
            },
            'llm': {
                'model': 'gpt-4',
                'fallback_models': ['gpt-3.5-turbo'],
                'openai': {'api_key': None}
            },
            'output': {
                'format': 'csv',
                'output_dir': './output'
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def setup_logging(self):
        """ログ設定を初期化"""
        log_config = self.config.get('logging', {})
        level = log_config.get('level', 'INFO')
        
        # 既存のロガーを削除
        logger.remove()
        
        # コンソール出力設定
        if log_config.get('console', {}).get('enabled', True):
            console_format = log_config.get('console', {}).get(
                'format', 
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            )
            logger.add(sys.stderr, level=level, format=console_format)
        
        # ファイル出力設定
        if log_config.get('file', {}).get('enabled', False):
            file_path = log_config.get('file', {}).get('path', './logs/arxiv_scraper.log')
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                file_path,
                level=level,
                rotation=log_config.get('file', {}).get('max_size', '10MB'),
                retention=log_config.get('file', {}).get('backup_count', 5)
            )
    
    def process_single_paper(
        self, 
        identifier: str, 
        output_filename: Optional[str] = None
    ) -> Dict:
        """
        単一論文を処理
        
        Args:
            identifier: arXiv ID、URL、またはPDFファイルパス
            output_filename: 出力ファイル名（オプション）
            
        Returns:
            処理結果辞書
        """
        logger.info(f"Starting processing for: {identifier}")
        start_time = time.time()
        
        try:
            # Step 1: 論文情報取得
            if identifier.endswith('.pdf'):
                # PDFファイル直接指定の場合
                pdf_path = identifier
                paper_metadata = None
            else:
                # arXiv ID/URLの場合
                paper = self.arxiv_client.get_paper_metadata(identifier)
                if not paper:
                    return {
                        'success': False,
                        'error': f'論文が見つかりません: {identifier}',
                        'processing_time': time.time() - start_time
                    }
                
                # PDFダウンロード
                pdf_path = self.arxiv_client.download_pdf(paper)
                if not pdf_path:
                    return {
                        'success': False,
                        'error': 'PDFダウンロードに失敗しました',
                        'processing_time': time.time() - start_time
                    }
                
                paper_metadata = {
                    'title': paper.title,
                    'authors': paper.authors,
                    'categories': paper.categories,
                    'published': paper.published,
                    'abs_url': paper.abs_url
                }
            
            # Step 2: OCR処理
            logger.info("Starting OCR processing...")
            ocr_start = time.time()
            ocr_result = self.ocr_manager.process_pdf(pdf_path)
            ocr_time = time.time() - ocr_start
            
            if not ocr_result.text:
                return {
                    'success': False,
                    'error': 'OCR処理に失敗しました',
                    'processing_time': time.time() - start_time
                }
            
            logger.info(f"OCR completed in {ocr_time:.2f}s using {ocr_result.model_used}")
            
            # Step 3: Reference削除
            logger.info("Starting reference removal...")
            ref_start = time.time()
            cleaned_result = self.reference_cleaner.clean_text(ocr_result.text)
            ref_time = time.time() - ref_start
            
            logger.info(f"Reference removal completed in {ref_time:.2f}s (confidence: {cleaned_result.confidence:.3f})")
            
            # Step 4: テキスト後処理
            logger.info("Starting text post-processing...")
            text_start = time.time()
            processed_result = self.text_processor.process_text(cleaned_result.cleaned_text)
            text_time = time.time() - text_start
            
            logger.info(f"Text processing completed in {text_time:.2f}s (quality: {processed_result.quality_score:.3f})")
            
            # Step 5: LLM抽出
            logger.info("Starting LLM extraction...")
            llm_start = time.time()
            extraction_result = self.llm_extractor.extract_from_text(
                processed_result.text, 
                paper_metadata
            )
            llm_time = time.time() - llm_start
            
            if not extraction_result.success:
                logger.warning("LLM extraction had issues, but proceeding with available data")
            
            logger.info(f"LLM extraction completed in {llm_time:.2f}s using {extraction_result.model_used}")
            
            # Step 6: CSV出力
            total_processing_time = time.time() - start_time
            csv_path = self.csv_generator.write_single_record(
                extraction_result,
                output_filename,
                total_processing_time
            )
            
            logger.info(f"Processing completed successfully in {total_processing_time:.2f}s")
            logger.info(f"Output saved to: {csv_path}")
            
            return {
                'success': True,
                'csv_path': csv_path,
                'processing_time': total_processing_time,
                'stages': {
                    'ocr_time': ocr_time,
                    'reference_time': ref_time,
                    'text_time': text_time,
                    'llm_time': llm_time
                },
                'quality_metrics': {
                    'ocr_confidence': ocr_result.confidence,
                    'reference_confidence': cleaned_result.confidence,
                    'text_quality': processed_result.quality_score,
                    'llm_confidence': sum(extraction_result.confidence_scores.values()) / len(extraction_result.confidence_scores) if extraction_result.confidence_scores else 0.0
                },
                'extraction_data': extraction_result.data
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_batch(
        self, 
        identifiers: List[str], 
        output_filename: Optional[str] = None
    ) -> Dict:
        """
        複数論文をバッチ処理
        
        Args:
            identifiers: arXiv ID、URL、またはPDFファイルパスのリスト
            output_filename: 出力ファイル名（オプション）
            
        Returns:
            バッチ処理結果辞書
        """
        logger.info(f"Starting batch processing for {len(identifiers)} papers")
        start_time = time.time()
        
        results = []
        extraction_results = []
        processing_times = []
        
        for i, identifier in enumerate(identifiers, 1):
            logger.info(f"Processing paper {i}/{len(identifiers)}: {identifier}")
            
            result = self.process_single_paper(identifier)
            results.append(result)
            
            if result['success']:
                # 成功した場合、CSVには書き込まずに結果を蓄積
                extraction_results.append(result['extraction_data'])
                processing_times.append(result['processing_time'])
            else:
                logger.error(f"Failed to process {identifier}: {result.get('error', 'Unknown error')}")
                # 失敗した場合はN/Aレコードを追加
                from .llm_extractor import ExtractionResult
                failed_result = ExtractionResult(
                    success=False,
                    data={
                        "手法の肝": "N/A",
                        "制限事項": "N/A", 
                        "対象ナレッジ": "N/A",
                        "URL": identifier if not identifier.endswith('.pdf') else "N/A",
                        "タイトル": "N/A",
                        "出版年": "N/A",
                        "研究分野": "N/A",
                        "課題設定": "N/A",
                        "論文の主張": "N/A"
                    },
                    model_used="failed",
                    processing_time=result['processing_time'],
                    confidence_scores={},
                    errors=[result.get('error', 'Processing failed')],
                    metadata={}
                )
                extraction_results.append(failed_result)
                processing_times.append(result['processing_time'])
        
        # バッチCSV出力
        if extraction_results:
            csv_path = self.csv_generator.write_batch_records(
                extraction_results,
                output_filename,
                processing_times
            )
        else:
            csv_path = None
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])
        
        logger.info(f"Batch processing completed: {success_count}/{len(identifiers)} successful in {total_time:.2f}s")
        if csv_path:
            logger.info(f"Batch output saved to: {csv_path}")
        
        return {
            'success': success_count > 0,
            'csv_path': csv_path,
            'total_processing_time': total_time,
            'success_count': success_count,
            'total_count': len(identifiers),
            'individual_results': results
        }
    
    def process_from_file(
        self, 
        input_file: str, 
        output_filename: Optional[str] = None
    ) -> Dict:
        """
        ファイルから論文リストを読み込んでバッチ処理
        
        Args:
            input_file: 論文IDリストファイル（1行1ID）
            output_filename: 出力ファイル名（オプション）
            
        Returns:
            処理結果辞書
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                identifiers = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Loaded {len(identifiers)} identifiers from {input_file}")
            return self.process_batch(identifiers, output_filename)
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            return {
                'success': False,
                'error': f'入力ファイルが見つかりません: {input_file}'
            }
        except Exception as e:
            logger.error(f"Failed to process file: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """コマンドライン実行用メイン関数"""
    parser = argparse.ArgumentParser(
        description="arXiv論文から9項目を自動抽出しCSV出力",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一論文処理
  python -m src.arxiv_extractor --id 2301.12345
  python -m src.arxiv_extractor --url https://arxiv.org/abs/2301.12345
  python -m src.arxiv_extractor --pdf paper.pdf
  
  # バッチ処理
  python -m src.arxiv_extractor --batch paper_list.txt
  python -m src.arxiv_extractor --ids 2301.12345,2301.12346,2301.12347
  
  # 設定ファイル指定
  python -m src.arxiv_extractor --id 2301.12345 --config config/custom_config.yaml
        """
    )
    
    # 入力オプション（排他的）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--id', help='arXiv ID (例: 2301.12345)')
    input_group.add_argument('--url', help='arXiv URL (例: https://arxiv.org/abs/2301.12345)')
    input_group.add_argument('--pdf', help='PDFファイルパス')
    input_group.add_argument('--batch', help='論文IDリストファイル（1行1ID）')
    input_group.add_argument('--ids', help='カンマ区切りの論文IDリスト')
    
    # その他のオプション
    parser.add_argument('--config', '-c', default='config/config.yaml', 
                       help='設定ファイルパス（デフォルト: config/config.yaml）')
    parser.add_argument('--output', '-o', help='出力ファイル名')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細ログ出力')
    parser.add_argument('--quiet', '-q', action='store_true', help='エラーのみ出力')
    
    args = parser.parse_args()
    
    # ログレベル調整
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    elif args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    
    try:
        # Extractorを初期化
        extractor = ArxivExtractor(args.config)
        
        # 入力に応じて処理実行
        if args.id:
            result = extractor.process_single_paper(args.id, args.output)
        elif args.url:
            result = extractor.process_single_paper(args.url, args.output)
        elif args.pdf:
            result = extractor.process_single_paper(args.pdf, args.output)
        elif args.batch:
            result = extractor.process_from_file(args.batch, args.output)
        elif args.ids:
            identifiers = [id.strip() for id in args.ids.split(',') if id.strip()]
            result = extractor.process_batch(identifiers, args.output)
        
        # 結果出力
        if result['success']:
            print(f"✅ 処理が完了しました")
            if 'csv_path' in result and result['csv_path']:
                print(f"📄 出力ファイル: {result['csv_path']}")
            
            if 'success_count' in result:
                print(f"📊 成功率: {result['success_count']}/{result['total_count']}")
            
            processing_time = result.get('processing_time', result.get('total_processing_time', 0))
            print(f"⏱️  処理時間: {processing_time:.2f}秒")
            
            sys.exit(0)
        else:
            print(f"❌ 処理に失敗しました: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  処理が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ 予期しないエラーが発生しました: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()