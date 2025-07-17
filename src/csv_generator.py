"""
CSV出力モジュール
抽出結果をCSV形式で出力・管理
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class CSVOutputConfig:
    """CSV出力設定"""
    encoding: str = "utf-8"
    delimiter: str = ","
    quotechar: str = '"'
    quoting: str = "minimal"  # minimal, all, non_numeric, none
    lineterminator: str = "\n"
    
    def get_quoting_constant(self) -> int:
        """quoting文字列を定数に変換"""
        quoting_map = {
            "minimal": csv.QUOTE_MINIMAL,
            "all": csv.QUOTE_ALL,
            "non_numeric": csv.QUOTE_NONNUMERIC,
            "none": csv.QUOTE_NONE
        }
        return quoting_map.get(self.quoting, csv.QUOTE_MINIMAL)


@dataclass
class ExtractionRecord:
    """抽出レコード"""
    # 9つの必須項目
    手法の肝: str = "N/A"
    制限事項: str = "N/A"
    対象ナレッジ: str = "N/A"
    URL: str = "N/A"
    タイトル: str = "N/A"
    出版年: str = "N/A"
    研究分野: str = "N/A"
    課題設定: str = "N/A"
    論文の主張: str = "N/A"
    
    # メタデータ（オプション）
    処理日時: str = ""
    信頼度スコア: str = ""
    使用モデル: str = ""
    処理時間: str = ""
    エラー情報: str = ""
    
    def __post_init__(self):
        if not self.処理日時:
            self.処理日時 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @classmethod
    def get_header(cls, include_metadata: bool = True) -> List[str]:
        """CSVヘッダーを取得"""
        core_fields = [
            "手法の肝", "制限事項", "対象ナレッジ", "URL", "タイトル", 
            "出版年", "研究分野", "課題設定", "論文の主張"
        ]
        
        if include_metadata:
            metadata_fields = ["処理日時", "信頼度スコア", "使用モデル", "処理時間", "エラー情報"]
            return core_fields + metadata_fields
        
        return core_fields
    
    def to_row(self, include_metadata: bool = True) -> List[str]:
        """CSVの行データに変換"""
        record_dict = asdict(self)
        header = self.get_header(include_metadata)
        return [record_dict.get(field, "") for field in header]
    
    @classmethod
    def from_extraction_result(
        cls, 
        extraction_result, 
        processing_time: float = 0.0,
        errors: Optional[List[str]] = None
    ) -> 'ExtractionRecord':
        """抽出結果からレコードを作成"""
        
        # 抽出データを取得
        data = extraction_result.data if hasattr(extraction_result, 'data') else {}
        
        # 信頼度スコアの計算
        confidence_scores = getattr(extraction_result, 'confidence_scores', {})
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
        
        # エラー情報の処理
        error_list = errors or getattr(extraction_result, 'errors', [])
        error_str = "; ".join(error_list) if error_list else ""
        
        return cls(
            手法の肝=data.get("手法の肝", "N/A"),
            制限事項=data.get("制限事項", "N/A"),
            対象ナレッジ=data.get("対象ナレッジ", "N/A"),
            URL=data.get("URL", "N/A"),
            タイトル=data.get("タイトル", "N/A"),
            出版年=data.get("出版年", "N/A"),
            研究分野=data.get("研究分野", "N/A"),
            課題設定=data.get("課題設定", "N/A"),
            論文の主張=data.get("論文の主張", "N/A"),
            信頼度スコア=f"{avg_confidence:.3f}",
            使用モデル=getattr(extraction_result, 'model_used', 'unknown'),
            処理時間=f"{processing_time:.2f}s",
            エラー情報=error_str
        )


class CSVGenerator:
    """CSV生成・管理クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_config = self._load_output_config(config)
        self.output_dir = Path(config.get('output_dir', './output'))
        self.filename_patterns = config.get('filename_patterns', {})
        self.default_values = config.get('default_values', {})
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CSVGenerator initialized with output directory: {self.output_dir}")
    
    def _load_output_config(self, config: Dict) -> CSVOutputConfig:
        """出力設定をロード"""
        csv_config = config.get('csv', {})
        
        return CSVOutputConfig(
            encoding=csv_config.get('encoding', 'utf-8'),
            delimiter=csv_config.get('delimiter', ','),
            quotechar=csv_config.get('quotechar', '"'),
            quoting=csv_config.get('quoting', 'minimal'),
            lineterminator=csv_config.get('lineterminator', '\n')
        )
    
    def generate_filename(self, mode: str = "single", custom_name: Optional[str] = None) -> str:
        """ファイル名を生成"""
        if custom_name:
            return custom_name if custom_name.endswith('.csv') else f"{custom_name}.csv"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if mode == "batch":
            pattern = self.filename_patterns.get('batch', 'arxiv_batch_{timestamp}.csv')
        else:
            pattern = self.filename_patterns.get('single', 'arxiv_extraction_{timestamp}.csv')
        
        return pattern.format(timestamp=timestamp)
    
    def write_single_record(
        self, 
        extraction_result,
        filename: Optional[str] = None,
        processing_time: float = 0.0,
        include_metadata: bool = True
    ) -> str:
        """単一レコードをCSVファイルに書き込み"""
        
        # ファイル名生成
        if not filename:
            filename = self.generate_filename("single")
        
        filepath = self.output_dir / filename
        
        try:
            # レコード作成
            record = ExtractionRecord.from_extraction_result(
                extraction_result, 
                processing_time
            )
            
            # CSV書き込み
            with open(filepath, 'w', newline='', encoding=self.output_config.encoding) as csvfile:
                writer = csv.writer(
                    csvfile,
                    delimiter=self.output_config.delimiter,
                    quotechar=self.output_config.quotechar,
                    quoting=self.output_config.get_quoting_constant(),
                    lineterminator=self.output_config.lineterminator
                )
                
                # ヘッダー書き込み
                writer.writerow(record.get_header(include_metadata))
                
                # データ書き込み
                writer.writerow(record.to_row(include_metadata))
            
            logger.info(f"Single record written to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to write single record: {str(e)}")
            raise
    
    def write_batch_records(
        self, 
        extraction_results: List,
        filename: Optional[str] = None,
        processing_times: Optional[List[float]] = None,
        include_metadata: bool = True
    ) -> str:
        """複数レコードをCSVファイルに書き込み"""
        
        if not extraction_results:
            raise ValueError("No extraction results provided")
        
        # ファイル名生成
        if not filename:
            filename = self.generate_filename("batch")
        
        filepath = self.output_dir / filename
        
        # 処理時間のデフォルト値
        if not processing_times:
            processing_times = [0.0] * len(extraction_results)
        
        try:
            with open(filepath, 'w', newline='', encoding=self.output_config.encoding) as csvfile:
                writer = csv.writer(
                    csvfile,
                    delimiter=self.output_config.delimiter,
                    quotechar=self.output_config.quotechar,
                    quoting=self.output_config.get_quoting_constant(),
                    lineterminator=self.output_config.lineterminator
                )
                
                # ヘッダー書き込み（最初のレコードから取得）
                first_record = ExtractionRecord.from_extraction_result(
                    extraction_results[0], 
                    processing_times[0]
                )
                writer.writerow(first_record.get_header(include_metadata))
                
                # 各レコードを書き込み
                for i, result in enumerate(extraction_results):
                    processing_time = processing_times[i] if i < len(processing_times) else 0.0
                    
                    record = ExtractionRecord.from_extraction_result(
                        result, 
                        processing_time
                    )
                    writer.writerow(record.to_row(include_metadata))
            
            logger.info(f"Batch records ({len(extraction_results)}) written to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to write batch records: {str(e)}")
            raise
    
    def append_record(
        self, 
        extraction_result,
        filename: str,
        processing_time: float = 0.0,
        include_metadata: bool = True
    ) -> str:
        """既存CSVファイルにレコードを追加"""
        
        filepath = self.output_dir / filename
        
        try:
            # ファイルが存在しない場合は新規作成
            if not filepath.exists():
                return self.write_single_record(
                    extraction_result, 
                    filename, 
                    processing_time, 
                    include_metadata
                )
            
            # レコード作成
            record = ExtractionRecord.from_extraction_result(
                extraction_result, 
                processing_time
            )
            
            # 追記モードで書き込み
            with open(filepath, 'a', newline='', encoding=self.output_config.encoding) as csvfile:
                writer = csv.writer(
                    csvfile,
                    delimiter=self.output_config.delimiter,
                    quotechar=self.output_config.quotechar,
                    quoting=self.output_config.get_quoting_constant(),
                    lineterminator=self.output_config.lineterminator
                )
                
                writer.writerow(record.to_row(include_metadata))
            
            logger.info(f"Record appended to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to append record: {str(e)}")
            raise
    
    def read_csv(self, filename: str) -> List[Dict[str, str]]:
        """CSVファイルを読み込み"""
        
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            records = []
            with open(filepath, 'r', encoding=self.output_config.encoding) as csvfile:
                reader = csv.DictReader(
                    csvfile,
                    delimiter=self.output_config.delimiter,
                    quotechar=self.output_config.quotechar
                )
                
                for row in reader:
                    records.append(dict(row))
            
            logger.info(f"Read {len(records)} records from: {filepath}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to read CSV: {str(e)}")
            raise
    
    def get_statistics(self, filename: str) -> Dict[str, Any]:
        """CSVファイルの統計情報を取得"""
        
        try:
            records = self.read_csv(filename)
            
            if not records:
                return {"record_count": 0}
            
            # 基本統計
            stats = {
                "record_count": len(records),
                "completion_rate": {},
                "avg_confidence": 0.0,
                "model_usage": {},
                "error_rate": 0.0
            }
            
            # 完了率計算（N/A以外の項目の割合）
            core_fields = ExtractionRecord.get_header(include_metadata=False)
            for field in core_fields:
                if field in records[0]:  # フィールドが存在する場合
                    non_na_count = sum(1 for record in records 
                                     if record.get(field, "N/A") != "N/A" and record.get(field, "N/A").strip())
                    stats["completion_rate"][field] = non_na_count / len(records)
            
            # 信頼度スコア統計
            if "信頼度スコア" in records[0]:
                confidence_scores = []
                for record in records:
                    try:
                        score = float(record.get("信頼度スコア", "0"))
                        confidence_scores.append(score)
                    except ValueError:
                        pass
                
                if confidence_scores:
                    stats["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
            
            # モデル使用統計
            if "使用モデル" in records[0]:
                for record in records:
                    model = record.get("使用モデル", "unknown")
                    stats["model_usage"][model] = stats["model_usage"].get(model, 0) + 1
            
            # エラー率
            if "エラー情報" in records[0]:
                error_count = sum(1 for record in records 
                                if record.get("エラー情報", "").strip())
                stats["error_rate"] = error_count / len(records)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {str(e)}")
            return {"error": str(e)}
    
    def convert_to_json(self, csv_filename: str, json_filename: Optional[str] = None) -> str:
        """CSVをJSONに変換"""
        
        if not json_filename:
            json_filename = csv_filename.replace('.csv', '.json')
        
        csv_filepath = self.output_dir / csv_filename
        json_filepath = self.output_dir / json_filename
        
        try:
            records = self.read_csv(csv_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(records, jsonfile, ensure_ascii=False, indent=2)
            
            logger.info(f"CSV converted to JSON: {json_filepath}")
            return str(json_filepath)
            
        except Exception as e:
            logger.error(f"Failed to convert CSV to JSON: {str(e)}")
            raise
    
    def list_output_files(self) -> List[Dict[str, Any]]:
        """出力ディレクトリ内のファイル一覧を取得"""
        
        try:
            files = []
            for filepath in self.output_dir.glob("*.csv"):
                stats = filepath.stat()
                
                file_info = {
                    "filename": filepath.name,
                    "full_path": str(filepath),
                    "size_bytes": stats.st_size,
                    "created": datetime.fromtimestamp(stats.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                    "modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # レコード数を取得（エラーが発生しても続行）
                try:
                    records = self.read_csv(filepath.name)
                    file_info["record_count"] = len(records)
                except:
                    file_info["record_count"] = "unknown"
                
                files.append(file_info)
            
            # 更新日時でソート
            files.sort(key=lambda x: x["modified"], reverse=True)
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list output files: {str(e)}")
            return []
    
    def validate_csv(self, filename: str) -> Dict[str, Any]:
        """CSVファイルの妥当性を検証"""
        
        try:
            records = self.read_csv(filename)
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "record_count": len(records)
            }
            
            if not records:
                validation_result["errors"].append("ファイルが空です")
                validation_result["valid"] = False
                return validation_result
            
            # ヘッダー検証
            expected_headers = ExtractionRecord.get_header(include_metadata=False)
            actual_headers = list(records[0].keys())
            
            missing_headers = set(expected_headers) - set(actual_headers)
            if missing_headers:
                validation_result["errors"].append(f"必須ヘッダーが不足: {missing_headers}")
                validation_result["valid"] = False
            
            # データ検証
            empty_url_count = sum(1 for record in records if not record.get("URL", "").strip() or record.get("URL") == "N/A")
            if empty_url_count > 0:
                validation_result["warnings"].append(f"URLが空のレコード: {empty_url_count}件")
            
            empty_title_count = sum(1 for record in records if not record.get("タイトル", "").strip() or record.get("タイトル") == "N/A")
            if empty_title_count > 0:
                validation_result["warnings"].append(f"タイトルが空のレコード: {empty_title_count}件")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"検証エラー: {str(e)}"],
                "warnings": [],
                "record_count": 0
            }


def create_csv_generator(config: Dict) -> CSVGenerator:
    """
    設定からCSVGeneratorを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        CSVGenerator インスタンス
    """
    return CSVGenerator(config.get('output', {}))