"""
テキスト後処理モジュール
OCR結果とReference削除後のテキストを最適化
"""

import re
import unicodedata
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
import spacy
from spacy.lang.en import English

@dataclass
class ProcessedText:
    """処理済みテキストの結果"""
    text: str
    sections: Dict[str, str]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]
    quality_score: float
    processing_notes: List[str]

class TextNormalizer:
    """テキスト正規化クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.remove_page_numbers = config.get('remove_page_numbers', True)
        self.remove_headers_footers = config.get('remove_headers_footers', True)
        self.normalize_whitespace = config.get('normalize_whitespace', True)
        
    def normalize(self, text: str) -> str:
        """テキストを正規化"""
        
        logger.debug("Starting text normalization")
        
        # Unicode正規化
        text = unicodedata.normalize('NFKC', text)
        
        # OCR特有のエラーを修正
        text = self._fix_ocr_errors(text)
        
        # ページ番号を削除
        if self.remove_page_numbers:
            text = self._remove_page_numbers(text)
        
        # ヘッダー・フッターを削除
        if self.remove_headers_footers:
            text = self._remove_headers_footers(text)
        
        # 空白を正規化
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # 不要な文字を削除
        text = self._remove_unwanted_chars(text)
        
        logger.debug("Text normalization completed")
        return text
    
    def _fix_ocr_errors(self, text: str) -> str:
        """OCR特有のエラーを修正"""
        
        # よくあるOCRエラーのパターン
        ocr_fixes = [
            # 文字置換
            (r'(?<=[a-z])I(?=[a-z])', 'l'),  # I -> l
            (r'(?<=[a-z])O(?=[a-z])', 'o'),  # O -> o
            (r'(?<=[a-z])0(?=[a-z])', 'o'),  # 0 -> o
            (r'(?<=[A-Z])l(?=[A-Z])', 'I'),  # l -> I
            (r'(?<=[A-Z])o(?=[A-Z])', 'O'),  # o -> O
            
            # 単語境界の修正
            (r'(?<=[a-z])(?=[A-Z])', ' '),   # 単語境界の追加
            (r'(?<=[.!?])(?=[A-Z])', ' '),   # 文末の空白追加
            
            # 不適切な改行の修正
            (r'(?<=[a-z])\n(?=[a-z])', ' '),  # 単語中の改行を空白に
            (r'(?<=[a-z])-\n(?=[a-z])', ''),  # ハイフンでの改行を結合
            
            # 数式の修正
            (r'(?<=\d)\s*x\s*(?=\d)', '×'),   # x -> ×
            (r'(?<=\d)\s*X\s*(?=\d)', '×'),   # X -> ×
        ]
        
        for pattern, replacement in ocr_fixes:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _remove_page_numbers(self, text: str) -> str:
        """ページ番号を削除"""
        
        # ページ番号パターン
        page_patterns = [
            r'^\s*\d+\s*$',  # 単独の数字
            r'^\s*Page\s+\d+\s*$',  # Page 1
            r'^\s*\d+\s*/\s*\d+\s*$',  # 1/10
            r'^\s*-\s*\d+\s*-\s*$',  # - 1 -
            r'^\s*\[\s*\d+\s*\]\s*$',  # [1]
        ]
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            is_page_number = any(re.match(pattern, line, re.IGNORECASE) 
                               for pattern in page_patterns)
            if not is_page_number:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _remove_headers_footers(self, text: str) -> str:
        """ヘッダー・フッターを削除"""
        
        lines = text.split('\n')
        if len(lines) < 10:
            return text
        
        # ヘッダー・フッターの候補を抽出
        header_candidates = self._extract_header_footer_candidates(lines[:5])
        footer_candidates = self._extract_header_footer_candidates(lines[-5:])
        
        # 繰り返し出現するパターンを削除
        filtered_lines = []
        for line in lines:
            if (not self._is_header_footer_line(line, header_candidates) and
                not self._is_header_footer_line(line, footer_candidates)):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _extract_header_footer_candidates(self, lines: List[str]) -> List[str]:
        """ヘッダー・フッターの候補を抽出"""
        
        candidates = []
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:  # 短い行のみ
                # 一般的なヘッダー・フッターパターン
                if any(keyword in line.lower() for keyword in 
                       ['arxiv', 'preprint', 'submitted', 'journal', 'conference']):
                    candidates.append(line)
        
        return candidates
    
    def _is_header_footer_line(self, line: str, candidates: List[str]) -> bool:
        """行がヘッダー・フッターかどうかを判定"""
        
        line = line.strip()
        if not line:
            return False
        
        # 候補との類似度をチェック
        for candidate in candidates:
            if self._similarity(line, candidate) > 0.8:
                return True
        
        return False
    
    def _similarity(self, s1: str, s2: str) -> float:
        """文字列の類似度を計算"""
        
        if not s1 or not s2:
            return 0.0
        
        # 簡単な類似度計算
        common = set(s1.lower().split()) & set(s2.lower().split())
        total = set(s1.lower().split()) | set(s2.lower().split())
        
        if not total:
            return 0.0
        
        return len(common) / len(total)
    
    def _normalize_whitespace(self, text: str) -> str:
        """空白を正規化"""
        
        # 連続する空白を単一に
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 連続する改行を最大2つに
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 行頭・行末の空白を削除
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # 前後の空白を削除
        text = text.strip()
        
        return text
    
    def _remove_unwanted_chars(self, text: str) -> str:
        """不要な文字を削除"""
        
        # 制御文字を削除（改行・タブ以外）
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 不可視文字を削除
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        
        return text

class SectionExtractor:
    """セクション抽出クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # セクション見出しパターン
        self.section_patterns = [
            r'^\s*(?:ABSTRACT|Abstract)\s*$',
            r'^\s*(?:INTRODUCTION|Introduction)\s*$',
            r'^\s*(?:RELATED\s+WORK|Related\s+Work)\s*$',
            r'^\s*(?:METHODOLOGY|Methodology|METHOD|Method)\s*$',
            r'^\s*(?:APPROACH|Approach)\s*$',
            r'^\s*(?:EXPERIMENTS?|Experiments?)\s*$',
            r'^\s*(?:RESULTS?|Results?)\s*$',
            r'^\s*(?:EVALUATION|Evaluation)\s*$',
            r'^\s*(?:DISCUSSION|Discussion)\s*$',
            r'^\s*(?:CONCLUSION|Conclusion)\s*$',
            r'^\s*(?:FUTURE\s+WORK|Future\s+Work)\s*$',
            r'^\s*\d+\.?\s+[A-Z][a-z]+.*$',  # 番号付き見出し
        ]
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """テキストからセクションを抽出"""
        
        logger.debug("Extracting sections from text")
        
        sections = {}
        lines = text.split('\n')
        
        current_section = 'main'
        current_lines = []
        
        for line in lines:
            # セクション見出しかチェック
            section_name = self._detect_section_header(line)
            
            if section_name:
                # 前のセクションを保存
                if current_lines:
                    sections[current_section] = '\n'.join(current_lines).strip()
                
                # 新しいセクションを開始
                current_section = section_name
                current_lines = []
            else:
                current_lines.append(line)
        
        # 最後のセクションを保存
        if current_lines:
            sections[current_section] = '\n'.join(current_lines).strip()
        
        logger.debug(f"Extracted {len(sections)} sections")
        return sections
    
    def _detect_section_header(self, line: str) -> Optional[str]:
        """セクション見出しを検出"""
        
        for pattern in self.section_patterns:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                # 見出しを正規化
                normalized = re.sub(r'^\s*\d+\.?\s*', '', line.strip())
                normalized = normalized.lower().replace(' ', '_')
                return normalized
        
        return None

class QualityAnalyzer:
    """テキスト品質分析クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.nlp = None
        self._init_nlp()
    
    def _init_nlp(self):
        """NLPモデルを初期化"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found")
            try:
                logger.info("Attempting to download spaCy model...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("spaCy model downloaded and loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to download spaCy model: {e}. Using basic English model")
                self.nlp = English()
    
    def analyze_quality(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """テキストの品質を分析"""
        
        if not text.strip():
            return 0.0, {}
        
        # 基本統計
        stats = self._calculate_basic_stats(text)
        
        # 言語品質
        language_score = self._analyze_language_quality(text)
        
        # 構造品質
        structure_score = self._analyze_structure_quality(text)
        
        # 内容品質
        content_score = self._analyze_content_quality(text)
        
        # 総合スコア
        total_score = (language_score * 0.3 + 
                      structure_score * 0.3 + 
                      content_score * 0.4)
        
        analysis = {
            'basic_stats': stats,
            'language_score': language_score,
            'structure_score': structure_score,
            'content_score': content_score,
            'total_score': total_score
        }
        
        return total_score, analysis
    
    def _calculate_basic_stats(self, text: str) -> Dict[str, Any]:
        """基本統計を計算"""
        
        # 基本的な統計情報
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.split('\n'))
        
        # 文の数
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # 段落数
        paragraphs = re.split(r'\n\s*\n', text)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'line_count': line_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_words_per_sentence': word_count / max(sentence_count, 1),
            'avg_sentences_per_paragraph': sentence_count / max(paragraph_count, 1)
        }
    
    def _analyze_language_quality(self, text: str) -> float:
        """言語品質を分析"""
        
        score = 0.0
        
        # 語彙の多様性
        words = text.lower().split()
        if words:
            unique_words = set(words)
            vocabulary_diversity = len(unique_words) / len(words)
            score += min(vocabulary_diversity * 2, 1.0) * 0.3
        
        # 文法エラーの推定
        if self.nlp:
            doc = self.nlp(text[:1000])  # 最初の1000文字で分析
            
            # 品詞の多様性
            pos_tags = [token.pos_ for token in doc if token.is_alpha]
            if pos_tags:
                pos_diversity = len(set(pos_tags)) / len(pos_tags)
                score += min(pos_diversity * 2, 1.0) * 0.3
            
            # 名詞・動詞の比率
            nouns = sum(1 for token in doc if token.pos_ in ['NOUN', 'PROPN'])
            verbs = sum(1 for token in doc if token.pos_ == 'VERB')
            total_content_words = nouns + verbs
            if total_content_words > 0:
                content_ratio = total_content_words / len([t for t in doc if t.is_alpha])
                score += min(content_ratio * 2, 1.0) * 0.4
        
        return min(score, 1.0)
    
    def _analyze_structure_quality(self, text: str) -> float:
        """構造品質を分析"""
        
        score = 0.0
        
        # 段落構造
        paragraphs = re.split(r'\n\s*\n', text)
        meaningful_paragraphs = [p for p in paragraphs if len(p.strip()) > 50]
        
        if meaningful_paragraphs:
            # 段落の長さの均等性
            lengths = [len(p) for p in meaningful_paragraphs]
            if len(lengths) > 1:
                mean_length = statistics.mean(lengths)
                if mean_length > 0:  # Division by zero check
                    std_length = statistics.stdev(lengths)
                    consistency = 1 - min(std_length / mean_length, 1.0)
                    score += consistency * 0.3
                else:
                    score += 0.3
            else:
                score += 0.3
        
        # 文の長さの多様性
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            if 10 <= avg_length <= 25:  # 理想的な文の長さ
                score += 0.3
            else:
                score += max(0, 0.3 - abs(avg_length - 17.5) / 50)
        
        # セクション構造の存在
        section_headers = sum(1 for line in text.split('\n') 
                            if re.match(r'^\s*(?:[A-Z][A-Z\s]*|[IVX]+\.|\d+\.)', line.strip()))
        if section_headers >= 3:
            score += 0.4
        elif section_headers >= 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_content_quality(self, text: str) -> float:
        """内容品質を分析"""
        
        score = 0.0
        
        # 学術的キーワードの存在
        academic_keywords = [
            'research', 'study', 'analysis', 'method', 'approach', 'experiment',
            'results', 'conclusion', 'propose', 'demonstrate', 'evaluate',
            'performance', 'algorithm', 'model', 'framework', 'system'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in academic_keywords if keyword in text_lower)
        keyword_score = min(keyword_count / 10, 1.0)
        score += keyword_score * 0.3
        
        # 数値・統計情報の存在
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        if len(numbers) >= 5:
            score += 0.2
        elif len(numbers) >= 2:
            score += 0.1
        
        # 引用・参照の痕跡（削除後も残るもの）
        citations = re.findall(r'(?:as shown in|according to|reported by|following)', text, re.IGNORECASE)
        if citations:
            score += min(len(citations) / 5, 0.2)
        
        # 図表への言及
        figure_refs = re.findall(r'(?:figure|table|fig\.|tab\.)\s*\d+', text, re.IGNORECASE)
        if figure_refs:
            score += min(len(figure_refs) / 3, 0.3)
        
        return min(score, 1.0)

class TextProcessor:
    """テキスト処理メインクラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.normalizer = TextNormalizer(config)
        self.section_extractor = SectionExtractor(config)
        self.quality_analyzer = QualityAnalyzer(config)
        
        logger.info("TextProcessor initialized")
    
    def process_text(self, text: str) -> ProcessedText:
        """テキストを包括的に処理"""
        
        logger.info("Starting comprehensive text processing")
        
        processing_notes = []
        
        # Step 1: テキスト正規化
        normalized_text = self.normalizer.normalize(text)
        if len(normalized_text) < len(text) * 0.8:
            processing_notes.append("Significant text reduction during normalization")
        
        # Step 2: セクション抽出
        sections = self.section_extractor.extract_sections(normalized_text)
        processing_notes.append(f"Extracted {len(sections)} sections")
        
        # Step 3: 品質分析
        quality_score, quality_analysis = self.quality_analyzer.analyze_quality(normalized_text)
        processing_notes.append(f"Quality score: {quality_score:.3f}")
        
        # Step 4: メタデータ作成
        metadata = {
            'original_length': len(text),
            'processed_length': len(normalized_text),
            'compression_ratio': len(normalized_text) / len(text) if text else 0,
            'sections_found': list(sections.keys()),
            'quality_analysis': quality_analysis
        }
        
        logger.info(f"Text processing completed. Quality score: {quality_score:.3f}")
        
        return ProcessedText(
            text=normalized_text,
            sections=sections,
            metadata=metadata,
            statistics=quality_analysis.get('basic_stats', {}),
            quality_score=quality_score,
            processing_notes=processing_notes
        )

def create_text_processor(config: Dict) -> TextProcessor:
    """
    設定からTextProcessorを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        TextProcessor インスタンス
    """
    return TextProcessor(config.get('text_processing', {}))