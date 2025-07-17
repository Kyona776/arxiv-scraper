"""
Reference削除モジュール
論文のReference部分を高精度で検出・削除する
"""

import re
import statistics
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from loguru import logger
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# NLTK データのダウンロード（初回のみ）
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

@dataclass
class ReferenceDetectionResult:
    """Reference検出結果"""
    reference_start: int
    reference_end: int
    confidence: float
    detected_patterns: List[str]
    removed_text: str
    cleaned_text: str
    section_title: Optional[str] = None

class ReferencePatternDetector:
    """Referenceパターン検出器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reference_patterns = config.get('reference_patterns', [])
        self.citation_patterns = config.get('citation_patterns', [])
        
        # 基本パターンを追加
        self._init_patterns()
    
    def _init_patterns(self):
        """検出パターンを初期化"""
        
        # Reference セクション見出しパターン
        self.section_patterns = [
            r'\n\s*REFERENCES?\s*\n',
            r'\n\s*References?\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
            r'\n\s*Bibliography\s*\n',
            r'\n\s*参考文献\s*\n',
            r'\n\s*文献\s*\n',
            r'\n\s*LITERATURE\s+CITED\s*\n',
            r'\n\s*Literature\s+Cited\s*\n',
            r'\n\s*CITATIONS?\s*\n',
            r'\n\s*Citations?\s*\n',
        ]
        
        # 参考文献エントリパターン
        self.reference_entry_patterns = [
            # IEEE形式: [1] Author, A. ...
            r'\[\s*\d+\s*\]\s*[A-Z][a-z]+(?:\s*,\s*[A-Z]\.)*',
            
            # APA形式: Author, A. (2023). Title...
            r'[A-Z][a-z]+\s*,\s*[A-Z]\.\s*(?:\([12]\d{3}[a-z]?\)\s*\.)?',
            
            # Nature形式: 1. Author, A. Title...
            r'^\s*\d+\.\s*[A-Z][a-z]+\s*,\s*[A-Z]\.',
            
            # 著者名 + 年パターン
            r'[A-Z][a-z]+(?:\s*,\s*[A-Z]\.)*\s*(?:et\s+al\.?)?\s*\([12]\d{3}[a-z]?\)',
            
            # arXiv形式
            r'arXiv:\d{4}\.\d{4,5}(?:v\d+)?',
            
            # DOI パターン
            r'doi:\s*10\.\d+/[^\s]+',
            r'https?://doi\.org/10\.\d+/[^\s]+',
            
            # URL パターン
            r'https?://[^\s]+',
            
            # 日本語文献パターン
            r'[ぁ-ん]+\s*[ァ-ヴ]*\s*[一-龯]+.*(?:[12]\d{3}|平成\d+|昭和\d+)',
        ]
        
        # 引用パターン（本文中から削除）
        self.inline_citation_patterns = [
            r'\[(\d+(?:[-–,]\s*\d+)*)\]',  # [1], [1-3], [1,2,3]
            r'\(([A-Za-z]+\s+(?:et\s+al\.?\s*)?(?:19|20)\d{2}[a-z]?(?:;\s*[A-Za-z]+\s+(?:et\s+al\.?\s*)?(?:19|20)\d{2}[a-z]?)*)\)',
            r'\((?:19|20)\d{2}[a-z]?\)',
            r'\([A-Za-z]+\s*(?:et\s+al\.?)?\s*,\s*(?:19|20)\d{2}[a-z]?\)',
        ]
    
    def find_reference_sections(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Reference セクションを検出
        
        Returns:
            List of (start_pos, end_pos, section_title)
        """
        sections = []
        
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                start = match.start()
                title = match.group().strip()
                
                # セクション終了位置を推定
                end = self._find_section_end(text, start)
                
                sections.append((start, end, title))
        
        return sections
    
    def _find_section_end(self, text: str, start: int) -> int:
        """セクション終了位置を推定"""
        
        # 次のセクション見出しを探す
        next_section_patterns = [
            r'\n\s*(?:APPENDIX|Appendix|付録)\s*\n',
            r'\n\s*(?:ACKNOWLEDGMENTS?|Acknowledgments?|謝辞)\s*\n',
            r'\n\s*(?:AUTHOR|Author|著者).*(?:INFORMATION|Information|情報)\s*\n',
            r'\n\s*(?:FUNDING|Funding|資金)\s*\n',
            r'\n\s*(?:CONFLICTS?|Conflicts?|利害関係)\s*\n',
        ]
        
        min_end = len(text)
        for pattern in next_section_patterns:
            match = re.search(pattern, text[start:], re.IGNORECASE)
            if match:
                min_end = min(min_end, start + match.start())
        
        return min_end
    
    def count_reference_indicators(self, text: str) -> Dict[str, int]:
        """Reference指標をカウント"""
        indicators = {}
        
        # 番号付きリストの数
        numbered_lists = re.findall(r'^\s*\d+\.\s', text, re.MULTILINE)
        indicators['numbered_lists'] = len(numbered_lists)
        
        # 角括弧の数
        bracketed_numbers = re.findall(r'\[\s*\d+\s*\]', text)
        indicators['bracketed_numbers'] = len(bracketed_numbers)
        
        # 著者名パターンの数
        author_patterns = re.findall(r'[A-Z][a-z]+\s*,\s*[A-Z]\.', text)
        indicators['author_patterns'] = len(author_patterns)
        
        # 年パターンの数
        year_patterns = re.findall(r'\([12]\d{3}[a-z]?\)', text)
        indicators['year_patterns'] = len(year_patterns)
        
        # DOI/URLの数
        doi_patterns = re.findall(r'doi:|https?://', text)
        indicators['doi_url_patterns'] = len(doi_patterns)
        
        return indicators
    
    def is_reference_section(self, text: str, threshold: float = 0.3) -> bool:
        """
        テキストがReference セクションかどうかを判定
        
        Args:
            text: 判定対象テキスト
            threshold: 判定閾値（0-1）
            
        Returns:
            Reference セクションかどうか
        """
        if not text.strip():
            return False
        
        indicators = self.count_reference_indicators(text)
        
        # 長さで正規化
        text_length = len(text)
        if text_length == 0:
            return False
        
        # スコア計算
        score = 0.0
        
        # 番号付きリストの密度
        if indicators['numbered_lists'] > 0:
            score += min(indicators['numbered_lists'] / (text_length / 1000), 0.3)
        
        # 角括弧の密度
        if indicators['bracketed_numbers'] > 0:
            score += min(indicators['bracketed_numbers'] / (text_length / 1000), 0.2)
        
        # 著者名パターンの密度
        if indicators['author_patterns'] > 0:
            score += min(indicators['author_patterns'] / (text_length / 1000), 0.2)
        
        # 年パターンの密度
        if indicators['year_patterns'] > 0:
            score += min(indicators['year_patterns'] / (text_length / 1000), 0.15)
        
        # DOI/URLの密度
        if indicators['doi_url_patterns'] > 0:
            score += min(indicators['doi_url_patterns'] / (text_length / 1000), 0.15)
        
        logger.debug(f"Reference section score: {score}, threshold: {threshold}")
        return score >= threshold

class ReferenceCleaner:
    """Reference削除メインクラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.remove_references = config.get('remove_references', True)
        self.citation_cleanup = config.get('citation_cleanup', True)
        self.min_text_length = config.get('min_text_length', 1000)
        self.max_reference_ratio = config.get('max_reference_ratio', 0.3)
        
        # パターン検出器を初期化
        self.pattern_detector = ReferencePatternDetector(config)
        
        # 停止語を初期化
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> ReferenceDetectionResult:
        """
        テキストからReferenceを削除
        
        Args:
            text: 入力テキスト
            
        Returns:
            ReferenceDetectionResult
        """
        logger.info("Starting reference cleaning process")
        
        original_length = len(text)
        
        if not self.remove_references:
            logger.info("Reference removal disabled")
            return ReferenceDetectionResult(
                reference_start=-1,
                reference_end=-1,
                confidence=0.0,
                detected_patterns=[],
                removed_text="",
                cleaned_text=text
            )
        
        # Step 1: Reference セクションを検出
        reference_sections = self.pattern_detector.find_reference_sections(text)
        
        if not reference_sections:
            logger.info("No reference sections found")
            cleaned_text = self._clean_inline_citations(text) if self.citation_cleanup else text
            return ReferenceDetectionResult(
                reference_start=-1,
                reference_end=-1,
                confidence=0.0,
                detected_patterns=[],
                removed_text="",
                cleaned_text=cleaned_text
            )
        
        # Step 2: 最も可能性の高いReference セクションを選択
        best_section = self._select_best_reference_section(text, reference_sections)
        
        if not best_section:
            logger.info("No valid reference section found")
            cleaned_text = self._clean_inline_citations(text) if self.citation_cleanup else text
            return ReferenceDetectionResult(
                reference_start=-1,
                reference_end=-1,
                confidence=0.0,
                detected_patterns=[],
                removed_text="",
                cleaned_text=cleaned_text
            )
        
        start, end, section_title = best_section
        
        # Step 3: 削除する範囲を精密化
        refined_start, refined_end = self._refine_removal_range(text, start, end)
        
        # Step 4: 削除実行
        removed_text = text[refined_start:refined_end]
        cleaned_text = text[:refined_start] + text[refined_end:]
        
        # Step 5: インライン引用を削除
        if self.citation_cleanup:
            cleaned_text = self._clean_inline_citations(cleaned_text)
        
        # Step 6: 品質チェック
        if not self._quality_check(cleaned_text):
            logger.warning("Quality check failed, returning original text")
            return ReferenceDetectionResult(
                reference_start=-1,
                reference_end=-1,
                confidence=0.0,
                detected_patterns=[],
                removed_text="",
                cleaned_text=text
            )
        
        # Step 7: 最終処理
        cleaned_text = self._final_cleanup(cleaned_text)
        
        # 信頼度を計算
        confidence = self._calculate_confidence(text, removed_text, cleaned_text)
        
        # 検出パターンを記録
        detected_patterns = [section_title] if section_title else []
        
        logger.info(f"Reference cleaning completed. Removed {len(removed_text)} characters ({len(removed_text)/original_length*100:.1f}%)")
        
        return ReferenceDetectionResult(
            reference_start=refined_start,
            reference_end=refined_end,
            confidence=confidence,
            detected_patterns=detected_patterns,
            removed_text=removed_text,
            cleaned_text=cleaned_text,
            section_title=section_title
        )
    
    def _select_best_reference_section(
        self, 
        text: str, 
        sections: List[Tuple[int, int, str]]
    ) -> Optional[Tuple[int, int, str]]:
        """最適なReference セクションを選択"""
        
        best_section = None
        best_score = 0.0
        
        for start, end, title in sections:
            section_text = text[start:end]
            
            # Reference セクションかどうかを判定
            if self.pattern_detector.is_reference_section(section_text):
                
                # スコアを計算
                score = self._calculate_section_score(section_text, start, end, len(text))
                
                if score > best_score:
                    best_score = score
                    best_section = (start, end, title)
        
        logger.debug(f"Selected reference section with score: {best_score}")
        return best_section
    
    def _calculate_section_score(self, section_text: str, start: int, end: int, total_length: int) -> float:
        """セクションのスコアを計算"""
        
        # 位置スコア（論文の後半にあるほど高い）
        position_score = start / total_length
        
        # 長さスコア（適度な長さが好ましい）
        length_ratio = len(section_text) / total_length
        if 0.05 <= length_ratio <= 0.25:  # 5-25%の長さが理想
            length_score = 1.0
        elif length_ratio < 0.05:
            length_score = length_ratio / 0.05
        else:
            length_score = max(0.0, 1.0 - (length_ratio - 0.25) / 0.25)
        
        # 指標スコア
        indicators = self.pattern_detector.count_reference_indicators(section_text)
        indicator_score = min(sum(indicators.values()) / len(section_text) * 1000, 1.0)
        
        # 総合スコア
        total_score = position_score * 0.3 + length_score * 0.4 + indicator_score * 0.3
        
        return total_score
    
    def _refine_removal_range(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """削除範囲を精密化"""
        
        # 前方の空白・改行を含める
        refined_start = start
        while refined_start > 0 and text[refined_start - 1] in ' \t\n\r':
            refined_start -= 1
        
        # 後方の空白・改行を含める
        refined_end = end
        while refined_end < len(text) and text[refined_end] in ' \t\n\r':
            refined_end += 1
        
        return refined_start, refined_end
    
    def _clean_inline_citations(self, text: str) -> str:
        """インライン引用を削除"""
        
        cleaned = text
        
        for pattern in self.pattern_detector.inline_citation_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _quality_check(self, cleaned_text: str) -> bool:
        """削除後の品質をチェック"""
        
        # 最小長チェック
        if len(cleaned_text) < self.min_text_length:
            logger.warning(f"Cleaned text too short: {len(cleaned_text)} < {self.min_text_length}")
            return False
        
        # 文章の構造チェック
        sentences = sent_tokenize(cleaned_text)
        if len(sentences) < 10:  # 最小文数
            logger.warning(f"Too few sentences: {len(sentences)}")
            return False
        
        # 単語の多様性チェック
        words = word_tokenize(cleaned_text.lower())
        meaningful_words = [w for w in words if w.isalpha() and w not in self.stop_words]
        if len(set(meaningful_words)) < len(meaningful_words) * 0.1:  # 10%未満の語彙多様性
            logger.warning("Low vocabulary diversity")
            return False
        
        return True
    
    def _final_cleanup(self, text: str) -> str:
        """最終的なクリーンアップ"""
        
        # 複数の空白を単一に変換
        text = re.sub(r'\s+', ' ', text)
        
        # 複数の改行を最大2つに制限
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 前後の空白を削除
        text = text.strip()
        
        return text
    
    def _calculate_confidence(self, original: str, removed: str, cleaned: str) -> float:
        """削除の信頼度を計算"""
        
        if not removed:
            return 0.0
        
        # 削除比率スコア
        removal_ratio = len(removed) / len(original)
        if 0.05 <= removal_ratio <= 0.3:  # 5-30%の削除が理想
            ratio_score = 1.0
        elif removal_ratio < 0.05:
            ratio_score = removal_ratio / 0.05
        else:
            ratio_score = max(0.0, 1.0 - (removal_ratio - 0.3) / 0.3)
        
        # 削除テキストの指標密度
        indicators = self.pattern_detector.count_reference_indicators(removed)
        indicator_density = sum(indicators.values()) / len(removed) * 1000
        indicator_score = min(indicator_density, 1.0)
        
        # 残存テキストの品質
        quality_score = 1.0 if self._quality_check(cleaned) else 0.0
        
        # 総合信頼度
        confidence = ratio_score * 0.3 + indicator_score * 0.4 + quality_score * 0.3
        
        return confidence

def create_reference_cleaner(config: Dict) -> ReferenceCleaner:
    """
    設定からReferenceCleanerを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        ReferenceCleaner インスタンス
    """
    return ReferenceCleaner(config.get('text_processing', {}))