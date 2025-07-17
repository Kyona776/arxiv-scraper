"""
テキスト後処理モジュールのユニットテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import statistics

from src.text_processor import (
    ProcessedText,
    TextNormalizer,
    SectionExtractor,
    QualityAnalyzer,
    TextProcessor,
    create_text_processor
)


class TestProcessedText:
    """ProcessedText データクラスのテスト"""
    
    def test_processed_text_creation(self):
        """ProcessedText オブジェクト作成テスト"""
        sections = {"introduction": "Intro text", "conclusion": "Conclusion text"}
        metadata = {"original_length": 1000, "processed_length": 900}
        statistics_data = {"word_count": 150, "sentence_count": 10}
        notes = ["Processing note 1", "Processing note 2"]
        
        result = ProcessedText(
            text="Processed text content",
            sections=sections,
            metadata=metadata,
            statistics=statistics_data,
            quality_score=0.85,
            processing_notes=notes
        )
        
        assert result.text == "Processed text content"
        assert result.sections["introduction"] == "Intro text"
        assert result.metadata["original_length"] == 1000
        assert result.statistics["word_count"] == 150
        assert result.quality_score == 0.85
        assert len(result.processing_notes) == 2


class TestTextNormalizer:
    """TextNormalizer クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'remove_page_numbers': True,
            'remove_headers_footers': True,
            'normalize_whitespace': True
        }
    
    @pytest.fixture
    def normalizer(self, config):
        """TextNormalizer インスタンス"""
        return TextNormalizer(config)
    
    def test_init(self, config):
        """初期化テスト"""
        normalizer = TextNormalizer(config)
        
        assert normalizer.remove_page_numbers is True
        assert normalizer.remove_headers_footers is True
        assert normalizer.normalize_whitespace is True
    
    def test_normalize_basic(self, normalizer):
        """基本正規化テスト"""
        messy_text = '''
   This    is   a   messy    text   with    extra   spaces.


   Multiple   newlines   here.




   Final paragraph.   
'''
        
        result = normalizer.normalize(messy_text)
        
        # 複数空白が単一になっている
        assert '    ' not in result
        assert '   ' not in result
        # 過度な改行が削除されている
        assert '\n\n\n' not in result
        # 基本的な内容は保持されている
        assert 'messy text' in result
        assert 'Multiple newlines' in result
    
    def test_fix_ocr_errors(self, normalizer):
        """OCRエラー修正テスト"""
        ocr_text = "ThislsATestWlthOCRErr0rs.Next5entence."
        
        result = normalizer._fix_ocr_errors(ocr_text)
        
        # OCRエラーの一部が修正されていることを確認
        assert result != ocr_text  # 何らかの変更があること
    
    def test_remove_page_numbers(self, normalizer):
        """ページ番号削除テスト"""
        text_with_pages = '''
Introduction text here.

1

Some content on page 1.

Page 2

More content here.

- 3 -

Final content.
'''
        
        result = normalizer._remove_page_numbers(text_with_pages)
        
        # ページ番号パターンが削除されていることを確認
        lines = result.split('\n')
        page_number_lines = [line for line in lines if line.strip() in ['1', 'Page 2', '- 3 -']]
        assert len(page_number_lines) == 0
        # 内容は保持されている
        assert 'Introduction text' in result
        assert 'Some content' in result
    
    def test_remove_headers_footers(self, normalizer):
        """ヘッダー・フッター削除テスト"""
        text_with_headers = '''
arXiv preprint submission

Main content paragraph 1.
This is important research content.

arXiv preprint submission

More important content here.
Scientific findings and analysis.

arXiv preprint submission
'''
        
        result = normalizer._remove_headers_footers(text_with_headers)
        
        # ヘッダー・フッターの出現回数が減っていることを確認
        original_count = text_with_headers.count('arXiv preprint submission')
        result_count = result.count('arXiv preprint submission')
        assert result_count < original_count
        # 主要コンテンツは保持
        assert 'important research content' in result
    
    def test_normalize_whitespace(self, normalizer):
        """空白正規化テスト"""
        text = "Multiple    spaces   and\n\n\n\nmultiple newlines."
        
        result = normalizer._normalize_whitespace(text)
        
        assert '    ' not in result
        assert '\n\n\n' not in result
        assert 'Multiple spaces' in result
    
    def test_remove_unwanted_chars(self, normalizer):
        """不要文字削除テスト"""
        text = "Normal text\x00with\x0Ccontrol\x1Fcharacters\u200Band\uFEFFinvisible chars."
        
        result = normalizer._remove_unwanted_chars(text)
        
        # 制御文字が削除されていることを確認
        assert '\x00' not in result
        assert '\x0C' not in result
        assert '\x1F' not in result
        assert '\u200B' not in result
        assert '\uFEFF' not in result
        # 通常のテキストは保持
        assert 'Normal text' in result
        assert 'with' in result
    
    def test_similarity(self, normalizer):
        """文字列類似度計算テスト"""
        s1 = "This is a test sentence"
        s2 = "This is a test sentence"
        s3 = "This is completely different"
        s4 = ""
        
        # 同一文字列
        assert normalizer._similarity(s1, s2) == 1.0
        
        # 異なる文字列
        similarity = normalizer._similarity(s1, s3)
        assert 0.0 <= similarity < 1.0
        
        # 空文字列
        assert normalizer._similarity(s1, s4) == 0.0
        assert normalizer._similarity(s4, s4) == 0.0


class TestSectionExtractor:
    """SectionExtractor クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {}
    
    @pytest.fixture
    def extractor(self, config):
        """SectionExtractor インスタンス"""
        return SectionExtractor(config)
    
    def test_init(self, extractor):
        """初期化テスト"""
        assert len(extractor.section_patterns) > 0
        # 基本的なパターンが含まれていることを確認
        patterns_str = '|'.join(extractor.section_patterns)
        assert 'ABSTRACT' in patterns_str or 'Abstract' in patterns_str
        assert 'INTRODUCTION' in patterns_str or 'Introduction' in patterns_str
    
    def test_extract_sections(self, extractor):
        """セクション抽出テスト"""
        academic_text = '''
Abstract
This is the abstract of the paper.

Introduction
This is the introduction section.

2. Methodology
This describes our method.

Results
These are our experimental results.

Conclusion
This is the conclusion.
'''
        
        sections = extractor.extract_sections(academic_text)
        
        # 基本的なセクションが抽出されていることを確認
        assert len(sections) >= 3
        assert any('abstract' in key.lower() for key in sections.keys())
        assert any('introduction' in key.lower() for key in sections.keys())
        assert any('conclusion' in key.lower() for key in sections.keys())
        
        # セクションの内容が正しいことを確認
        for key, content in sections.items():
            if 'abstract' in key.lower():
                assert 'abstract of the paper' in content
            elif 'introduction' in key.lower():
                assert 'introduction section' in content
    
    def test_detect_section_header(self, extractor):
        """セクション見出し検出テスト"""
        test_cases = [
            ('Abstract', 'abstract'),
            ('INTRODUCTION', 'introduction'),
            ('2. Methodology', 'methodology'),
            ('3.1 Experimental Setup', 'experimental_setup'),
            ('Not a header', None),
            ('Regular paragraph text.', None),
        ]
        
        for input_line, expected in test_cases:
            result = extractor._detect_section_header(input_line)
            if expected:
                assert expected in result.lower() if result else False
            else:
                assert result is None


class TestQualityAnalyzer:
    """QualityAnalyzer クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {}
    
    @pytest.fixture
    def analyzer(self, config):
        """QualityAnalyzer インスタンス（spaCyモック付き）"""
        with patch('src.text_processor.spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            return QualityAnalyzer(config)
    
    def test_init(self, config):
        """初期化テスト"""
        with patch('src.text_processor.spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            
            analyzer = QualityAnalyzer(config)
            assert analyzer.nlp == mock_nlp
    
    def test_init_fallback(self, config):
        """spaCyモデル読み込み失敗時のフォールバックテスト"""
        with patch('src.text_processor.spacy.load', side_effect=OSError("Model not found")), \
             patch('subprocess.check_call') as mock_subprocess, \
             patch('src.text_processor.English') as mock_english:
            
            # ダウンロード失敗をシミュレート
            mock_subprocess.side_effect = Exception("Download failed")
            mock_english_instance = Mock()
            mock_english.return_value = mock_english_instance
            
            analyzer = QualityAnalyzer(config)
            assert analyzer.nlp == mock_english_instance
    
    def test_analyze_quality_empty_text(self, analyzer):
        """空テキストの品質分析テスト"""
        score, analysis = analyzer.analyze_quality("")
        
        assert score == 0.0
        assert analysis == {}
    
    def test_analyze_quality_good_text(self, analyzer):
        """高品質テキストの分析テスト"""
        good_text = '''
This is a comprehensive research paper that presents novel findings in machine learning.
The methodology is robust and the experimental design is rigorous. We conducted extensive
experiments on multiple datasets to validate our approach. The results demonstrate
significant improvements over existing methods. Figure 1 shows the main architecture.
Table 2 presents the experimental results. According to Smith et al., previous work
has established the theoretical foundations. Our research builds upon these findings
and extends them to new applications.
''' * 3  # より長いテキストにする
        
        score, analysis = analyzer.analyze_quality(good_text)
        
        assert 0.0 <= score <= 1.0
        assert 'basic_stats' in analysis
        assert 'language_score' in analysis
        assert 'structure_score' in analysis
        assert 'content_score' in analysis
        assert 'total_score' in analysis
        
        # 基本統計のチェック
        stats = analysis['basic_stats']
        assert stats['word_count'] > 0
        assert stats['sentence_count'] > 0
        assert stats['char_count'] > 0
    
    def test_calculate_basic_stats(self, analyzer):
        """基本統計計算テスト"""
        text = '''
First paragraph with multiple sentences. This is the second sentence.

Second paragraph here. Another sentence. And one more.

Third paragraph with content.
'''
        
        stats = analyzer._calculate_basic_stats(text)
        
        assert stats['char_count'] > 0
        assert stats['word_count'] > 0
        assert stats['sentence_count'] >= 5  # 少なくとも5文
        assert stats['paragraph_count'] >= 3  # 3段落
        assert stats['avg_words_per_sentence'] > 0
        assert stats['avg_sentences_per_paragraph'] > 0
    
    def test_analyze_language_quality(self, analyzer):
        """言語品質分析テスト"""
        # モックNLPの設定
        mock_tokens = []
        for word in ['This', 'is', 'a', 'test', 'sentence', 'with', 'various', 'words']:
            token = Mock()
            token.is_alpha = True
            token.pos_ = 'NOUN' if word in ['test', 'sentence', 'words'] else 'DET'
            mock_tokens.append(token)
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        analyzer.nlp.return_value = mock_doc
        
        text = "This is a test sentence with various words and good vocabulary diversity."
        
        score = analyzer._analyze_language_quality(text)
        
        assert 0.0 <= score <= 1.0
    
    def test_analyze_structure_quality(self, analyzer):
        """構造品質分析テスト"""
        structured_text = '''
This is the first paragraph with sufficient length to be meaningful for analysis.

This is the second paragraph that also has adequate length for quality assessment.

This is the third paragraph maintaining consistent structure and flow.

This is the fourth paragraph completing the structural analysis test.
'''
        
        score = analyzer._analyze_structure_quality(structured_text)
        
        assert 0.0 <= score <= 1.0
    
    def test_analyze_content_quality(self, analyzer):
        """内容品質分析テスト"""
        academic_text = '''
This research presents a novel approach to machine learning using advanced algorithms.
Our study demonstrates significant performance improvements through extensive experiments.
The proposed method shows superior results compared to baseline approaches.
Figure 3 illustrates the system architecture while Table 1 shows performance metrics.
According to recent studies, this represents a major advance in the field.
'''
        
        score = analyzer._analyze_content_quality(academic_text)
        
        assert 0.0 <= score <= 1.0
        # 学術的キーワードが含まれているので、ある程度のスコアが期待される
        assert score > 0.1


class TestTextProcessor:
    """TextProcessor クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'remove_page_numbers': True,
            'remove_headers_footers': True,
            'normalize_whitespace': True
        }
    
    @pytest.fixture
    def processor(self, config):
        """TextProcessor インスタンス（モック付き）"""
        with patch('src.text_processor.spacy.load'):
            return TextProcessor(config)
    
    def test_init(self, config):
        """初期化テスト"""
        with patch('src.text_processor.spacy.load'):
            processor = TextProcessor(config)
            
            assert isinstance(processor.normalizer, TextNormalizer)
            assert isinstance(processor.section_extractor, SectionExtractor)
            assert isinstance(processor.quality_analyzer, QualityAnalyzer)
    
    def test_process_text_comprehensive(self, processor):
        """包括的テキスト処理テスト"""
        input_text = '''
    Abstract    

This   paper   presents   a   novel   approach   to   machine   learning.


Introduction

Machine learning has become increasingly important. Our method shows improvements.


Methodology   

We propose a new algorithm based on deep neural networks.   The approach is innovative.


Results

Extensive experiments demonstrate the effectiveness of our method. Figure 1 shows results.


Conclusion

We have presented a novel approach that outperforms existing methods significantly.
''' * 2  # より長いテキストにする
        
        result = processor.process_text(input_text)
        
        # 結果の基本チェック
        assert isinstance(result, ProcessedText)
        assert len(result.text) > 0
        assert len(result.sections) > 0
        assert 0.0 <= result.quality_score <= 1.0
        assert len(result.processing_notes) > 0
        
        # メタデータのチェック
        assert 'original_length' in result.metadata
        assert 'processed_length' in result.metadata
        assert 'compression_ratio' in result.metadata
        assert 'sections_found' in result.metadata
        
        # 正規化されたテキストのチェック
        assert '   ' not in result.text  # 複数空白が削除されている
        assert '\n\n\n' not in result.text  # 過度な改行が削除されている
        
        # セクションが抽出されていることを確認
        assert any('abstract' in key.lower() for key in result.sections.keys())
        assert any('introduction' in key.lower() for key in result.sections.keys())
    
    def test_process_text_quality_degradation(self, processor):
        """品質劣化の検出テスト"""
        long_text = "A" * 1000  # 1000文字のテキスト
        short_result = "A" * 500  # 処理後に半分になった場合
        
        # ノーマライザーをモックして短縮をシミュレート
        processor.normalizer.normalize = Mock(return_value=short_result)
        
        result = processor.process_text(long_text)
        
        # 大幅な短縮が記録されていることを確認
        assert any('reduction' in note.lower() or 'significant' in note.lower() 
                  for note in result.processing_notes)
    
    def test_process_text_empty_input(self, processor):
        """空入力のテスト"""
        result = processor.process_text("")
        
        assert isinstance(result, ProcessedText)
        assert result.text == ""
        assert result.quality_score == 0.0


class TestCreateTextProcessor:
    """create_text_processor 関数のテスト"""
    
    def test_create_processor(self):
        """プロセッサー作成テスト"""
        config = {
            'text_processing': {
                'remove_page_numbers': False,
                'normalize_whitespace': True
            }
        }
        
        with patch('src.text_processor.spacy.load'):
            processor = create_text_processor(config)
            
            assert isinstance(processor, TextProcessor)
            assert processor.normalizer.remove_page_numbers is False
            assert processor.normalizer.normalize_whitespace is True


class TestIntegration:
    """統合テスト"""
    
    def test_end_to_end_text_processing(self):
        """エンドツーエンドテキスト処理テスト"""
        config = {
            'remove_page_numbers': True,
            'remove_headers_footers': True,
            'normalize_whitespace': True
        }
        
        # 実際の論文に近いテキスト
        raw_paper_text = '''
   arXiv:2023.12345v1 [cs.LG] 15 Jan 2023

   Abstract    

This   paper   presents   a   comprehensive   study   of   machine   learning   algorithms.
We   propose   a   novel   approach   that   significantly   outperforms   existing   methods.


   1

Introduction

Machine learning has become increasingly important in recent years. Many researchers
have worked on improving algorithms and methodologies. Our approach builds on
previous work but introduces several key innovations.

The main contributions of this paper are: (1) a new neural architecture,
(2) improved training procedures, and (3) comprehensive experimental validation.

   2

   2.1 Related Work

Previous research in this area includes several important contributions. The seminal
work by Johnson (2020) established theoretical foundations. Recent advances have
shown practical improvements in various applications.

arXiv preprint - Do not distribute

   3. Methodology   

Our approach consists of three main components. The first component handles data
preprocessing efficiently. The second component implements the core algorithm with
optimizations. The third component provides fine-tuning capabilities.

Figure 1 shows the overall architecture. Table 2 presents experimental results.
According to recent studies, this represents a significant advance.

   4

Results and Discussion

We conducted extensive experiments on five benchmark datasets. The results demonstrate
consistent improvements over baseline methods across all evaluation metrics tested.
Statistical significance was confirmed using standard tests.

   5

Conclusion

We have presented a novel machine learning approach that achieves state-of-the-art
performance on multiple benchmark datasets. Future work will explore additional
applications and theoretical analysis of the proposed method.

arXiv preprint - Do not distribute
'''
        
        with patch('src.text_processor.spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp
            
            processor = TextProcessor(config)
            result = processor.process_text(raw_paper_text)
        
        # 基本的な処理結果の検証
        assert isinstance(result, ProcessedText)
        assert len(result.text) > 0
        
        # 不要要素が削除されていることを確認
        # ページ番号削除
        assert '\n1\n' not in result.text
        assert '\n2\n' not in result.text
        
        # ヘッダー・フッター削除
        processed_arxiv_count = result.text.count('arXiv preprint')
        original_arxiv_count = raw_paper_text.count('arXiv preprint')
        assert processed_arxiv_count <= original_arxiv_count
        
        # 空白正規化
        assert '   ' not in result.text
        
        # 重要なコンテンツが保持されていることを確認
        assert 'machine learning' in result.text.lower()
        assert 'methodology' in result.text.lower()
        assert 'conclusion' in result.text.lower()
        
        # セクション抽出の確認
        assert len(result.sections) >= 3
        section_keys = [key.lower() for key in result.sections.keys()]
        assert any('abstract' in key for key in section_keys)
        assert any('introduction' in key for key in section_keys)
        assert any('conclusion' in key for key in section_keys)
        
        # 品質スコアの確認
        assert 0.0 <= result.quality_score <= 1.0
        
        # メタデータの確認
        assert result.metadata['original_length'] > result.metadata['processed_length']
        assert 0.0 < result.metadata['compression_ratio'] < 1.0


# Mock decorators for heavy dependencies
@pytest.fixture(autouse=True)
def mock_heavy_imports():
    """重いライブラリのインポートをモック"""
    with patch.dict('sys.modules', {
        'spacy': Mock(),
        'spacy.lang.en': Mock(),
    }):
        yield


if __name__ == '__main__':
    pytest.main([__file__])