"""
Reference削除モジュールのユニットテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import re

from src.reference_cleaner import (
    ReferencePatternDetector,
    ReferenceCleaner,
    ReferenceDetectionResult,
    create_reference_cleaner
)


class TestReferencePatternDetector:
    """ReferencePatternDetectorクラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'reference_patterns': ['REFERENCES', 'Bibliography'],
            'citation_patterns': [r'\[\d+\]', r'\(\d{4}\)'],
            'citation_cleanup': True
        }
    
    @pytest.fixture
    def detector(self, config):
        """ReferencePatternDetectorインスタンス"""
        return ReferencePatternDetector(config)
    
    def test_init_patterns(self, detector):
        """パターン初期化テスト"""
        assert len(detector.section_patterns) > 0
        assert len(detector.reference_entry_patterns) > 0
        assert len(detector.inline_citation_patterns) > 0
        
        # 基本的なパターンが含まれていることを確認
        section_pattern_str = '|'.join(detector.section_patterns)
        assert 'REFERENCES' in section_pattern_str
        assert 'Bibliography' in section_pattern_str
    
    def test_find_reference_sections(self, detector):
        """Reference セクション検出テスト"""
        text = '''
Introduction
This is the introduction section.

Methodology
This describes our method.

REFERENCES

[1] Smith, J. (2023). A paper about something.
[2] Doe, J. (2022). Another important work.

Appendix
Some additional information.
'''
        
        sections = detector.find_reference_sections(text)
        
        assert len(sections) >= 1
        # 最初に見つかったセクションをチェック
        start, end, title = sections[0]
        assert 'REFERENCES' in title
        assert start > 0
        assert end > start
    
    def test_count_reference_indicators(self, detector):
        """Reference指標カウントテスト"""
        reference_text = '''
[1] Smith, J. (2023). A comprehensive study. doi:10.1000/123
[2] Brown, A., et al. (2022). Advanced techniques. https://example.com
3. Wilson, K. Machine learning approaches.
'''
        
        indicators = detector.count_reference_indicators(reference_text)
        
        assert indicators['numbered_lists'] >= 1  # "3." パターン
        assert indicators['bracketed_numbers'] >= 2  # [1], [2]
        assert indicators['author_patterns'] >= 2  # Smith, J. と Brown, A.
        assert indicators['year_patterns'] >= 2  # (2023), (2022)
        assert indicators['doi_url_patterns'] >= 2  # doi: と https://
    
    def test_is_reference_section_true(self, detector):
        """Reference セクション判定テスト（True の場合）"""
        reference_text = '''
[1] Smith, J. (2023). Deep learning fundamentals. Nature 567, 123-134.
[2] Brown, A., et al. (2022). Advanced neural networks. doi:10.1038/s41586-022-1234-5
[3] Wilson, K. (2021). Machine learning applications. https://arxiv.org/abs/2101.12345
'''
        
        result = detector.is_reference_section(reference_text)
        assert result is True
    
    def test_is_reference_section_false(self, detector):
        """Reference セクション判定テスト（False の場合）"""
        main_text = '''
This paper presents a novel approach to machine learning. Our method builds on 
previous work in the field and demonstrates significant improvements over 
existing techniques. The experimental results show that our approach achieves
state-of-the-art performance on several benchmark datasets.
'''
        
        result = detector.is_reference_section(main_text)
        assert result is False
    
    def test_is_reference_section_empty(self, detector):
        """空テキストの判定テスト"""
        result = detector.is_reference_section('')
        assert result is False
        
        result = detector.is_reference_section('   \n  \t  ')
        assert result is False


class TestReferenceCleaner:
    """ReferenceCleanerクラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'remove_references': True,
            'citation_cleanup': True,
            'min_text_length': 100,
            'max_reference_ratio': 0.3,
            'reference_patterns': ['REFERENCES', 'Bibliography'],
            'citation_patterns': [r'\[\d+\]', r'\(\d{4}\)']
        }
    
    @pytest.fixture
    def cleaner(self, config):
        """ReferenceCleanerインスタンス"""
        return ReferenceCleaner(config)
    
    def test_init(self, config):
        """初期化テスト"""
        cleaner = ReferenceCleaner(config)
        
        assert cleaner.remove_references is True
        assert cleaner.citation_cleanup is True
        assert cleaner.min_text_length == 100
        assert cleaner.max_reference_ratio == 0.3
        assert isinstance(cleaner.pattern_detector, ReferencePatternDetector)
    
    def test_clean_text_disabled(self, config):
        """Reference削除無効化テスト"""
        config['remove_references'] = False
        cleaner = ReferenceCleaner(config)
        
        text = "Sample text with REFERENCES section"
        result = cleaner.clean_text(text)
        
        assert isinstance(result, ReferenceDetectionResult)
        assert result.cleaned_text == text
        assert result.confidence == 0.0
        assert result.reference_start == -1
    
    def test_clean_text_no_references_found(self, cleaner):
        """Reference セクションが見つからない場合のテスト"""
        text = '''
Introduction

This is a paper about machine learning. The approach we propose is novel and
effective. Our experiments demonstrate significant improvements over baseline
methods. The results are very promising and suggest future research directions.

Conclusion

We have presented a new method that works well.
''' * 5  # 十分な長さにする
        
        result = cleaner.clean_text(text)
        
        assert isinstance(result, ReferenceDetectionResult)
        assert result.reference_start == -1
        assert result.confidence == 0.0
        assert len(result.cleaned_text) > 0
    
    def test_clean_text_with_references(self, cleaner):
        """Reference セクションありのテスト"""
        text = '''
Introduction

This is a paper about machine learning [1]. The approach we propose builds on 
previous work [2,3] and demonstrates significant improvements. Our method is 
based on deep learning techniques (Smith et al., 2023) and achieves 
state-of-the-art results.

The experimental setup follows standard protocols. We evaluate on multiple 
datasets and compare against several baseline methods. The results show 
consistent improvements across all metrics tested.

Conclusion

We have presented a novel approach to machine learning that outperforms 
existing methods. Future work will explore additional applications and 
improvements to the proposed technique.

REFERENCES

[1] Smith, J. (2023). Deep learning fundamentals. Nature 567, 123-134. 
    doi:10.1038/s41586-023-1234-5

[2] Brown, A., Wilson, K., and Davis, M. (2022). Advanced neural network 
    architectures for computer vision. arXiv:2201.12345v2 [cs.CV]

[3] Johnson, L., et al. (2021). Large-scale machine learning systems.
    Proceedings of ICML 2021, pp. 1234-1245. https://proceedings.mlr.press/

[4] Williams, R. and Taylor, S. (2020). Optimization techniques for deep
    learning. Journal of Machine Learning Research 21(15):1-34.
'''
        
        result = cleaner.clean_text(text)
        
        assert isinstance(result, ReferenceDetectionResult)
        assert result.reference_start >= 0
        assert result.reference_end > result.reference_start
        assert len(result.removed_text) > 0
        assert 'REFERENCES' in result.removed_text
        assert '[1] Smith, J.' in result.removed_text
        assert 'REFERENCES' not in result.cleaned_text
        assert result.confidence > 0.5
    
    def test_clean_inline_citations(self, cleaner):
        """インライン引用削除テスト"""
        text = '''
This method [1,2,3] shows great promise. Previous work (Smith et al., 2023) 
has demonstrated similar results. The technique (2022) is well established.
Another study [5-7] confirms these findings.
'''
        
        cleaned = cleaner._clean_inline_citations(text)
        
        # 引用が削除されていることを確認
        assert '[1,2,3]' not in cleaned
        assert '(Smith et al., 2023)' not in cleaned
        assert '(2022)' not in cleaned
        assert '[5-7]' not in cleaned
        # 基本的なテキストは残っている
        assert 'This method' in cleaned
        assert 'shows great promise' in cleaned
    
    def test_quality_check_pass(self, cleaner):
        """品質チェック合格テスト"""
        # 十分な長さと品質のテキスト
        good_text = '''
This is a comprehensive research paper that presents novel findings in the field
of machine learning. The methodology is sound and the experimental design is 
rigorous. We conducted extensive experiments on multiple datasets to validate
our approach. The results demonstrate significant improvements over existing
methods. The statistical analysis confirms the reliability of our findings.
The implications of this work are substantial and will influence future research
in this important area. We believe this contribution advances the state of the
art and provides valuable insights for practitioners and researchers alike.
The technical details are thoroughly documented and the reproducibility of the
results is ensured through careful documentation of all experimental procedures.
''' * 3  # さらに長くする
        
        result = cleaner._quality_check(good_text)
        assert result is True
    
    def test_quality_check_fail_too_short(self, cleaner):
        """品質チェック失敗テスト（短すぎる）"""
        short_text = "This is too short."
        
        result = cleaner._quality_check(short_text)
        assert result is False
    
    def test_calculate_confidence(self, cleaner):
        """信頼度計算テスト"""
        original = "A" * 1000  # 1000文字の元テキスト
        removed = '''
REFERENCES
[1] Smith, J. (2023). Paper title.
[2] Brown, A. (2022). Another paper.
'''  # 典型的な参考文献
        cleaned = "A" * 800  # 削除後のテキスト
        
        confidence = cleaner._calculate_confidence(original, removed, cleaned)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0  # 何らかの信頼度があることを確認
    
    def test_final_cleanup(self, cleaner):
        """最終クリーンアップテスト"""
        messy_text = '''
   This    is   a   messy    text   with    extra   spaces.


   And   multiple   newlines.




   More content here.   
'''
        
        cleaned = cleaner._final_cleanup(messy_text)
        
        # 複数の空白が単一になっている
        assert '    ' not in cleaned
        assert '   ' not in cleaned
        # 過度な改行が削除されている
        assert '\n\n\n' not in cleaned
        # 前後の空白が削除されている
        assert not cleaned.startswith(' ')
        assert not cleaned.endswith(' ')
    
    def test_select_best_reference_section(self, cleaner):
        """最適なReference セクション選択テスト"""
        text = '''
Early REFERENCES section (shouldn't be selected)
[1] Some ref.

Main content here with lots of text and important information.

Real REFERENCES section at the end
[1] Smith, J. (2023). Important paper. doi:10.1000/123
[2] Brown, A., et al. (2022). Another work. arXiv:2201.12345
[3] Wilson, K. (2021). Third reference. Nature 567, 123-134.
'''
        
        # Referenceセクションを見つける
        sections = cleaner.pattern_detector.find_reference_sections(text)
        
        if sections:
            best_section = cleaner._select_best_reference_section(text, sections)
            
            if best_section:
                start, end, title = best_section
                section_text = text[start:end]
                
                # より後方にあり、より適切なセクションが選ばれることを確認
                assert 'Real REFERENCES' in section_text or len(section_text.split('[')) >= 3


class TestCreateReferenceCleaner:
    """create_reference_cleaner 関数のテスト"""
    
    def test_create_cleaner(self):
        """クリーナー作成テスト"""
        config = {
            'text_processing': {
                'remove_references': True,
                'citation_cleanup': False,
                'min_text_length': 500
            }
        }
        
        cleaner = create_reference_cleaner(config)
        
        assert isinstance(cleaner, ReferenceCleaner)
        assert cleaner.remove_references is True
        assert cleaner.citation_cleanup is False
        assert cleaner.min_text_length == 500


class TestReferenceDetectionResult:
    """ReferenceDetectionResult データクラスのテスト"""
    
    def test_result_creation(self):
        """結果オブジェクト作成テスト"""
        result = ReferenceDetectionResult(
            reference_start=100,
            reference_end=500,
            confidence=0.85,
            detected_patterns=['REFERENCES'],
            removed_text='[1] Reference text',
            cleaned_text='Clean main text',
            section_title='REFERENCES'
        )
        
        assert result.reference_start == 100
        assert result.reference_end == 500
        assert result.confidence == 0.85
        assert 'REFERENCES' in result.detected_patterns
        assert result.section_title == 'REFERENCES'


class TestIntegration:
    """統合テスト"""
    
    def test_end_to_end_reference_cleaning(self):
        """エンドツーエンドのReference削除テスト"""
        config = {
            'remove_references': True,
            'citation_cleanup': True,
            'min_text_length': 50,
            'max_reference_ratio': 0.5
        }
        
        cleaner = ReferenceCleaner(config)
        
        # 実際の論文に近いテキスト
        academic_paper = '''
Abstract

This paper presents a novel approach to machine learning that significantly 
outperforms existing methods. We introduce a new algorithm based on deep 
neural networks that can handle complex datasets efficiently.

1. Introduction

Machine learning has become increasingly important in recent years [1]. Many 
researchers have worked on improving algorithms [2,3]. Our approach builds 
on previous work (Smith et al., 2023) but introduces several key innovations.

The main contributions of this paper are: (1) a new neural architecture, 
(2) improved training procedures, and (3) comprehensive experimental validation.

2. Related Work

Previous research in this area includes several important contributions. The 
seminal work by Johnson (2020) established the theoretical foundations. Recent 
advances by Brown et al. (2022) have shown practical improvements.

3. Methodology

Our approach consists of three main components. The first component handles 
data preprocessing. The second component implements the core algorithm. The 
third component provides optimization and fine-tuning capabilities.

4. Experiments

We conducted extensive experiments on five benchmark datasets. The results 
demonstrate consistent improvements over baseline methods across all metrics. 
Statistical significance was confirmed using standard tests.

5. Conclusion

We have presented a novel machine learning approach that achieves state-of-the-art 
performance. Future work will explore additional applications and theoretical 
analysis of the proposed method.

REFERENCES

[1] Smith, J. (2023). Foundations of machine learning. MIT Press, Cambridge, MA.

[2] Brown, A., Wilson, K., and Davis, M. (2022). Advanced algorithms for neural 
    networks. Journal of Machine Learning Research, 23(45):123-145.

[3] Johnson, L., Taylor, S., and Anderson, R. (2021). Deep learning architectures: 
    A comprehensive survey. Nature Machine Intelligence, 3:234-251. 
    doi:10.1038/s42256-021-00345-1

[4] Williams, P. et al. (2020). Large-scale optimization for machine learning. 
    Proceedings of the International Conference on Machine Learning, pp. 1234-1245.

[5] Davis, K. and Thompson, J. (2019). Statistical methods for model evaluation. 
    Annual Review of Statistics and Its Application, 6:123-145.
'''
        
        result = cleaner.clean_text(academic_paper)
        
        # 基本的な検証
        assert isinstance(result, ReferenceDetectionResult)
        assert result.reference_start >= 0
        assert result.confidence > 0.3
        
        # Reference セクションが削除されていることを確認
        assert 'REFERENCES' not in result.cleaned_text
        assert '[1] Smith, J.' not in result.cleaned_text
        assert 'MIT Press' not in result.cleaned_text
        
        # 重要なコンテンツが残っていることを確認
        assert 'Abstract' in result.cleaned_text
        assert 'Introduction' in result.cleaned_text
        assert 'Methodology' in result.cleaned_text
        assert 'Conclusion' in result.cleaned_text
        assert 'novel approach' in result.cleaned_text
        
        # 削除されたテキストに参考文献が含まれていることを確認
        assert 'REFERENCES' in result.removed_text
        assert '[1] Smith, J.' in result.removed_text
    
    def test_multiple_reference_formats(self):
        """複数のReference形式のテスト"""
        config = {'remove_references': True, 'citation_cleanup': True}
        cleaner = ReferenceCleaner(config)
        
        # 様々なReference形式を含むテキスト
        text_with_various_formats = '''
Main content of the paper goes here with lots of important information.

BIBLIOGRAPHY

1. Author, A. (2023). Book title. Publisher.
2. Writer, B. et al. Journal paper title. Journal Name 45(2):123-130.

Additional text after references.
''' * 3  # 長さを確保
        
        result = cleaner.clean_text(text_with_various_formats)
        
        # BIBLIOGRAPHY セクションが検出・削除されることを確認
        if result.reference_start >= 0:
            assert 'BIBLIOGRAPHY' not in result.cleaned_text
            assert 'Author, A. (2023)' not in result.cleaned_text


# Mock decorators for heavy dependencies
@pytest.fixture(autouse=True)
def mock_nltk():
    """NLTK の重い処理をモック"""
    with patch('src.reference_cleaner.sent_tokenize') as mock_sent, \
         patch('src.reference_cleaner.word_tokenize') as mock_word:
        
        mock_sent.side_effect = lambda text: text.split('. ')
        mock_word.side_effect = lambda text: text.split()
        
        yield mock_sent, mock_word


if __name__ == '__main__':
    pytest.main([__file__])