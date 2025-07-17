"""
OCR処理モジュールのユニットテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
from PIL import Image
import io

from src.ocr_processor import (
    OCRResult,
    OCRProcessor,
    MistralOCRProcessor,
    NougatOCRProcessor,
    UnstructuredOCRProcessor,
    SuryaOCRProcessor,
    OCRManager,
    create_ocr_manager
)


class TestOCRResult:
    """OCRResult データクラスのテスト"""
    
    def test_ocr_result_creation(self):
        """OCRResult オブジェクト作成テスト"""
        result = OCRResult(
            text="Extracted text content",
            confidence=0.95,
            processing_time=12.34,
            model_used="test_model",
            page_count=5,
            metadata={"test": "data"},
            errors=["warning message"]
        )
        
        assert result.text == "Extracted text content"
        assert result.confidence == 0.95
        assert result.processing_time == 12.34
        assert result.model_used == "test_model"
        assert result.page_count == 5
        assert result.metadata["test"] == "data"
        assert "warning message" in result.errors


class TestMistralOCRProcessor:
    """MistralOCRProcessor クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'device': 'cpu',
            'max_length': 1024,
            'temperature': 0.1
        }
    
    @pytest.fixture
    def processor(self, config):
        """MistralOCRProcessor インスタンス"""
        return MistralOCRProcessor(config)
    
    def test_init(self, config):
        """初期化テスト"""
        processor = MistralOCRProcessor(config)
        
        assert processor.device == 'cpu'
        assert processor.max_length == 1024
        assert processor.temperature == 0.1
        assert processor.model is None
        assert processor.tokenizer is None
    
    def test_is_available(self, processor):
        """利用可能性チェックテスト"""
        # Mistral OCR は実際には存在しないので False を返すべき
        result = processor.is_available()
        assert result is False
    
    def test_load_model_raises_error(self, processor):
        """モデルロードエラーテスト"""
        with pytest.raises(ImportError):
            processor._load_model()
    
    @patch('src.ocr_processor.fitz')
    def test_process_pdf_model_load_failure(self, mock_fitz, processor):
        """PDFファイル処理でモデルロード失敗テスト"""
        # モックPDFドキュメント
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.close = Mock()
        mock_fitz.open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            result = processor.process_pdf(temp_pdf.name)
            
            assert isinstance(result, OCRResult)
            assert result.text == ""
            assert result.confidence == 0.0
            assert result.model_used == "mistral_ocr"
            assert len(result.errors) > 0
    
    def test_process_image_returns_empty(self, processor):
        """画像処理で空文字列を返すテスト"""
        # テスト用画像作成
        img = Image.new('RGB', (100, 100), color='white')
        
        result = processor.process_image(img)
        assert result == ""


class TestNougatOCRProcessor:
    """NougatOCRProcessor クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'device': 'cpu',
            'model_size': 'base',
            'recompute': False
        }
    
    @pytest.fixture
    def processor(self, config):
        """NougatOCRProcessor インスタンス"""
        return NougatOCRProcessor(config)
    
    def test_init(self, config):
        """初期化テスト"""
        processor = NougatOCRProcessor(config)
        
        assert processor.device == 'cpu'
        assert processor.model_size == 'base'
        assert processor.recompute is False
        assert processor.model is None
    
    def test_is_available_no_nougat(self, processor):
        """Nougat未インストール時の利用可能性チェック"""
        with patch('builtins.__import__', side_effect=ImportError):
            result = processor.is_available()
            assert result is False
    
    @patch('src.ocr_processor.nougat', create=True)
    def test_is_available_with_nougat(self, mock_nougat, processor):
        """Nougatインストール時の利用可能性チェック"""
        result = processor.is_available()
        assert result is True
    
    @patch('src.ocr_processor.NougatModel', create=True)
    @patch('src.ocr_processor.LazyDataset', create=True)
    def test_load_model_success(self, mock_dataset, mock_model, processor):
        """モデルロード成功テスト"""
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        processor._load_model()
        
        assert processor.model == mock_model_instance
        mock_model.from_pretrained.assert_called_once_with("facebook/nougat-base")
        mock_model_instance.to.assert_called_once_with('cpu')
    
    def test_load_model_import_error(self, processor):
        """モデルロードでImportErrorテスト"""
        with patch('src.ocr_processor.NougatModel', side_effect=ImportError("No nougat")):
            with pytest.raises(ImportError):
                processor._load_model()
    
    @patch('src.ocr_processor.NougatModel', create=True)
    @patch('src.ocr_processor.LazyDataset', create=True)
    def test_process_pdf_success(self, mock_dataset, mock_model, processor):
        """PDF処理成功テスト"""
        # モックの設定
        mock_model_instance = Mock()
        mock_model_instance.inference.return_value = "Extracted text from page"
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_dataset_instance = Mock()
        mock_dataset_instance.__iter__ = Mock(return_value=iter([Mock(), Mock()]))  # 2ページ
        mock_dataset.return_value = mock_dataset_instance
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            result = processor.process_pdf(temp_pdf.name)
            
            assert isinstance(result, OCRResult)
            assert result.model_used == "nougat"
            assert result.confidence == 0.90
            # inference が呼ばれた回数分のテキストが結合される
            expected_text = "Extracted text from page\n\nExtracted text from page"
            assert result.text == expected_text
    
    def test_process_image_not_supported(self, processor):
        """画像処理非対応テスト"""
        img = Image.new('RGB', (100, 100), color='white')
        
        result = processor.process_image(img)
        assert result == ""


class TestUnstructuredOCRProcessor:
    """UnstructuredOCRProcessor クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'strategy': 'hi_res',
            'model_name': 'yolox'
        }
    
    @pytest.fixture
    def processor(self, config):
        """UnstructuredOCRProcessor インスタンス"""
        return UnstructuredOCRProcessor(config)
    
    def test_init(self, config):
        """初期化テスト"""
        processor = UnstructuredOCRProcessor(config)
        
        assert processor.strategy == 'hi_res'
        assert processor.model_name == 'yolox'
    
    def test_is_available_no_unstructured(self, processor):
        """Unstructured未インストール時のテスト"""
        with patch('builtins.__import__', side_effect=ImportError):
            result = processor.is_available()
            assert result is False
    
    @patch('src.ocr_processor.unstructured', create=True)
    def test_is_available_with_unstructured(self, mock_unstructured, processor):
        """Unstructuredインストール時のテスト"""
        result = processor.is_available()
        assert result is True
    
    @patch('src.ocr_processor.partition_pdf', create=True)
    def test_process_pdf_success(self, mock_partition, processor):
        """PDF処理成功テスト"""
        # モック要素の作成
        mock_elements = [
            Mock(text="First paragraph text"),
            Mock(text="Second paragraph text"),
            Mock(text="Third paragraph text")
        ]
        mock_partition.return_value = mock_elements
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            result = processor.process_pdf(temp_pdf.name)
            
            assert isinstance(result, OCRResult)
            assert result.model_used == "unstructured"
            assert result.confidence == 0.85
            assert "First paragraph text" in result.text
            assert "Second paragraph text" in result.text
            assert "Third paragraph text" in result.text
            assert result.metadata['elements_count'] == 3
    
    @patch('src.ocr_processor.partition_image', create=True)
    def test_process_image_success(self, mock_partition, processor):
        """画像処理成功テスト"""
        mock_elements = [
            Mock(text="Image text 1"),
            Mock(text="Image text 2")
        ]
        mock_partition.return_value = mock_elements
        
        img = Image.new('RGB', (100, 100), color='white')
        
        result = processor.process_image(img)
        assert "Image text 1" in result
        assert "Image text 2" in result
    
    @patch('src.ocr_processor.partition_image', create=True)
    def test_process_image_with_file_path(self, mock_partition, processor):
        """ファイルパス指定での画像処理テスト"""
        mock_elements = [Mock(text="File image text")]
        mock_partition.return_value = mock_elements
        
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_img:
            result = processor.process_image(temp_img.name)
            assert "File image text" in result


class TestSuryaOCRProcessor:
    """SuryaOCRProcessor クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'device': 'cpu',
            'batch_size': 2,
            'det_batch_size': 2
        }
    
    @pytest.fixture
    def processor(self, config):
        """SuryaOCRProcessor インスタンス"""
        return SuryaOCRProcessor(config)
    
    def test_init(self, config):
        """初期化テスト"""
        processor = SuryaOCRProcessor(config)
        
        assert processor.device == 'cpu'
        assert processor.batch_size == 2
        assert processor.det_batch_size == 2
        assert processor.model is None
    
    def test_is_available_no_surya(self, processor):
        """Surya未インストール時のテスト"""
        with patch('builtins.__import__', side_effect=ImportError):
            result = processor.is_available()
            assert result is False
    
    @patch('src.ocr_processor.surya', create=True)
    def test_is_available_with_surya(self, mock_surya, processor):
        """Suryaインストール時のテスト"""
        result = processor.is_available()
        assert result is True
    
    @patch('src.ocr_processor.run_ocr', create=True)
    @patch('src.ocr_processor.segformer', create=True)
    @patch('src.ocr_processor.rec_model', create=True)
    @patch('src.ocr_processor.fitz')
    def test_process_pdf_success(self, mock_fitz, mock_rec_model, mock_segformer, mock_run_ocr, processor):
        """PDF処理成功テスト"""
        # モックPDFドキュメント設定
        mock_page = Mock()
        mock_pix = Mock()
        mock_pix.tobytes.return_value = b'fake_image_data'
        mock_page.get_pixmap.return_value = mock_pix
        
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=2)  # 2ページ
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = Mock()
        mock_fitz.open.return_value = mock_doc
        
        # モックOCR結果
        mock_text_line1 = Mock(text="Text from page 1 line 1")
        mock_text_line2 = Mock(text="Text from page 1 line 2")
        mock_prediction1 = Mock(text_lines=[mock_text_line1, mock_text_line2])
        
        mock_text_line3 = Mock(text="Text from page 2 line 1")
        mock_prediction2 = Mock(text_lines=[mock_text_line3])
        
        mock_run_ocr.return_value = [mock_prediction1, mock_prediction2]
        
        # モデルロード設定
        mock_segformer.load_model.return_value = Mock()
        mock_rec_model.load_model.return_value = Mock()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            result = processor.process_pdf(temp_pdf.name)
            
            assert isinstance(result, OCRResult)
            assert result.model_used == "surya"
            assert result.confidence == 0.80
            assert "Text from page 1 line 1" in result.text
            assert "Text from page 2 line 1" in result.text
            assert result.page_count == 2
    
    @patch('src.ocr_processor.run_ocr', create=True)
    @patch('src.ocr_processor.segformer', create=True)
    @patch('src.ocr_processor.rec_model', create=True)
    def test_process_image_success(self, mock_rec_model, mock_segformer, mock_run_ocr, processor):
        """画像処理成功テスト"""
        # モックOCR結果
        mock_text_line = Mock(text="Image text line")
        mock_prediction = Mock(text_lines=[mock_text_line])
        mock_run_ocr.return_value = [mock_prediction]
        
        # モデルロード設定
        mock_segformer.load_model.return_value = Mock()
        mock_rec_model.load_model.return_value = Mock()
        
        img = Image.new('RGB', (100, 100), color='white')
        
        result = processor.process_image(img)
        assert "Image text line" in result


class TestOCRManager:
    """OCRManager クラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'model': 'unstructured',
            'fallback_models': ['surya', 'nougat'],
            'device': 'cpu',
            'batch_size': 2
        }
    
    @pytest.fixture
    def manager(self, config):
        """OCRManager インスタンス"""
        return OCRManager(config)
    
    def test_init(self, config):
        """初期化テスト"""
        manager = OCRManager(config)
        
        assert manager.primary_model == 'unstructured'
        assert manager.fallback_models == ['surya', 'nougat']
    
    @patch.object(UnstructuredOCRProcessor, 'is_available', return_value=True)
    @patch.object(SuryaOCRProcessor, 'is_available', return_value=True)
    @patch.object(NougatOCRProcessor, 'is_available', return_value=True)
    @patch.object(MistralOCRProcessor, 'is_available', return_value=False)
    def test_init_processors(self, mock_mistral, mock_nougat, mock_surya, mock_unstructured, config):
        """プロセッサー初期化テスト"""
        manager = OCRManager(config)
        
        # 利用可能なプロセッサーが初期化されていることを確認
        assert 'unstructured' in manager.processors
        assert 'surya' in manager.processors
        assert 'nougat' in manager.processors
        assert 'mistral_ocr' not in manager.processors
    
    def test_get_available_models(self, manager):
        """利用可能モデル取得テスト"""
        # モックプロセッサーを追加
        manager.processors = {
            'unstructured': Mock(),
            'surya': Mock()
        }
        
        models = manager.get_available_models()
        assert 'unstructured' in models
        assert 'surya' in models
        assert len(models) == 2
    
    def test_process_pdf_no_processors(self, config):
        """プロセッサーなしでの処理テスト"""
        manager = OCRManager(config)
        manager.processors = {}  # 利用可能プロセッサーなし
        
        with pytest.raises(RuntimeError, match="No OCR processors available"):
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
                manager.process_pdf(temp_pdf.name)
    
    def test_process_pdf_with_fallback(self, manager):
        """フォールバック機能テスト"""
        # プライマリが失敗、フォールバックが成功するシナリオ
        mock_primary = Mock()
        mock_primary.process_pdf.side_effect = Exception("Primary failed")
        
        mock_fallback = Mock()
        mock_fallback.process_pdf.return_value = OCRResult(
            text="Fallback success",
            confidence=0.8,
            processing_time=5.0,
            model_used="fallback_model",
            page_count=1,
            metadata={},
            errors=[]
        )
        
        manager.processors = {
            'primary': mock_primary,
            'fallback': mock_fallback
        }
        manager.primary_model = 'primary'
        manager.fallback_models = ['fallback']
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            result = manager.process_pdf(temp_pdf.name)
            
            assert result.text == "Fallback success"
            assert result.model_used == "fallback_model"
    
    def test_process_pdf_quality_check(self, manager):
        """品質チェック機能テスト"""
        # 品質が低い結果を返すモック
        mock_processor = Mock()
        mock_processor.process_pdf.return_value = OCRResult(
            text="",  # 空のテキスト（低品質）
            confidence=0.8,
            processing_time=5.0,
            model_used="test_model",
            page_count=1,
            metadata={},
            errors=[]
        )
        
        manager.processors = {'test': mock_processor}
        manager.primary_model = 'test'
        manager.fallback_models = []
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            result = manager.process_pdf(temp_pdf.name)
            
            # 品質チェックに失敗した場合の結果
            assert result.text == ""
            assert result.confidence == 0.0
            assert result.model_used == "none"


class TestCreateOCRManager:
    """create_ocr_manager 関数のテスト"""
    
    def test_create_manager(self):
        """マネージャー作成テスト"""
        config = {
            'ocr': {
                'model': 'unstructured',
                'device': 'cpu',
                'batch_size': 4
            }
        }
        
        manager = create_ocr_manager(config)
        
        assert isinstance(manager, OCRManager)
        assert manager.primary_model == 'unstructured'


class TestIntegration:
    """統合テスト"""
    
    @patch.object(UnstructuredOCRProcessor, 'is_available', return_value=True)
    def test_end_to_end_ocr_workflow(self, mock_available):
        """エンドツーエンドOCRワークフローテスト"""
        config = {
            'model': 'unstructured',
            'fallback_models': [],
            'device': 'cpu'
        }
        
        # モックプロセッサーの作成
        mock_processor = Mock(spec=UnstructuredOCRProcessor)
        mock_processor.process_pdf.return_value = OCRResult(
            text="This is a sample research paper about machine learning. The authors present a novel approach to deep neural networks.",
            confidence=0.95,
            processing_time=10.5,
            model_used="unstructured",
            page_count=3,
            metadata={"strategy": "hi_res"},
            errors=[]
        )
        
        manager = OCRManager(config)
        manager.processors = {'unstructured': mock_processor}
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            result = manager.process_pdf(temp_pdf.name)
            
            assert isinstance(result, OCRResult)
            assert "machine learning" in result.text
            assert result.confidence > 0.9
            assert result.model_used == "unstructured"
            assert result.page_count == 3


# Mock 用のフィクスチャ
@pytest.fixture(autouse=True)
def mock_heavy_imports():
    """重いライブラリのインポートをモック"""
    with patch.dict('sys.modules', {
        'torch': Mock(),
        'transformers': Mock(),
        'nougat': Mock(),
        'unstructured': Mock(),
        'surya': Mock(),
        'spacy': Mock(),
        'fitz': Mock(),
    }):
        yield


if __name__ == '__main__':
    pytest.main([__file__])