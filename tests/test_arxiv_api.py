"""
arXiv APIモジュールのユニットテスト
"""

import pytest
import unittest.mock as mock
from unittest.mock import Mock, patch, MagicMock
import requests
import feedparser
from datetime import datetime
import tempfile
import os

from src.arxiv_api import ArxivAPI, ArxivPaper, create_arxiv_client


class TestArxivAPI:
    """ArxivAPIクラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            'base_url': 'http://export.arxiv.org/api/query',
            'rate_limit': {
                'requests_per_second': 2,
                'burst_size': 3
            },
            'pdf': {
                'user_agent': 'test-scraper/1.0',
                'timeout': 30,
                'max_retries': 2
            }
        }
    
    @pytest.fixture
    def arxiv_api(self, config):
        """ArxivAPIインスタンス"""
        return ArxivAPI(config)
    
    def test_init(self, config):
        """初期化テスト"""
        api = ArxivAPI(config)
        
        assert api.base_url == 'http://export.arxiv.org/api/query'
        assert api.requests_per_second == 2
        assert api.burst_size == 3
        assert api.user_agent == 'test-scraper/1.0'
        assert api.timeout == 30
        assert api.max_retries == 2
    
    def test_extract_arxiv_id_from_url(self, arxiv_api):
        """URL形式のID抽出テスト"""
        test_cases = [
            ('https://arxiv.org/abs/2301.12345', '2301.12345'),
            ('https://arxiv.org/pdf/2301.12345.pdf', '2301.12345'),
            ('http://arxiv.org/abs/math.GT/0309136', 'math.GT/0309136'),
            ('arxiv:2301.12345', '2301.12345'),
            ('2301.12345', '2301.12345'),
            ('math.GT/0309136', 'math.GT/0309136'),
        ]
        
        for input_id, expected in test_cases:
            result = arxiv_api._extract_arxiv_id(input_id)
            assert result == expected, f"Failed for input: {input_id}"
    
    def test_rate_limiting(self, arxiv_api):
        """レート制限テスト"""
        import time
        
        # レート制限が適用されることを確認
        start_time = time.time()
        arxiv_api._rate_limit()
        arxiv_api._rate_limit()
        end_time = time.time()
        
        # 2回の呼び出しで少なくとも0.5秒（1/2 requests_per_second）かかることを確認
        elapsed = end_time - start_time
        assert elapsed >= 0.4, f"Rate limiting not working: elapsed={elapsed}"
    
    @patch('src.arxiv_api.feedparser.parse')
    @patch('requests.Session.get')
    def test_get_paper_metadata_success(self, mock_get, mock_parse, arxiv_api):
        """論文メタデータ取得成功テスト"""
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.content = b'<feed>mock feed content</feed>'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # フィードパーサーのモックを設定
        mock_entry = Mock()
        mock_entry.id = 'http://arxiv.org/abs/2301.12345v1'
        mock_entry.title = 'Test Paper Title'
        mock_entry.summary = 'Test abstract content'
        mock_entry.published = '2023-01-30T18:00:00Z'
        mock_entry.updated = '2023-01-30T18:00:00Z'
        mock_entry.authors = [Mock(name='Author One'), Mock(name='Author Two')]
        mock_entry.tags = [Mock(term='cs.LG'), Mock(term='cs.AI')]
        
        mock_feed = Mock()
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed
        
        # テスト実行
        result = arxiv_api.get_paper_metadata('2301.12345')
        
        # 結果検証
        assert result is not None
        assert isinstance(result, ArxivPaper)
        assert result.id == '2301.12345v1'
        assert result.title == 'Test Paper Title'
        assert result.summary == 'Test abstract content'
        assert len(result.authors) == 2
        assert 'cs.LG' in result.categories
        assert result.pdf_url == 'https://arxiv.org/pdf/2301.12345v1.pdf'
        assert result.abs_url == 'https://arxiv.org/abs/2301.12345v1'
    
    @patch('src.arxiv_api.feedparser.parse')
    @patch('requests.Session.get')
    def test_get_paper_metadata_not_found(self, mock_get, mock_parse, arxiv_api):
        """論文が見つからない場合のテスト"""
        # 空のフィードを返すモック
        mock_response = Mock()
        mock_response.content = b'<feed>empty feed</feed>'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        mock_feed = Mock()
        mock_feed.entries = []
        mock_parse.return_value = mock_feed
        
        # テスト実行
        result = arxiv_api.get_paper_metadata('invalid.id')
        
        # 結果検証
        assert result is None
    
    @patch('requests.Session.get')
    def test_get_paper_metadata_network_error(self, mock_get, arxiv_api):
        """ネットワークエラーのテスト"""
        # ネットワークエラーを発生させる
        mock_get.side_effect = requests.RequestException("Network error")
        
        # テスト実行
        result = arxiv_api.get_paper_metadata('2301.12345')
        
        # 結果検証
        assert result is None
    
    @patch('builtins.open', new_callable=mock.mock_open)
    @patch('requests.Session.get')
    def test_download_pdf_success(self, mock_get, mock_file, arxiv_api):
        """PDF ダウンロード成功テスト"""
        # テスト用の論文オブジェクト
        paper = ArxivPaper(
            id='2301.12345',
            title='Test Paper',
            authors=['Author One'],
            summary='Test summary',
            published='2023-01-30T18:00:00Z',
            updated='2023-01-30T18:00:00Z',
            categories=['cs.LG'],
            pdf_url='https://arxiv.org/pdf/2301.12345.pdf',
            abs_url='https://arxiv.org/abs/2301.12345'
        )
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'pdf content chunk 1', b'pdf content chunk 2'])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # テンポラリディレクトリを使用
        with tempfile.TemporaryDirectory() as temp_dir:
            result = arxiv_api.download_pdf(paper, temp_dir)
            
            # 結果検証
            assert result is not None
            assert result.endswith('2301.12345.pdf')
            assert temp_dir in result
    
    @patch('requests.Session.get')
    def test_download_pdf_failure(self, mock_get, arxiv_api):
        """PDF ダウンロード失敗テスト"""
        # テスト用の論文オブジェクト
        paper = ArxivPaper(
            id='2301.12345',
            title='Test Paper',
            authors=['Author One'],
            summary='Test summary',
            published='2023-01-30T18:00:00Z',
            updated='2023-01-30T18:00:00Z',
            categories=['cs.LG'],
            pdf_url='https://arxiv.org/pdf/2301.12345.pdf',
            abs_url='https://arxiv.org/abs/2301.12345'
        )
        
        # ダウンロードエラーを発生させる
        mock_get.side_effect = requests.RequestException("Download failed")
        
        # テスト実行
        with tempfile.TemporaryDirectory() as temp_dir:
            result = arxiv_api.download_pdf(paper, temp_dir)
            
            # 結果検証
            assert result is None
    
    @patch('src.arxiv_api.feedparser.parse')
    @patch('requests.Session.get')
    def test_search_papers(self, mock_get, mock_parse, arxiv_api):
        """論文検索テスト"""
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.content = b'<feed>search results</feed>'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # 検索結果のモック
        mock_entries = []
        for i in range(3):
            entry = Mock()
            entry.id = f'http://arxiv.org/abs/230{i}.12345v1'
            entry.title = f'Test Paper {i}'
            entry.summary = f'Test abstract {i}'
            entry.published = '2023-01-30T18:00:00Z'
            entry.updated = '2023-01-30T18:00:00Z'
            entry.authors = [Mock(name=f'Author {i}')]
            entry.tags = [Mock(term='cs.LG')]
            mock_entries.append(entry)
        
        mock_feed = Mock()
        mock_feed.entries = mock_entries
        mock_parse.return_value = mock_feed
        
        # テスト実行
        results = arxiv_api.search_papers('machine learning', max_results=3)
        
        # 結果検証
        assert len(results) == 3
        assert all(isinstance(paper, ArxivPaper) for paper in results)
        assert results[0].title == 'Test Paper 0'
        assert results[2].title == 'Test Paper 2'
    
    def test_get_paper_from_url(self, arxiv_api):
        """URL から論文取得テスト"""
        with patch.object(arxiv_api, 'get_paper_metadata') as mock_get_metadata:
            mock_paper = Mock(spec=ArxivPaper)
            mock_get_metadata.return_value = mock_paper
            
            result = arxiv_api.get_paper_from_url('https://arxiv.org/abs/2301.12345')
            
            assert result == mock_paper
            mock_get_metadata.assert_called_once_with('2301.12345')
    
    def test_batch_download(self, arxiv_api):
        """バッチダウンロードテスト"""
        identifiers = ['2301.12345', '2301.12346', 'invalid.id']
        
        with patch.object(arxiv_api, 'get_paper_metadata') as mock_get_metadata, \
             patch.object(arxiv_api, 'download_pdf') as mock_download:
            
            # 最初の2つは成功、3つ目は失敗
            mock_papers = [Mock(spec=ArxivPaper), Mock(spec=ArxivPaper), None]
            mock_get_metadata.side_effect = mock_papers
            
            mock_download.side_effect = ['/path/to/file1.pdf', '/path/to/file2.pdf']
            
            with tempfile.TemporaryDirectory() as temp_dir:
                results = arxiv_api.batch_download(identifiers, temp_dir)
            
            # 結果検証
            assert len(results) == 3
            assert results[0]['status'] == 'success'
            assert results[1]['status'] == 'success'
            assert results[2]['status'] == 'failed'
            assert results[2]['error'] == 'Metadata not found'


class TestCreateArxivClient:
    """create_arxiv_client 関数のテスト"""
    
    def test_create_client(self):
        """クライアント作成テスト"""
        config = {
            'arxiv': {
                'base_url': 'http://test.arxiv.org',
                'rate_limit': {'requests_per_second': 1}
            }
        }
        
        client = create_arxiv_client(config)
        
        assert isinstance(client, ArxivAPI)
        assert client.base_url == 'http://test.arxiv.org'
        assert client.requests_per_second == 1


class TestArxivPaper:
    """ArxivPaper データクラスのテスト"""
    
    def test_arxiv_paper_creation(self):
        """ArxivPaper オブジェクト作成テスト"""
        paper = ArxivPaper(
            id='2301.12345',
            title='Test Paper Title',
            authors=['Author One', 'Author Two'],
            summary='Test abstract content',
            published='2023-01-30T18:00:00Z',
            updated='2023-01-30T18:00:00Z',
            categories=['cs.LG', 'cs.AI'],
            pdf_url='https://arxiv.org/pdf/2301.12345.pdf',
            abs_url='https://arxiv.org/abs/2301.12345',
            doi='10.1000/test.doi',
            journal_ref='Test Journal 2023',
            comments='Test comments'
        )
        
        assert paper.id == '2301.12345'
        assert paper.title == 'Test Paper Title'
        assert len(paper.authors) == 2
        assert 'Author One' in paper.authors
        assert len(paper.categories) == 2
        assert paper.doi == '10.1000/test.doi'
        assert paper.journal_ref == 'Test Journal 2023'
        assert paper.comments == 'Test comments'
    
    def test_arxiv_paper_minimal(self):
        """最小限のArxivPaper オブジェクト作成テスト"""
        paper = ArxivPaper(
            id='2301.12345',
            title='Test Paper Title',
            authors=['Author One'],
            summary='Test abstract content',
            published='2023-01-30T18:00:00Z',
            updated='2023-01-30T18:00:00Z',
            categories=['cs.LG'],
            pdf_url='https://arxiv.org/pdf/2301.12345.pdf',
            abs_url='https://arxiv.org/abs/2301.12345'
        )
        
        assert paper.doi is None
        assert paper.journal_ref is None
        assert paper.comments is None


# Integration test helpers
@pytest.fixture
def sample_arxiv_feed():
    """サンプルのarXivフィード（実際の形式に近い）"""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <updated>2023-01-30T18:00:00Z</updated>
    <published>2023-01-30T18:00:00Z</published>
    <title>Sample Research Paper Title</title>
    <summary>This is a sample abstract content for testing purposes.</summary>
    <author>
      <name>John Doe</name>
    </author>
    <author>
      <name>Jane Smith</name>
    </author>
    <arxiv:category xmlns:arxiv="http://arxiv.org/schemas/atom" scheme="http://arxiv.org/schemas/atom" term="cs.LG"/>
    <arxiv:category xmlns:arxiv="http://arxiv.org/schemas/atom" scheme="http://arxiv.org/schemas/atom" term="cs.AI"/>
  </entry>
</feed>'''


class TestIntegration:
    """統合テスト"""
    
    @patch('requests.Session.get')
    def test_full_paper_retrieval_workflow(self, mock_get, sample_arxiv_feed):
        """完全な論文取得ワークフローのテスト"""
        # モックレスポンス設定
        mock_response = Mock()
        mock_response.content = sample_arxiv_feed.encode()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # ArxivAPI インスタンス作成
        config = {
            'base_url': 'http://export.arxiv.org/api/query',
            'rate_limit': {'requests_per_second': 10},
            'pdf': {'timeout': 30, 'max_retries': 3}
        }
        api = ArxivAPI(config)
        
        # 論文メタデータ取得
        paper = api.get_paper_metadata('2301.12345')
        
        # 結果検証
        assert paper is not None
        assert paper.id == '2301.12345v1'
        assert 'Sample Research Paper Title' in paper.title
        assert len(paper.authors) == 2
        assert 'John Doe' in paper.authors


if __name__ == '__main__':
    pytest.main([__file__])