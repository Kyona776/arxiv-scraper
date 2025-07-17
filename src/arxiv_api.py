"""
arXiv API連携モジュール
arXiv論文の取得、メタデータ抽出、PDFダウンロード機能を提供
"""

import os
import re
import time
import requests
import feedparser
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from loguru import logger

@dataclass
class ArxivPaper:
    """arXiv論文のメタデータを格納するデータクラス"""
    id: str
    title: str
    authors: List[str]
    summary: str
    published: str
    updated: str
    categories: List[str]
    pdf_url: str
    abs_url: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comments: Optional[str] = None

class ArxivAPI:
    """arXiv API連携クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config.get('base_url', 'http://export.arxiv.org/api/query')
        self.rate_limit_config = config.get('rate_limit', {})
        self.pdf_config = config.get('pdf', {})
        self.last_request_time = 0
        
        # レート制限設定
        self.requests_per_second = self.rate_limit_config.get('requests_per_second', 1)
        self.burst_size = self.rate_limit_config.get('burst_size', 3)
        
        # PDF設定
        self.user_agent = self.pdf_config.get('user_agent', 'arXiv-Scraper/1.0')
        self.timeout = self.pdf_config.get('timeout', 30)
        self.max_retries = self.pdf_config.get('max_retries', 3)
        
        # セッションの設定
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent
        })
        
        logger.info(f"ArxivAPI initialized with rate limit: {self.requests_per_second} req/sec")
    
    def _rate_limit(self):
        """レート制限を実装"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _extract_arxiv_id(self, identifier: str) -> str:
        """
        様々な形式のarXiv識別子からIDを抽出
        
        Args:
            identifier: arXiv ID, URL, または論文識別子
            
        Returns:
            正規化されたarXiv ID
        """
        # URL形式の場合
        if identifier.startswith('http'):
            # https://arxiv.org/abs/2301.12345 -> 2301.12345
            # https://arxiv.org/pdf/2301.12345.pdf -> 2301.12345
            match = re.search(r'arxiv\.org/(?:abs|pdf)/([^/?]+)', identifier)
            if match:
                return match.group(1).replace('.pdf', '')
        
        # 直接ID形式の場合
        # arxiv:2301.12345 -> 2301.12345
        if identifier.startswith('arxiv:'):
            return identifier[6:]
        
        # 既に正規化されたIDの場合
        if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', identifier):
            return identifier
        
        # 古い形式のID (math.GT/0309136)
        if re.match(r'^[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$', identifier):
            return identifier
        
        # その他の場合はそのまま返す
        return identifier
    
    def get_paper_metadata(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        arXiv IDから論文のメタデータを取得
        
        Args:
            arxiv_id: arXiv ID
            
        Returns:
            ArxivPaper オブジェクト or None
        """
        try:
            # IDを正規化
            normalized_id = self._extract_arxiv_id(arxiv_id)
            logger.info(f"Fetching metadata for arXiv ID: {normalized_id}")
            
            # レート制限を適用
            self._rate_limit()
            
            # APIクエリを構築
            params = {
                'id_list': normalized_id,
                'max_results': 1
            }
            
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # フィードをパース
            feed = feedparser.parse(response.content)
            
            if not feed.entries:
                logger.warning(f"No entries found for arXiv ID: {normalized_id}")
                return None
            
            entry = feed.entries[0]
            
            # メタデータを抽出
            paper = self._parse_entry(entry)
            logger.info(f"Successfully fetched metadata for: {paper.title}")
            
            return paper
            
        except requests.RequestException as e:
            logger.error(f"Request failed for arXiv ID {arxiv_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching metadata for {arxiv_id}: {str(e)}")
            return None
    
    def _parse_entry(self, entry) -> ArxivPaper:
        """フィードエントリからArxivPaperオブジェクトを作成"""
        
        # IDを抽出 (http://arxiv.org/abs/2301.12345v1 -> 2301.12345v1)
        arxiv_id = entry.id.split('/')[-1]
        
        # 著者リストを抽出
        authors = []
        if hasattr(entry, 'authors'):
            authors = [author.name for author in entry.authors]
        elif hasattr(entry, 'author'):
            authors = [entry.author]
        
        # カテゴリを抽出
        categories = []
        if hasattr(entry, 'arxiv_category'):
            categories = [entry.arxiv_category]
        if hasattr(entry, 'tags'):
            categories.extend([tag.term for tag in entry.tags])
        
        # PDF URLを生成
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        
        # オプション項目を抽出
        doi = getattr(entry, 'arxiv_doi', None)
        journal_ref = getattr(entry, 'arxiv_journal_ref', None)
        comments = getattr(entry, 'arxiv_comment', None)
        
        return ArxivPaper(
            id=arxiv_id,
            title=entry.title,
            authors=authors,
            summary=entry.summary,
            published=entry.published,
            updated=entry.updated,
            categories=categories,
            pdf_url=pdf_url,
            abs_url=abs_url,
            doi=doi,
            journal_ref=journal_ref,
            comments=comments
        )
    
    def download_pdf(self, paper: ArxivPaper, output_dir: str = "data") -> Optional[str]:
        """
        PDFをダウンロード
        
        Args:
            paper: ArxivPaper オブジェクト
            output_dir: 出力ディレクトリ
            
        Returns:
            ダウンロードしたPDFファイルのパス or None
        """
        try:
            # 出力ディレクトリを作成
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # ファイル名を生成
            filename = f"{paper.id}.pdf"
            filepath = output_path / filename
            
            # 既存ファイルをチェック
            if filepath.exists():
                logger.info(f"PDF already exists: {filepath}")
                return str(filepath)
            
            logger.info(f"Downloading PDF: {paper.pdf_url}")
            
            # PDFをダウンロード（リトライ機能付き）
            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(
                        paper.pdf_url, 
                        timeout=self.timeout,
                        stream=True
                    )
                    response.raise_for_status()
                    
                    # ファイルに保存
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info(f"Successfully downloaded: {filepath}")
                    return str(filepath)
                    
                except requests.RequestException as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # 指数バックオフ
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Failed to download PDF for {paper.id}: {str(e)}")
            return None
    
    def search_papers(self, query: str, max_results: int = 10) -> List[ArxivPaper]:
        """
        キーワードで論文を検索
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            
        Returns:
            ArxivPaper オブジェクトのリスト
        """
        try:
            logger.info(f"Searching papers with query: {query}")
            
            # レート制限を適用
            self._rate_limit()
            
            # APIクエリを構築
            params = {
                'search_query': query,
                'max_results': max_results,
                'sortBy': self.config.get('sort_by', 'relevance'),
                'sortOrder': self.config.get('sort_order', 'descending')
            }
            
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # フィードをパース
            feed = feedparser.parse(response.content)
            
            papers = []
            for entry in feed.entries:
                paper = self._parse_entry(entry)
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers")
            return papers
            
        except requests.RequestException as e:
            logger.error(f"Search request failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            return []
    
    def get_paper_from_url(self, url: str) -> Optional[ArxivPaper]:
        """
        URLから論文情報を取得
        
        Args:
            url: arXiv URL
            
        Returns:
            ArxivPaper オブジェクト or None
        """
        try:
            arxiv_id = self._extract_arxiv_id(url)
            return self.get_paper_metadata(arxiv_id)
        except Exception as e:
            logger.error(f"Failed to get paper from URL {url}: {str(e)}")
            return None
    
    def batch_download(self, identifiers: List[str], output_dir: str = "data") -> List[Dict]:
        """
        複数論文のバッチダウンロード
        
        Args:
            identifiers: arXiv ID またはURLのリスト
            output_dir: 出力ディレクトリ
            
        Returns:
            処理結果のリスト
        """
        results = []
        
        for i, identifier in enumerate(identifiers, 1):
            logger.info(f"Processing paper {i}/{len(identifiers)}: {identifier}")
            
            try:
                # メタデータを取得
                paper = self.get_paper_metadata(identifier)
                if not paper:
                    results.append({
                        'identifier': identifier,
                        'status': 'failed',
                        'error': 'Metadata not found',
                        'paper': None,
                        'pdf_path': None
                    })
                    continue
                
                # PDFをダウンロード
                pdf_path = self.download_pdf(paper, output_dir)
                
                results.append({
                    'identifier': identifier,
                    'status': 'success' if pdf_path else 'pdf_failed',
                    'error': None if pdf_path else 'PDF download failed',
                    'paper': paper,
                    'pdf_path': pdf_path
                })
                
            except Exception as e:
                logger.error(f"Failed to process {identifier}: {str(e)}")
                results.append({
                    'identifier': identifier,
                    'status': 'failed',
                    'error': str(e),
                    'paper': None,
                    'pdf_path': None
                })
        
        # 結果をサマリー
        success_count = sum(1 for r in results if r['status'] == 'success')
        logger.info(f"Batch download completed: {success_count}/{len(identifiers)} successful")
        
        return results

def create_arxiv_client(config: Dict) -> ArxivAPI:
    """
    設定からArxivAPIクライアントを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        ArxivAPI インスタンス
    """
    return ArxivAPI(config.get('arxiv', {}))