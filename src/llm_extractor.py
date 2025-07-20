"""
LLM解析モジュール
論文テキストから9つの指定項目を抽出
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from loguru import logger

from openrouter_manager import create_openrouter_manager, OpenRouterManager


@dataclass
class ExtractionItem:
    """抽出項目の定義"""
    name: str
    description: str
    max_length: int
    required: bool = True


@dataclass
class ExtractionResult:
    """抽出結果"""
    success: bool
    data: Dict[str, str]
    model_used: str
    processing_time: float
    confidence_scores: Dict[str, float]
    errors: List[str]
    metadata: Dict[str, Any]


class LLMProvider(ABC):
    """LLMプロバイダーの抽象基底クラス"""
    
    @abstractmethod
    def extract_information(
        self, 
        text: str, 
        items: List[ExtractionItem],
        paper_metadata: Optional[Dict] = None
    ) -> ExtractionResult:
        """情報抽出を実行"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """プロバイダーが利用可能かチェック"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPTプロバイダー"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 2000)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        # API key設定
        self.api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not found")
        
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """OpenAIクライアントを初期化"""
        try:
            import openai
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model: {self.model}")
            else:
                logger.warning("OpenAI client not initialized: missing API key")
        except ImportError:
            logger.error("OpenAI library not installed")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def is_available(self) -> bool:
        """OpenAIが利用可能かチェック"""
        return self.client is not None and self.api_key is not None
    
    def extract_information(
        self, 
        text: str, 
        items: List[ExtractionItem],
        paper_metadata: Optional[Dict] = None
    ) -> ExtractionResult:
        """GPTを使用した情報抽出"""
        start_time = time.time()
        
        if not self.is_available():
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=0,
                confidence_scores={},
                errors=["OpenAI client not available"],
                metadata={}
            )
        
        try:
            # プロンプト生成
            prompt = self._create_extraction_prompt(text, items, paper_metadata)
            
            # GPT API呼び出し（リトライ機能付き）
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "あなたは学術論文分析の専門家です。与えられた論文から指定された情報を正確に抽出してください。"
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"}
                    )
                    
                    # レスポンス解析
                    result = self._parse_response(response, items)
                    result.processing_time = time.time() - start_time
                    result.model_used = self.model
                    
                    logger.info(f"OpenAI extraction completed in {result.processing_time:.2f}s")
                    return result
                    
                except Exception as e:
                    logger.warning(f"OpenAI API attempt {attempt + 1} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # 指数バックオフ
                    else:
                        raise
        
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=time.time() - start_time,
                confidence_scores={},
                errors=[str(e)],
                metadata={}
            )
    
    def _create_extraction_prompt(
        self, 
        text: str, 
        items: List[ExtractionItem],
        paper_metadata: Optional[Dict] = None
    ) -> str:
        """抽出用プロンプトを生成"""
        
        # 項目説明の作成
        items_description = []
        for item in items:
            required_str = "必須" if item.required else "任意"
            items_description.append(
                f"- {item.name}: {item.description} "
                f"(最大{item.max_length}文字, {required_str})"
            )
        
        items_list = "\n".join(items_description)
        
        # メタデータ情報の追加
        metadata_str = ""
        if paper_metadata:
            metadata_str = f"""
論文メタデータ:
- タイトル: {paper_metadata.get('title', 'N/A')}
- 著者: {', '.join(paper_metadata.get('authors', []))}
- 分野: {', '.join(paper_metadata.get('categories', []))}
- 発表年: {paper_metadata.get('published', 'N/A')[:4] if paper_metadata.get('published') else 'N/A'}
- URL: {paper_metadata.get('abs_url', 'N/A')}
"""
        
        prompt = f"""以下の学術論文テキストから、指定された項目を抽出してください。

{metadata_str}

抽出対象項目:
{items_list}

抽出ルール:
1. 各項目について、論文の内容を正確に読み取り、適切な情報を抽出してください
2. 情報が明記されていない場合は "N/A" と回答してください
3. 指定された文字数制限を守ってください
4. 日本語で回答してください
5. 結果は以下のJSON形式で出力してください:

{{
  "手法の肝": "抽出した内容",
  "制限事項": "抽出した内容",
  "対象ナレッジ": "抽出した内容",
  "URL": "抽出した内容",
  "タイトル": "抽出した内容",
  "出版年": "抽出した内容",
  "研究分野": "抽出した内容",
  "課題設定": "抽出した内容",
  "論文の主張": "抽出した内容"
}}

論文テキスト:
{text[:10000]}  # 最初の10000文字に制限
"""
        
        return prompt
    
    def _parse_response(
        self, 
        response, 
        items: List[ExtractionItem]
    ) -> ExtractionResult:
        """API レスポンスを解析"""
        
        try:
            content = response.choices[0].message.content
            
            # JSON解析
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # JSON解析失敗時は文字列から抽出を試行
                data = self._extract_from_text(content, items)
            
            # データ検証と正規化
            validated_data = {}
            confidence_scores = {}
            errors = []
            
            for item in items:
                value = data.get(item.name, "N/A")
                
                # 文字数制限チェック
                if len(value) > item.max_length:
                    value = value[:item.max_length] + "..."
                    errors.append(f"{item.name}が文字数制限を超過（切り詰め）")
                
                # 必須項目チェック
                if item.required and (not value or value == "N/A"):
                    errors.append(f"必須項目{item.name}が未抽出")
                    confidence_scores[item.name] = 0.0
                else:
                    confidence_scores[item.name] = self._calculate_confidence(value)
                
                validated_data[item.name] = value
            
            # 全体の成功判定
            success = len([e for e in errors if "必須項目" in e]) == 0
            
            return ExtractionResult(
                success=success,
                data=validated_data,
                model_used=self.model,
                processing_time=0,  # 後で設定
                confidence_scores=confidence_scores,
                errors=errors,
                metadata={
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Response parsing failed: {str(e)}")
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=0,
                confidence_scores={},
                errors=[f"レスポンス解析エラー: {str(e)}"],
                metadata={}
            )
    
    def _extract_from_text(self, text: str, items: List[ExtractionItem]) -> Dict[str, str]:
        """テキストから直接抽出（JSON解析失敗時のフォールバック）"""
        data = {}
        
        for item in items:
            # 簡単なパターンマッチングで抽出を試行
            pattern_variations = [
                f'"{item.name}"\\s*:\\s*"([^"]*)"',
                f'{item.name}\\s*:\\s*([^\n]*)',
                f'{item.name}\\s*は\\s*([^\n]*)',
            ]
            
            import re
            for pattern in pattern_variations:
                match = re.search(pattern, text)
                if match:
                    data[item.name] = match.group(1).strip()
                    break
            else:
                data[item.name] = "N/A"
        
        return data
    
    def _calculate_confidence(self, value: str) -> float:
        """抽出結果の信頼度を計算"""
        if not value or value == "N/A":
            return 0.0
        
        # 簡単な信頼度計算
        score = 0.5  # ベーススコア
        
        # 長さによる加点
        if len(value) > 10:
            score += 0.2
        if len(value) > 50:
            score += 0.2
        
        # 具体性による加点
        if any(keyword in value.lower() for keyword in ['手法', '方法', 'algorithm', 'method']):
            score += 0.1
        
        return min(score, 1.0)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude プロバイダー"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = config.get('model', 'claude-3-sonnet-20240229')
        self.max_tokens = config.get('max_tokens', 2000)
        self.temperature = config.get('temperature', 0.1)
        
        # API key設定
        self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("Anthropic API key not found")
        
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Anthropicクライアントを初期化"""
        try:
            import anthropic
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Anthropic client initialized with model: {self.model}")
            else:
                logger.warning("Anthropic client not initialized: missing API key")
        except ImportError:
            logger.error("Anthropic library not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
    
    def is_available(self) -> bool:
        """Anthropicが利用可能かチェック"""
        return self.client is not None and self.api_key is not None
    
    def extract_information(
        self, 
        text: str, 
        items: List[ExtractionItem],
        paper_metadata: Optional[Dict] = None
    ) -> ExtractionResult:
        """Claudeを使用した情報抽出"""
        start_time = time.time()
        
        if not self.is_available():
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=0,
                confidence_scores={},
                errors=["Anthropic client not available"],
                metadata={}
            )
        
        try:
            # プロンプト生成（OpenAIと同様）
            prompt = self._create_extraction_prompt(text, items, paper_metadata)
            
            # Claude API呼び出し
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # レスポンス解析（OpenAIと同様の処理）
            result = self._parse_claude_response(response, items)
            result.processing_time = time.time() - start_time
            result.model_used = self.model
            
            logger.info(f"Anthropic extraction completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Anthropic extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=time.time() - start_time,
                confidence_scores={},
                errors=[str(e)],
                metadata={}
            )
    
    def _create_extraction_prompt(self, text: str, items: List[ExtractionItem], paper_metadata: Optional[Dict] = None) -> str:
        """OpenAIプロバイダーと同じプロンプト生成ロジック"""
        # OpenAIProviderと同じ実装を使用
        provider = OpenAIProvider({})
        return provider._create_extraction_prompt(text, items, paper_metadata)
    
    def _parse_claude_response(self, response, items: List[ExtractionItem]) -> ExtractionResult:
        """Claude レスポンスを解析"""
        try:
            content = response.content[0].text
            
            # JSON解析を試行
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # OpenAIProviderと同じフォールバック処理
                data = self._extract_from_text(content, items)
            
            # データ検証（OpenAIProviderと同様）
            validated_data = {}
            confidence_scores = {}
            errors = []
            
            for item in items:
                value = data.get(item.name, "N/A")
                
                if len(value) > item.max_length:
                    value = value[:item.max_length] + "..."
                    errors.append(f"{item.name}が文字数制限を超過（切り詰め）")
                
                if item.required and (not value or value == "N/A"):
                    errors.append(f"必須項目{item.name}が未抽出")
                    confidence_scores[item.name] = 0.0
                else:
                    confidence_scores[item.name] = self._calculate_confidence(value)
                
                validated_data[item.name] = value
            
            success = len([e for e in errors if "必須項目" in e]) == 0
            
            return ExtractionResult(
                success=success,
                data=validated_data,
                model_used=self.model,
                processing_time=0,
                confidence_scores=confidence_scores,
                errors=errors,
                metadata={
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Claude response parsing failed: {str(e)}")
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=0,
                confidence_scores={},
                errors=[f"レスポンス解析エラー: {str(e)}"],
                metadata={}
            )
    
    def _extract_from_text(self, text: str, items: List[ExtractionItem]) -> Dict[str, str]:
        """OpenAIProviderと同じフォールバック処理"""
        provider = OpenAIProvider({})
        return provider._extract_from_text(text, items)
    
    def _calculate_confidence(self, value: str) -> float:
        """OpenAIProviderと同じ信頼度計算"""
        provider = OpenAIProvider({})
        return provider._calculate_confidence(value)


class OpenRouterProvider(LLMProvider):
    """OpenRouter API プロバイダー"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = config.get('model', 'anthropic/claude-3-sonnet')
        self.base_url = config.get('base_url', 'https://openrouter.ai/api/v1')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 2000)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.timeout = config.get('timeout', 120)
        self.site_url = config.get('site_url', 'https://arxiv-scraper.local')
        self.site_name = config.get('site_name', 'ArXiv Scraper')
        
        # API key設定
        self.api_key = config.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            logger.warning("OpenRouter API key not found")
        
        # OpenRouter manager for model validation and info
        self.openrouter_manager = None
        try:
            self.openrouter_manager = create_openrouter_manager({'openrouter': config})
            logger.info("OpenRouter manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter manager: {e}")
        
        # Validate model on initialization
        self._validate_model()
    
    def _validate_model(self):
        """Validate the configured model"""
        if not self.openrouter_manager or not self.openrouter_manager.is_available():
            return
        
        try:
            if self.openrouter_manager.validate_model(self.model):
                logger.info(f"Model {self.model} validated successfully")
            else:
                logger.warning(f"Model {self.model} not found in OpenRouter. Using anyway.")
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
    
    def is_available(self) -> bool:
        """OpenRouterが利用可能かチェック"""
        return bool(self.api_key)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.openrouter_manager or not self.openrouter_manager.is_available():
            return {'error': 'OpenRouter manager not available'}
        
        try:
            return self.openrouter_manager.get_model_info(self.model)
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        if not self.openrouter_manager or not self.openrouter_manager.is_available():
            return []
        
        try:
            models = self.openrouter_manager.get_models()
            return [model.id for model in models]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def get_model_recommendations(self, task_type: str = 'general', budget: Optional[float] = None) -> List[str]:
        """Get model recommendations"""
        if not self.openrouter_manager or not self.openrouter_manager.is_available():
            return []
        
        try:
            models = self.openrouter_manager.get_recommendations(task_type, budget)
            return [model.id for model in models]
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []
    
    def extract_information(
        self, 
        text: str, 
        items: List[ExtractionItem],
        paper_metadata: Optional[Dict] = None
    ) -> ExtractionResult:
        """OpenRouterを使用した情報抽出"""
        start_time = time.time()
        
        if not self.is_available():
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=0,
                confidence_scores={},
                errors=["OpenRouter API key not available"],
                metadata={}
            )
        
        try:
            # プロンプト生成
            prompt = self._create_extraction_prompt(text, items, paper_metadata)
            
            # OpenRouter API呼び出し（リトライ機能付き）
            for attempt in range(self.max_retries):
                try:
                    response = self._call_openrouter_api(prompt)
                    
                    # レスポンス解析
                    result = self._parse_openrouter_response(response, items)
                    result.processing_time = time.time() - start_time
                    result.model_used = self.model
                    
                    logger.info(f"OpenRouter extraction completed in {result.processing_time:.2f}s")
                    return result
                    
                except Exception as e:
                    logger.warning(f"OpenRouter API attempt {attempt + 1} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # 指数バックオフ
                    else:
                        raise
        
        except Exception as e:
            logger.error(f"OpenRouter extraction failed: {str(e)}")
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=time.time() - start_time,
                confidence_scores={},
                errors=[str(e)],
                metadata={}
            )
    
    def _call_openrouter_api(self, prompt: str) -> Dict:
        """OpenRouter APIを呼び出し"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': self.site_url,
            'X-Title': self.site_name
        }
        
        # システムプロンプトとユーザープロンプトに分割
        system_prompt = "あなたは学術論文分析の専門家です。与えられた論文から指定された情報を正確に抽出してください。必ずJSON形式で回答してください。"
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stream': False
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code == 429:
            raise Exception("Rate limit exceeded")
        elif response.status_code != 200:
            response.raise_for_status()
        
        return response.json()
    
    def _parse_openrouter_response(self, response: Dict, items: List[ExtractionItem]) -> ExtractionResult:
        """OpenRouter APIレスポンスを解析"""
        try:
            if 'choices' not in response or not response['choices']:
                raise Exception("Invalid response format")
            
            content = response['choices'][0]['message']['content']
            
            # JSON解析
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # JSON解析失敗時はフォールバック処理
                data = self._extract_from_text(content, items)
            
            # データ検証と正規化
            validated_data = {}
            confidence_scores = {}
            errors = []
            
            for item in items:
                value = data.get(item.name, "N/A")
                
                # 文字数制限チェック
                if len(value) > item.max_length:
                    value = value[:item.max_length] + "..."
                    errors.append(f"{item.name}が文字数制限を超過（切り詰め）")
                
                # 必須項目チェック
                if item.required and (not value or value == "N/A"):
                    errors.append(f"必須項目{item.name}が未抽出")
                    confidence_scores[item.name] = 0.0
                else:
                    confidence_scores[item.name] = self._calculate_confidence(value)
                
                validated_data[item.name] = value
            
            # 全体の成功判定
            success = len([e for e in errors if "必須項目" in e]) == 0
            
            # メタデータの取得
            metadata = {}
            if 'usage' in response:
                metadata['usage'] = response['usage']
            
            return ExtractionResult(
                success=success,
                data=validated_data,
                model_used=self.model,
                processing_time=0,  # 後で設定
                confidence_scores=confidence_scores,
                errors=errors,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"OpenRouter response parsing failed: {str(e)}")
            return ExtractionResult(
                success=False,
                data={},
                model_used=self.model,
                processing_time=0,
                confidence_scores={},
                errors=[f"レスポンス解析エラー: {str(e)}"],
                metadata={}
            )
    
    def _create_extraction_prompt(
        self, 
        text: str, 
        items: List[ExtractionItem],
        paper_metadata: Optional[Dict] = None
    ) -> str:
        """抽出用プロンプトを生成"""
        # 項目説明の作成
        items_description = []
        for item in items:
            required_str = "必須" if item.required else "任意"
            items_description.append(
                f"- {item.name}: {item.description} "
                f"(最大{item.max_length}文字, {required_str})"
            )
        
        items_list = "\n".join(items_description)
        
        # メタデータ情報の追加
        metadata_str = ""
        if paper_metadata:
            metadata_str = f"""
論文メタデータ:
- タイトル: {paper_metadata.get('title', 'N/A')}
- 著者: {', '.join(paper_metadata.get('authors', []))}
- 分野: {', '.join(paper_metadata.get('categories', []))}
- 発表年: {paper_metadata.get('published', 'N/A')[:4] if paper_metadata.get('published') else 'N/A'}
- URL: {paper_metadata.get('abs_url', 'N/A')}
"""
        
        prompt = f"""以下の学術論文テキストから、指定された項目を抽出してください。

{metadata_str}

抽出対象項目:
{items_list}

抽出ルール:
1. 各項目について、論文の内容を正確に読み取り、適切な情報を抽出してください
2. 情報が明記されていない場合は "N/A" と回答してください
3. 指定された文字数制限を守ってください
4. 日本語で回答してください
5. 結果は以下のJSON形式で出力してください:

{{
  "手法の肝": "抽出した内容",
  "制限事項": "抽出した内容",
  "対象ナレッジ": "抽出した内容",
  "URL": "抽出した内容",
  "タイトル": "抽出した内容",
  "出版年": "抽出した内容",
  "研究分野": "抽出した内容",
  "課題設定": "抽出した内容",
  "論文の主張": "抽出した内容"
}}

論文テキスト:
{text[:10000]}  # 最初の10000文字に制限
"""
        
        return prompt
    
    def _extract_from_text(self, text: str, items: List[ExtractionItem]) -> Dict[str, str]:
        """テキストから直接抽出（JSON解析失敗時のフォールバック）"""
        data = {}
        
        for item in items:
            # 簡単なパターンマッチングで抽出を試行
            pattern_variations = [
                f'"{item.name}"\\s*:\\s*"([^"]*)"',
                f'{item.name}\\s*:\\s*([^\n]*)',
                f'{item.name}\\s*は\\s*([^\n]*)',
            ]
            
            import re
            for pattern in pattern_variations:
                match = re.search(pattern, text)
                if match:
                    data[item.name] = match.group(1).strip()
                    break
            else:
                data[item.name] = "N/A"
        
        return data
    
    def _calculate_confidence(self, value: str) -> float:
        """抽出結果の信頼度を計算"""
        if not value or value == "N/A":
            return 0.0
        
        # 簡単な信頼度計算
        score = 0.5  # ベーススコア
        
        # 長さによる加点
        if len(value) > 10:
            score += 0.2
        if len(value) > 50:
            score += 0.2
        
        # 具体性による加点
        if any(keyword in value.lower() for keyword in ['手法', '方法', 'algorithm', 'method']):
            score += 0.1
        
        return min(score, 1.0)


class LLMExtractor:
    """LLM情報抽出メインクラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.primary_provider = config.get('model', 'gpt-4')
        self.fallback_models = config.get('fallback_models', [])
        
        # 抽出項目を設定から読み込み
        self.extraction_items = self._load_extraction_items(config)
        
        # プロバイダーを初期化
        self.providers = {}
        self._init_providers()
        
        logger.info(f"LLMExtractor initialized with {len(self.providers)} providers")
    
    def _load_extraction_items(self, config: Dict) -> List[ExtractionItem]:
        """設定から抽出項目をロード"""
        items_config = config.get('extraction_items', [])
        
        if not items_config:
            # デフォルトの9項目
            items_config = [
                {
                    "name": "手法の肝",
                    "description": "論文の核となる技術手法・アプローチ",
                    "max_length": 500
                },
                {
                    "name": "制限事項", 
                    "description": "手法の限界や制約条件",
                    "max_length": 500
                },
                {
                    "name": "対象ナレッジ",
                    "description": "扱う知識領域・データ種別", 
                    "max_length": 500
                },
                {
                    "name": "URL",
                    "description": "arXiv論文のURL",
                    "max_length": 200
                },
                {
                    "name": "タイトル",
                    "description": "論文タイトル",
                    "max_length": 300
                },
                {
                    "name": "出版年",
                    "description": "発表年",
                    "max_length": 50
                },
                {
                    "name": "研究分野", 
                    "description": "分野分類",
                    "max_length": 200
                },
                {
                    "name": "課題設定",
                    "description": "解決しようとする問題",
                    "max_length": 500
                },
                {
                    "name": "論文の主張",
                    "description": "主要な貢献・結論", 
                    "max_length": 500
                }
            ]
        
        return [
            ExtractionItem(
                name=item["name"],
                description=item["description"], 
                max_length=item["max_length"],
                required=item.get("required", True)
            )
            for item in items_config
        ]
    
    def _init_providers(self):
        """LLMプロバイダーを初期化"""
        
        # OpenAI provider
        if 'openai' in self.config:
            try:
                provider = OpenAIProvider(self.config['openai'])
                if provider.is_available():
                    self.providers['gpt-4'] = provider
                    self.providers['gpt-3.5-turbo'] = provider
                    logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
        
        # Anthropic provider
        if 'anthropic' in self.config:
            try:
                provider = AnthropicProvider(self.config['anthropic'])
                if provider.is_available():
                    self.providers['claude-3-opus'] = provider
                    self.providers['claude-3-sonnet'] = provider
                    logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic provider: {str(e)}")
        
        # OpenRouter provider
        if 'openrouter' in self.config:
            try:
                provider = OpenRouterProvider(self.config['openrouter'])
                if provider.is_available():
                    # OpenRouter supports many models - add the ones specified in config
                    model_name = self.config['openrouter'].get('model', 'anthropic/claude-3-sonnet')
                    self.providers[model_name] = provider
                    
                    # Add common OpenRouter model aliases
                    common_models = {
                        'anthropic/claude-3-opus': provider,
                        'anthropic/claude-3-sonnet': provider,
                        'anthropic/claude-3-haiku': provider,
                        'openai/gpt-4': provider,
                        'openai/gpt-3.5-turbo': provider,
                        'google/gemini-pro': provider,
                        'mistral/mistral-large': provider,
                        'mistral/mistral-medium': provider,
                        'meta/llama-3-70b': provider,
                        'cohere/command-r-plus': provider
                    }
                    
                    # Only add models that are not already registered
                    for model, prov in common_models.items():
                        if model not in self.providers:
                            self.providers[model] = prov
                    
                    logger.info(f"OpenRouter provider initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenRouter provider: {str(e)}")
    
    def extract_from_text(
        self, 
        text: str, 
        paper_metadata: Optional[Dict] = None
    ) -> ExtractionResult:
        """テキストから情報を抽出"""
        
        if not text.strip():
            return ExtractionResult(
                success=False,
                data={},
                model_used="none",
                processing_time=0,
                confidence_scores={},
                errors=["入力テキストが空です"],
                metadata={}
            )
        
        # 試行するモデルの順序を決定
        models_to_try = [self.primary_provider] + self.fallback_models
        models_to_try = [m for m in models_to_try if m in self.providers]
        
        if not models_to_try:
            return ExtractionResult(
                success=False,
                data={},
                model_used="none", 
                processing_time=0,
                confidence_scores={},
                errors=["利用可能なLLMプロバイダーがありません"],
                metadata={}
            )
        
        logger.info(f"Starting extraction with {len(models_to_try)} models to try")
        
        # メタデータの前処理
        if paper_metadata:
            # URLとタイトルはメタデータから直接取得
            metadata_values = {
                "URL": paper_metadata.get('abs_url', 'N/A'),
                "タイトル": paper_metadata.get('title', 'N/A'),
                "出版年": paper_metadata.get('published', 'N/A')[:4] if paper_metadata.get('published') else 'N/A',
                "研究分野": ', '.join(paper_metadata.get('categories', []))[:200] if paper_metadata.get('categories') else 'N/A'
            }
        else:
            metadata_values = {}
        
        # モデルを順番に試行
        for model_name in models_to_try:
            try:
                logger.info(f"Trying extraction with {model_name}")
                provider = self.providers[model_name]
                
                result = provider.extract_information(text, self.extraction_items, paper_metadata)
                
                # メタデータから取得できる値を上書き
                for key, value in metadata_values.items():
                    if value != 'N/A':
                        result.data[key] = value
                        result.confidence_scores[key] = 1.0
                
                # 品質チェック
                if self._quality_check(result):
                    logger.info(f"Extraction successful with {model_name}")
                    return result
                else:
                    logger.warning(f"Quality check failed for {model_name}")
                    continue
                    
            except Exception as e:
                logger.error(f"Extraction failed with {model_name}: {str(e)}")
                continue
        
        # すべて失敗した場合
        logger.error("All extraction attempts failed")
        return ExtractionResult(
            success=False,
            data={item.name: "N/A" for item in self.extraction_items},
            model_used="failed",
            processing_time=0,
            confidence_scores={item.name: 0.0 for item in self.extraction_items},
            errors=["すべてのLLMプロバイダーで抽出に失敗しました"],
            metadata={}
        )
    
    def _quality_check(self, result: ExtractionResult) -> bool:
        """抽出結果の品質をチェック"""
        if not result.success:
            return False
        
        # 必須項目のチェック
        required_items = [item for item in self.extraction_items if item.required]
        for item in required_items:
            value = result.data.get(item.name, "")
            if not value or value == "N/A":
                logger.warning(f"Required item missing: {item.name}")
                return False
        
        # 最小品質チェック
        non_na_count = sum(1 for value in result.data.values() if value and value != "N/A")
        if non_na_count < len(self.extraction_items) * 0.7:  # 70%以上の項目が抽出されている
            logger.warning(f"Too many N/A values: {non_na_count}/{len(self.extraction_items)}")
            return False
        
        return True
    
    def get_available_models(self) -> List[str]:
        """利用可能なモデルのリストを取得"""
        return list(self.providers.keys())


def create_llm_extractor(config: Dict) -> LLMExtractor:
    """
    設定からLLMExtractorを作成
    
    Args:
        config: 設定辞書
        
    Returns:
        LLMExtractor インスタンス
    """
    return LLMExtractor(config.get('llm', {}))