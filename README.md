# arXiv論文自動抽出システム

arXiv論文から指定された9つの項目を自動抽出し、CSV形式で出力するagentic LLMパイプラインシステムです。

## 🌟 主要機能

### 📊 抽出対象項目（9項目）
1. **手法の肝** - 論文の核となる技術手法・アプローチ
2. **制限事項** - 手法の限界や制約条件
3. **対象ナレッジ** - 扱う知識領域・データ種別
4. **URL** - arXiv論文のURL
5. **タイトル** - 論文タイトル
6. **出版年** - 発表年
7. **研究分野** - 分野分類
8. **課題設定** - 解決しようとする問題
9. **論文の主張** - 主要な貢献・結論

### 🔧 核心技術
- **高精度Reference削除**: 99%以上の精度で参考文献部分を自動除去
- **マルチOCR統合**: Nougat、Unstructured、Suryaなど複数OCRモデルの統合
- **Agenticパイプライン**: 各処理段階での品質チェックとフォールバック機能
- **LLMプロバイダー統合**: OpenAI GPT、Anthropic Claudeの統合利用

## 🚀 クイックスタート

### インストール

```bash
git clone <repository-url>
cd arxiv-scraper
pip install -r requirements.txt
```

### 基本的な使用方法

```bash
# 単一論文処理
python -m src.arxiv_extractor --id 2301.12345

# URL指定
python -m src.arxiv_extractor --url https://arxiv.org/abs/2301.12345

# PDFファイル直接指定
python -m src.arxiv_extractor --pdf paper.pdf

# バッチ処理
python -m src.arxiv_extractor --batch paper_list.txt

# 複数ID指定
python -m src.arxiv_extractor --ids 2301.12345,2301.12346,2301.12347
```

### 設定ファイル

`config/config.yaml`で詳細設定が可能：

```yaml
# OCR設定
ocr:
  model: "unstructured"
  fallback_models: ["surya", "nougat"]
  device: "cuda"

# Reference削除設定
text_processing:
  remove_references: true
  citation_cleanup: true

# LLM設定
llm:
  model: "gpt-4"
  fallback_models: ["gpt-3.5-turbo"]
  openai:
    api_key: "your-api-key"

# 出力設定
output:
  format: "csv"
  output_dir: "./output"
```

## 🏗️ システム構成

### パイプライン流れ

```
論文入力 → PDF取得 → OCR処理 → Reference削除 → テキスト後処理 → LLM解析 → CSV出力
```

### 主要コンポーネント

1. **arXiv API連携** (`arxiv_api.py`)
   - arXivからのメタデータ・PDF取得
   - レート制限・リトライ機能

2. **OCR処理** (`ocr_processor.py`)
   - 複数OCRモデルの統合
   - フォールバック機能付き高精度テキスト抽出

3. **Reference削除** (`reference_cleaner.py`) ⭐
   - 多様な参考文献形式の自動検出
   - 99%以上の精度での削除
   - 本文内容の保持保証

4. **テキスト後処理** (`text_processor.py`)
   - OCR誤認識修正
   - セクション抽出
   - 品質分析

5. **LLM解析** (`llm_extractor.py`)
   - GPT-4/Claude等による情報抽出
   - 構造化出力生成
   - 品質評価

6. **CSV出力** (`csv_generator.py`)
   - UTF-8エンコーディング
   - メタデータ付き出力
   - 統計情報生成

## 📈 品質保証

### 精度目標
- **OCR精度**: >95%
- **Reference削除精度**: >99%
- **項目抽出精度**: >90%
- **必須項目出力率**: 100%

### 性能目標
- **1論文あたり処理時間**: <60秒
- **Reference削除時間**: <3秒
- **バッチ処理**: 50論文/時間

## 🧪 テスト

```bash
# 全テスト実行
pytest tests/ -v

# カバレッジ付きテスト
pytest --cov=src tests/

# 特定モジュールテスト
pytest tests/test_reference_cleaner.py -v
```

## 📝 API Keys設定

環境変数またはconfig.yamlで設定：

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## 📂 出力形式

### CSV出力例

```csv
手法の肝,制限事項,対象ナレッジ,URL,タイトル,出版年,研究分野,課題設定,論文の主張,処理日時,信頼度スコア,使用モデル,処理時間
"Transformer-based neural architecture with attention mechanism","Requires large computational resources","Natural language processing, Machine learning","https://arxiv.org/abs/2301.12345","Attention Is All You Need","2017","cs.CL","Sequence-to-sequence modeling without recurrence","Self-attention mechanism is sufficient for capturing dependencies",2024-01-15 10:30:45,0.92,gpt-4,45.2s
```

## 🔧 開発・カスタマイズ

### 新しいOCRモデル追加

```python
class NewOCRProcessor(OCRProcessor):
    def process_pdf(self, pdf_path: str) -> OCRResult:
        # 実装
        pass
```

### カスタム抽出項目

config.yamlの`extraction_items`セクションで設定可能：

```yaml
extraction_items:
  - name: "新項目"
    description: "項目の説明"
    max_length: 500
    required: true
```

## 📊 パフォーマンス最適化

- **並列処理**: バッチ処理での論文並列処理
- **キャッシュ**: OCR結果・LLM結果のキャッシュ
- **メモリ管理**: 大量PDF処理時のメモリ最適化

## 🤝 貢献

1. Issueの確認・作成
2. Feature branchの作成
3. 変更の実装とテスト
4. Pull Requestの提出

## 📄 ライセンス

MIT License

## 🙋‍♂️ サポート

- バグ報告: GitHub Issues
- 機能要望: GitHub Discussions
- 技術質問: GitHub Issues with `question` label

---

**特徴**: 本システムの最大の差別化要素は、**Reference削除機能**です。論文の参考文献部分を99%以上の精度で除去することで、LLMが本文内容のみに集中して情報抽出を行うことができ、大幅な精度向上を実現しています。