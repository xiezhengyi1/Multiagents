# MultiAgents

## 项目说明

这是一个面向 5G/PCF 策略控制场景的多智能体项目。当前仓库里已经接入了一套面向 `SmPolicy` 和 `URSP` 的标准知识库流水线，核心脚本是：

- [build_pcf_policy_kb.py](/Users/xiezhengyi/Desktop/文件夹/6G+AI/code/MuiltiAgents/knowledge_scripts/build_pcf_policy_kb.py)

这个脚本的职责不是做一个通用 5G RAG，而是构建一个 Release 18 冻结、面向 PCF 标准对象精读的标准知识库。

## `build_pcf_policy_kb.py` 的总体逻辑

脚本分四段：

1. `fetch`
   从 ETSI/3GPP 官方源下载 PDF 和 OpenAPI YAML。

2. `build`
   解析 PDF / YAML，切块，生成结构化 JSONL 和精确索引文件。

3. `ingest`
   把处理后的记录写入 PostgreSQL + PGVector。

4. `all`
   串行执行 `fetch -> build -> ingest`。

命令行入口：

```powershell
.\.venv\Scripts\python.exe knowledge_scripts\build_pcf_policy_kb.py fetch
.\.venv\Scripts\python.exe knowledge_scripts\build_pcf_policy_kb.py build
.\.venv\Scripts\python.exe knowledge_scripts\build_pcf_policy_kb.py ingest
.\.venv\Scripts\python.exe knowledge_scripts\build_pcf_policy_kb.py all
```

## Docling 版 PDF 知识库脚本

仓库中还提供一个独立的 docling 版 PDF 构建脚本：

```powershell
.\.venv\Scripts\python.exe knowledge_scripts\build_pcf_policy_kb_docling.py build
.\.venv\Scripts\python.exe knowledge_scripts\build_pcf_policy_kb_docling.py ingest
.\.venv\Scripts\python.exe knowledge_scripts\build_pcf_policy_kb_docling.py all
```

这个脚本只处理 `knowledge_scripts/data/pcf_policy_r18/raw/*.pdf` 中的 PCF PDF 语料，不处理 OpenAPI YAML。
它会把产物写到独立目录 `knowledge_scripts/data/pcf_policy_r18_docling/processed/`，不会覆盖原有 `build_pcf_policy_kb.py` 的输出。
入库时也会使用独立的 PGVector collection 名称，避免和旧知识库混用。

## 数据源

`default_sources()` 固定了首批标准源，分两类：

- ETSI PDF
  - `23.503`
  - `29.512`
  - `29.514`
  - `24.526`
  - `29.525`
  - `29.519`
  - `24.501`
  - `23.501`
- 3GPP OpenAPI YAML
  - `TS29512_Npcf_SMPolicyControl.yaml`
  - `TS29525_Npcf_UEPolicyControl.yaml`
  - `TS29571_CommonData.yaml`

每个源都带这些元数据：

- `source_id`
- `spec_id`
- `title`
- `release`
- `version`
- `source_url`
- `doc_type`
- `policy_domain`
- `local_name`

抓取后会生成：

- `knowledge_scripts/data/pcf_policy_r18/raw/*`
- `knowledge_scripts/data/pcf_policy_r18/processed/source_manifest.json`

其中 `source_manifest.json` 记录：

- 本地路径
- SHA-256
- 抓取时间

## 使用什么 PDF 解析器

PDF 解析器是：

- `pypdf.PdfReader`

脚本里实际使用位置在 `extract_pdf_sections()`：

```python
reader = PdfReader(source["local_path"])
page_text = sanitize_text(page.extract_text() or "")
```

这说明当前逻辑依赖 PDF 自带 text layer，不做 OCR，也不做版面重建。

结论：

- 优点：轻量、快、依赖少
- 缺点：对目录页、封面、双栏、复杂表格、页眉页脚噪声不够稳

这也是为什么现在会出现少量错误切块，比如把封面文字误识别为 clause。

## 使用什么分块方法

### 1. PDF 的主切块策略

PDF 的主切块函数是：

- `extract_pdf_sections()`
- `split_tables()`
- `split_large_section()`

核心思路不是按固定 token 直接粗切，而是：

1. 先按“疑似章节标题”切出 section
2. 再把 section 内的表格拆开
3. 再把超长 section 做二次切块

### 2. 章节识别方法

章节标题靠这个正则：

```python
HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,5})\s+(?P<title>.+)$")
```

它会把像下面这种行识别成标题：

- `6.6.2.1 Structure Description`
- `4.2.3.2 SM Policy Association Update request`

约束还有一条：

```python
if match and estimate_tokens(line) < 40:
```

也就是说：

- 必须是数字编号开头
- 整行不能太长

这是一个启发式章节切分，不是基于 PDF 目录结构的精确解析。

### 3. 表格拆分方法

表格识别正则：

```python
TABLE_RE = re.compile(r"^(Table\s+\d+(?:\.\d+)?(?:[-:]\d+)?[^\n]*)$", re.MULTILINE)
```

`split_tables()` 会把一个 section 拆成：

- `body`
- `table`

也就是：

- 表格前正文单独保留
- 表格本体单独保留

表格块会被标记为：

- `doc_type = "table"`

普通正文保留原始 `stage2` / `stage3`。

### 4. 超长 section 的二次切块

二次切块函数是：

- `split_large_section()`

阈值：

```python
MIN_CHUNK_TOKENS = 350
MAX_CHUNK_TOKENS = 900
```

逻辑是：

1. 先用 `estimate_tokens()` 估 token 数
2. 如果 section 不超过 `MAX_CHUNK_TOKENS`，直接保留
3. 如果太长，就按空行段落拆分
4. 再把过小的 chunk 回并到前一个 chunk

所以当前分块是：

- 一级：章节/小节
- 二级：表格与正文分离
- 三级：长段落按空行继续拆

不是：

- Sentence splitter
- RecursiveCharacterTextSplitter
- tokenizer-aware semantic chunking

### 5. OpenAPI 的切块方法

OpenAPI 不是整文件入库，而是按对象级拆分：

- `build_openapi_operation_chunks()`
  - 每个 `path + method` 生成一个 operation chunk
- `build_openapi_schema_chunks()`
  - 每个 `components.schemas.<name>` 生成一个 schema chunk

所以 YAML 的最小块单位是：

- 一条 operation
- 一个 schema object

这比把整份 YAML 扔进向量库更合理，适合查：

- `SmPolicyDecision`
- `QosData`
- `CreateSMPolicy`
- `PolicyAssociationUpdateRequest`

## 使用什么“分词器”

这里要明确：这个脚本没有接入 NLP tokenizer，也没有接 BPE/WordPiece/SentencePiece。

它实际用的是一个非常简单的正则词元切分：

```python
TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")
```

真正的“分词”函数是：

```python
def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]
```

这意味着当前 tokenization 是：

- 英文/数字/下划线/点/连字符 级别切分
- 全部转小写

不是：

- tiktoken
- jieba
- spaCy
- Hugging Face tokenizer

### `estimate_tokens()` 也不是正式 tokenizer

```python
def estimate_tokens(text: str) -> int:
    return max(len(normalized.split()), math.ceil(len(normalized) / 4))
```

这只是一个长度估算器，用来控制 chunk 大小。

它的作用是：

- 粗略估算 chunk 是否超长

它不是 embedding 模型或 LLM 的真实 token 计数。

## 文本清洗怎么做

文本清洗函数是：

- `sanitize_text()`

它做的是轻量清洗，不改写内容，主要删除：

- 页眉页脚
- 单独页码
- 一些 ETSI/3GPP 抬头行
- 连续空白

典型逻辑：

- 删除 `3GPP TS xx.xxx`
- 删除 `ETSI TS ...`
- 删除纯数字页码
- 删除 `Intellectual Property Rights`

这意味着它是“去噪”，不是“重写”。

## 元数据如何建立

所有 chunk 都会通过 `make_metadata()` 写入统一元数据，字段包括：

- `source_id`
- `spec_id`
- `release`
- `version`
- `source_url`
- `doc_type`
- `policy_domain`
- `clause_path`
- `clause_title`
- `page_start`
- `page_end`
- `table_id`
- `schema_name`
- `operation_id`
- `object_tags`
- `canonical_title`
- `normalized_terms`
- `related_specs`
- `citation_anchor`

这套元数据是后续检索排序的核心基础。

## `object_tags` 怎么来的

对象标签由 `infer_object_tags()` 生成。

它不是模型抽取，而是规则匹配。会在文本里找这些标准对象：

- `SmPolicyContextData`
- `SmPolicyDecision`
- `PccRule`
- `QosData`
- `SessionRule`
- `TrafficControlData`
- `ChargingData`
- `PolicyControlRequestTrigger`
- `RevalidationTime`
- `UsageMonitoringData`
- `URSP rule`
- `Traffic descriptor`
- `Route selection descriptor`
- `Route Selection Validation Criteria`
- `OS Id`
- `DNN`
- `S-NSSAI`

这一步的作用是：

- 后续 exact index 加权
- cross-spec 关联
- 查询扩展

## Glossary 是怎么建的

Glossary 由 `build_glossary_records()` 生成。

它的输入有两部分：

1. 手工定义的 `CANONICAL_TERM_ALIASES`
2. clause/schema chunk 中自动识别出的 `object_tags`

生成后，每条 glossary 记录包含：

- canonical term
- aliases
- related specs
- related objects
- citations

例如：

- `SmPolicyDecision`
- `UE Route Selection Policy`
- `Traffic descriptor`
- `Route selection descriptor`

这层的目标不是替代原文，而是做：

- 术语归一
- 别名扩展
- 跨规范跳转

## 索引是如何建立的

这里有两种索引，不要混淆。

### 1. 向量索引

向量索引最后写入 PGVector，集合名在 [langchain_pg.py](/Users/xiezhengyi/Desktop/文件夹/6G+AI/code/MuiltiAgents/database/langchain_pg.py) 中定义：

- `pcf_sm_policy_clauses_r18`
- `pcf_sm_policy_schema_r18`
- `pcf_ursp_clauses_r18`
- `pcf_ursp_schema_r18`
- `pcf_policy_glossary_r18`

对应入库函数：

- `ingest_processed_corpus()`

写入逻辑：

1. 读取 `clauses.jsonl`
2. 读取 `schema.jsonl`
3. 读取 `glossary.jsonl`
4. 按 `policy_domain` 分流
5. 每个 collection 调 `rebuild_pgvector_collection()`
6. 用 `store.add_documents(..., ids=...)` 写入

这部分是 dense retrieval 的底座。

### 2. 精确索引

精确索引是本地 JSON 文件，不是数据库全文索引。

函数：

- `build_exact_index()`

输出：

- `schema_exact_index.json`
- `glossary_exact_index.json`

结构包含：

- `doc_count`
- `document_frequency`
- `documents`

其中每篇文档会保存：

- `token_counts`
- `title_tokens`
- `object_tokens`
- `citation_anchor`
- `policy_domain`
- `spec_id`

### 3. 精确索引如何打分

检索函数：

- `search_exact_index()`

打分不是标准 BM25 实现，而是一个自定义 TF-IDF 近似：

```python
idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
score += tf * idf * field_boost
```

并且字段有额外加权：

- 正文 token：`+1`
- title token：`+5`
- object_tags token：`+4`
- normalized_terms token：`+3`
- aliases token：`+2`

查询时再做二次 boost：

- 如果词在 `title_tokens` 中：`field_boost += 1.5`
- 如果词在 `object_tokens` 中：`field_boost += 1.0`

结论：

- 这是“偏 schema/object/title 的 exact index”
- 不是 PostgreSQL FTS
- 也不是 Lucene/BM25 正式实现

## `normalized_terms` 是怎么来的

查询扩展由：

- `normalize_query_terms()`

完成。

它的逻辑是：

1. 先正则切词
2. 再扫描 `CANONICAL_TERM_ALIASES`
3. 如果命中 canonical term 或 alias
4. 就把 canonical name 和所有 alias 的 token 一起加入
5. 最后去重

例如用户查：

- `URSP`

会扩成：

- `UE Route Selection Policy`
- `route selection policy`
- `UE policy route selection`

这是当前 exact retrieval 能命中别名和跨规范对象的关键。

## 处理后的产物有哪些

`build` 阶段会生成：

- `clauses.jsonl`
- `schema.jsonl`
- `glossary.jsonl`
- `spec_object_map.json`
- `term_alias_map.json`
- `schema_exact_index.json`
- `glossary_exact_index.json`
- `retrieval_eval_queries.json`

它们的用途分别是：

- `clauses.jsonl`
  - PDF 正文 / 表格分块
- `schema.jsonl`
  - OpenAPI operation / schema 分块
- `glossary.jsonl`
  - 术语层
- `spec_object_map.json`
  - 标准对象到具体 citation 的映射
- `term_alias_map.json`
  - alias 到 canonical term 的映射
- `schema_exact_index.json`
  - schema 精确索引
- `glossary_exact_index.json`
  - glossary 精确索引
- `retrieval_eval_queries.json`
  - 预设检索测试问题

## 当前脚本的优点

- 结构清晰，`fetch/build/ingest` 分层明确
- PDF 与 OpenAPI 分别处理，没有混装
- schema 粒度正确，适合查标准对象定义
- glossary / alias / cross-spec 这层已经具备实用价值
- collection 按域拆开，利于后续路由

## 当前脚本的局限

这部分要说清楚，不然会误判质量。

### 1. PDF 章节识别是启发式，不够稳

`HEADING_RE` 只靠行文本判断标题，所以：

- 封面页
- 目录页
- 排版异常页

可能被误切成 clause。

### 2. 表格解析只是“文本分段”，不是真正结构化表解析

当前 `split_tables()` 只是把表文本从 section 里切出来，并没有识别：

- 列名
- 行边界
- 单元格结构

所以它更像“表块检索”，不是“表格数据库”。

### 3. token 估算很粗糙

`estimate_tokens()` 只是长度估算，不是 embedding/LLM 的真实 token 数。

### 4. exact index 是自定义近似，不是工业级搜索引擎

优点是简单可控，缺点是：

- 无倒排压缩
- 无 phrase query
- 无 positional index
- 无复杂 BM25 参数

### 5. 没有 OCR / layout-aware parser

对复杂 PDF 的鲁棒性有限。

## 如果你问“这个脚本本质上是什么”

可以把它理解成：

- 一个 Release-18 固定源的标准采集器
- 一个基于 `pypdf + 正则 + YAML` 的轻量解析器
- 一个以 `section/schema/object` 为核心粒度的 chunk builder
- 一个自定义 exact index builder
- 一个 PGVector 多 collection 入库器

不是：

- 通用文档解析平台
- 复杂 PDF OCR 系统
- 工业级全文检索引擎

## 推荐阅读顺序

如果你要继续改这个脚本，建议按下面顺序读：

1. [build_pcf_policy_kb.py](/Users/xiezhengyi/Desktop/文件夹/6G+AI/code/MuiltiAgents/knowledge_scripts/build_pcf_policy_kb.py)
   先看 `default_sources`、`extract_pdf_sections`、`extract_openapi_chunks`、`build_exact_index`

2. [langchain_pg.py](/Users/xiezhengyi/Desktop/文件夹/6G+AI/code/MuiltiAgents/database/langchain_pg.py)
   看 PGVector collection 的定义和入库方式

3. [knowledge_tool.py](/Users/xiezhengyi/Desktop/文件夹/6G+AI/code/MuiltiAgents/tools/knowledge_tool.py)
   看 exact/glossary/cross-spec/vector 四层是怎么消费这些产物的

## 一句话总结

`build_pcf_policy_kb.py` 当前采用的是：

- `pypdf` 解析 PDF
- `yaml.safe_load` 解析 OpenAPI
- 基于章节标题、表头正则和空行段落的启发式分块
- 基于简单正则 token 的自定义分词
- 基于加权 TF-IDF 近似的本地 exact index
- 基于 PGVector 多 collection 的向量索引入库

如果你的目标是“快速得到一个能查 `SmPolicyDecision / PccRule / QosData / URSP descriptor` 的标准库”，这套逻辑是够用的。  
如果你的目标是“高精度还原 PDF 正文与表格结构”，那下一步就该升级 PDF 解析和表格结构化了。
