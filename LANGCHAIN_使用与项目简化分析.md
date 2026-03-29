# LangChain 使用文档与本项目简化分析

更新时间：2026-03-28  
适用范围：以本仓库 `pyproject.toml` 中声明的 `langchain>=1.2.3`、`langgraph>=1.0.5`、`langchain-openai>=1.1.7` 为主  
文档目标：

1. 用中文梳理 LangChain 当前主线用法。
2. 按“函数/方法 -> 作用 -> 示例”的方式给出可直接参考的代码。
3. 结合本项目现状，分析数据库、memory、工具调用、结构化输出等地方是否值得用 LangChain 简化。

---

## 1. 先说结论

### 1.1 LangChain 现在的主线是什么

LangChain v1 的主线已经很明确：

- 快速搭 agent：优先用 `langchain.agents.create_agent`
- 直接调模型：优先用 `langchain.chat_models.init_chat_model`
- 组合 prompt / model / parser：优先用 Runnable / LCEL（`|` 组合）
- 短期记忆 / 长期记忆 / 持久化：优先用 LangGraph 的 `checkpointer` 和 `store`
- 向量检索：优先用标准 `Embeddings + VectorStore + Retriever` 抽象
- 老式 `Chain` / `Memory` 类很多已经退到 `langchain-classic`，新代码不建议继续堆旧抽象

### 1.2 本项目能否用 LangChain 简化

可以，但要分层看：

- 可以明显简化：
  - 模型调用
  - 工具调用循环
  - 结构化输出解析
  - 记忆管理
  - 语义检索
- 不应该让 LangChain 接管：
  - 业务数据库表设计
  - 策略编译 / 策略归一化
  - SLA 评估逻辑
  - PCF 下发逻辑
  - 你们领域里的确定性规则

一句话概括：  
`LangChain 适合接管“LLM 周边基础设施层”，不适合接管“电信策略业务规则层”。`

---

## 2. 本项目现状速览

我先根据仓库代码确认了几个关键点。

### 2.1 已经在用 LangChain 的地方

- `agents/basemodel.py`
  - 用了 `langchain_openai.ChatOpenAI`
- `agents/intent_encoding/agent.py`
  - 用了 `ChatPromptTemplate`
  - 用了 `PydanticOutputParser`
  - 用了 `llm.bind_tools(...)`
- `agents/optimization_strategy/agent.py`
  - 用了 `@tool`
  - 用了 `ChatPromptTemplate`
  - 用了 `PydanticOutputParser`
  - 用了 `llm.bind_tools(...)`
- `tools/knowledge_tool.py`
  - 用了 `langchain_core.tools.tool`

所以这个项目不是“要不要引入 LangChain”，而是“要不要把现在手写的半套 agent 基础设施收拢到 LangChain/LangGraph 的正式抽象上”。

### 2.2 当前手写较多的地方

- `agents/MemoryManager.py`
  - 短期记忆：`deque`
  - 长期记忆：本地 JSON 文件
  - 检索：numpy 手工余弦相似度
- `tools/knowledge_tool.py`
  - embedding：直接调用 `openai.OpenAI().embeddings.create(...)`
  - 检索：直接写 SQLAlchemy + pgvector 排序
- `agents/intent_encoding/agent.py`
  - 手写 tool loop
  - 手动解析模型输出代码块
- `agents/optimization_strategy/agent.py`
  - 同样手写 tool loop
  - 同样手动清理 JSON 输出
- `system_coordinator.py`
  - memory 注入、上下文拼接、轮次协作都由 coordinator 自己管理

### 2.3 我确认到的几个问题

这些不是风格问题，是实打实的逻辑/工程一致性问题。

1. `agents/basemodel.py` 注释和实现不一致  
   注释写的是“优先 `OPENAI_API_KEY`，否则尝试 `DASHSCOPE_API_KEY`”，但实际代码只读取了 `OPENAI_API_KEY`。  
   注释还写了 `OPENAI_BASE_URL` 默认 fallback 到 DashScope，但实际也没有默认值。

2. 当前 `.venv` 似乎不可直接执行  
   我尝试用 `.venv\Scripts\python.exe` 读取已安装包版本，进程创建失败，路径指向 `uv` 管理的 Python。  
   这说明“依赖已声明”不等于“当前虚拟环境一定可直接运行”。

3. 你现在的 PGVector 用法和 LangChain 官方主线不是同一路  
   当前仓库依赖是：
   - `pgvector`
   - `sqlalchemy`
   - `psycopg2-binary`

   但 LangChain 官方现在推荐的 PGVector 集成是 `langchain-postgres`，且要求 `psycopg3`。  
   所以如果要切过去，这不是 import 改一行就行，而是一次明确迁移。

---

## 3. LangChain 包生态怎么理解

### 3.1 主要包

| 包 | 作用 | 什么时候用 |
| --- | --- | --- |
| `langchain` | v1 主入口，agent、模型初始化、核心开发体验 | 新项目优先 |
| `langchain-core` | 底层抽象：messages、prompts、runnables、tools | 基础接口 |
| `langchain-openai` | OpenAI / 兼容 OpenAI API 的模型与 embedding 集成 | 你这个项目会常用 |
| `langchain-text-splitters` | 文本切分 | 做知识库/RAG 时用 |
| `langgraph` | 有状态、多节点、可持久化 agent/runtime | 做 memory、流程编排时用 |
| `langgraph-checkpoint-*` | checkpointer 持久化后端 | 需要短期记忆持久化时用 |
| `langsmith` | tracing / debugging / eval | 调试线上 agent 时用 |
| `langchain-classic` | 老 API 兼容层 | 新代码尽量少用 |

### 3.2 LangChain / LangGraph / Deep Agents 的关系

- `LangChain`
  - 偏高层，适合快速搭建 agent、tool-calling、structured output
- `LangGraph`
  - 偏底层，适合自己控制状态机、节点、持久化、人工介入
- `Deep Agents`
  - 更高一层的“带电池 agent harness”

对这个仓库而言：

- 你们的 IEA -> OSA -> PDA 闭环本质上更像“可控工作流”
- 因此：
  - 单个 agent 内部：LangChain 很适合
  - 整体多 agent 闭环：LangGraph 比纯 `create_agent` 更贴切

---

## 4. 模型调用：`init_chat_model` / `ChatOpenAI`

### 4.1 `init_chat_model`

作用：  
统一初始化聊天模型，减少 provider 特定代码。

常用方法：

- `invoke(input)`
- `stream(input)`
- `batch(inputs)`
- `with_structured_output(schema)`
- `bind_tools(tools)`

示例：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai:gpt-4.1",
    temperature=0,
    max_tokens=1000,
)

resp = model.invoke("Explain URLLC in one paragraph.")
print(resp.content)
```

### 4.2 `ChatOpenAI`

作用：  
当你需要显式指定 OpenAI 兼容端点、API Key、provider 参数时，用它更直接。

示例：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    api_key="your-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = llm.invoke("Summarize the UE policy context.")
print(resp.content)
```

### 4.3 `invoke`

作用：  
单次请求，拿完整结果。

示例：

```python
response = model.invoke("What is slice isolation?")
print(response.content)
```

### 4.4 `stream`

作用：  
边生成边消费输出。

示例：

```python
for chunk in model.stream("Explain network slicing briefly."):
    print(chunk.text, end="", flush=True)
```

### 4.5 `batch`

作用：  
批量执行多个输入；适合 IO 型任务并行。

示例：

```python
inputs = [
    "What is eMBB?",
    "What is URLLC?",
    "What is mMTC?",
]
results = model.batch(inputs)
for item in results:
    print(item.content)
```

### 4.6 `with_structured_output`

作用：  
要求模型按 schema 返回结构化对象，避免手工 `json.loads` + 代码块清洗。

示例：

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class FlowDecision(BaseModel):
    supi: str = Field(description="UE identifier")
    flow_id: str = Field(description="Target flow id")
    action: str = Field(description="modify, add, or delete")

model = init_chat_model("openai:gpt-4.1", temperature=0)
structured_model = model.with_structured_output(FlowDecision)

result = structured_model.invoke(
    "Extract: supi=imsi-20893002, flow_id=flow-1001, action=modify"
)
print(result)
```

---

## 5. 消息与 Prompt：`Messages` / `ChatPromptTemplate`

### 5.1 `SystemMessage` / `HumanMessage` / `AIMessage`

作用：  
LangChain 中模型上下文的标准载体。

示例：

```python
from langchain.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("You are a telecom policy assistant."),
    HumanMessage("Reduce bandwidth of video flow for supi imsi-20893002."),
]

response = model.invoke(messages)
print(response.content)
```

### 5.2 `ChatPromptTemplate.from_messages(...)`

作用：  
把 prompt 定义为结构化消息模板，而不是手工拼字符串。

常用方法：

- `from_messages(...)`
- `invoke(vars)`
- `format_messages(**kwargs)`
- `partial(...)`

示例：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a network policy expert."),
        ("human", "User input: {user_input}"),
        ("human", "Context: {context}"),
    ]
)

messages = prompt.format_messages(
    user_input="Lower video bandwidth for imsi-20893002",
    context="Latest snapshot shows congestion on slice 2",
)

response = model.invoke(messages)
print(response.content)
```

### 5.3 `MessagesPlaceholder`

作用：  
在 prompt 中插入一段已有对话历史。

示例：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

messages = prompt.invoke(
    {
        "history": [
            ("human", "Hi"),
            ("ai", "Hello"),
        ],
        "question": "What did I just say?",
    }
)
```

### 5.4 `partial(...)`

作用：  
预填一部分模板变量，减少重复传参。

示例：

```python
base_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are {role}."),
        ("human", "{question}"),
    ]
)

prompt = base_prompt.partial(role="a telecom optimization assistant")
response = (prompt | model).invoke({"question": "What is GBR?"})
```

---

## 6. Runnable / LCEL：LangChain 组合表达式

LCEL = LangChain Expression Language。  
核心思想：`prompt | model | parser`

### 6.1 `Runnable`

作用：  
统一“可调用组件”的接口。

常用方法：

- `invoke`
- `ainvoke`
- `batch`
- `abatch`
- `stream`
- `with_retry`
- `assign`
- `bind`
- `get_graph`

### 6.2 `RunnableSequence`

作用：  
顺序执行多个 Runnable；通常由 `|` 自动构造。

示例：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

prompt = ChatPromptTemplate.from_messages(
    [("human", "Translate to English: {text}")]
)
model = init_chat_model("openai:gpt-4.1")

chain = prompt | model
result = chain.invoke({"text": "降低视频流带宽"})
print(result.content)
```

### 6.3 `RunnableParallel`

作用：  
把同一输入并发送给多个分支。

示例：

```python
from langchain_core.runnables import RunnableLambda

chain = RunnableLambda(lambda x: x + 1) | {
    "double": RunnableLambda(lambda x: x * 2),
    "triple": RunnableLambda(lambda x: x * 3),
}

print(chain.invoke(1))
# {"double": 4, "triple": 6}
```

### 6.4 `with_retry(...)`

作用：  
给 Runnable 加重试策略。

注意：  
这只是基础设施层的调用重试，不是业务 fallback。  
你在 AGENTS.md 里要求“不要靠兜底掩盖真实问题”，这个要求是对的。  
所以如果用了 `with_retry`，应只对网络抖动/临时 API 错误生效，不应吞掉结构化输出错误再偷偷走别的逻辑。

示例：

```python
safe_chain = (prompt | model).with_retry()
result = safe_chain.invoke({"text": "hello"})
```

---

## 7. 结构化输出：推荐优先级

推荐优先级：

1. `with_structured_output(schema)`  
2. `create_agent(..., response_format=schema)`  
3. `PydanticOutputParser`

原因：

- 1 和 2 更贴近 LangChain v1 主线
- `PydanticOutputParser` 仍可用，但通常意味着你要手写 prompt 约束和解析兜底

### 7.1 `PydanticOutputParser`

作用：  
把 LLM 文字输出解析成 Pydantic 对象。

常用方法：

- `get_format_instructions()`
- `parse(text)`
- `invoke(text)`

示例：

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Intent(BaseModel):
    supi: str = Field(description="User supi")
    operation_type: str = Field(description="modify/add/delete")

parser = PydanticOutputParser(pydantic_object=Intent)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Return valid JSON.\n{format_instructions}"),
        ("human", "{text}"),
    ]
)

chain = prompt | model | parser
result = chain.invoke(
    {
        "text": "Please reduce bandwidth for supi imsi-20893002",
        "format_instructions": parser.get_format_instructions(),
    }
)
print(result)
```

### 7.2 `JsonOutputParser`

作用：  
当你只想拿 JSON，而不想先建 Pydantic 类时使用。

示例：

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = prompt | model | parser
result = chain.invoke({"text": "Return a JSON with fields a and b"})
print(result)
```

---

## 8. 工具：`@tool` / `bind_tools` / `ToolRuntime`

### 8.1 `@tool`

作用：  
把普通函数包装成可被模型调用的工具。

示例：

```python
from langchain.tools import tool

@tool
def get_ue_context(supi: str) -> str:
    """Get UE context by SUPI."""
    return f"context for {supi}"
```

### 8.2 `bind_tools(tools)`

作用：  
把工具绑定给模型；模型此后可以返回 `tool_calls`。

示例：

```python
model_with_tools = model.bind_tools([get_ue_context])
response = model_with_tools.invoke("Fetch UE context for imsi-20893002")

print(response.tool_calls)
```

### 8.3 `tool_choice`

作用：  
强制模型必须调用某个工具，或必须调用任意一个工具。

示例：

```python
model_with_tools = model.bind_tools([get_ue_context], tool_choice="any")
```

### 8.4 `parallel_tool_calls=False`

作用：  
禁用并行工具调用。  
对需要串行、强一致、严格顺序副作用的工具很有用。

示例：

```python
model_with_tools = model.bind_tools(
    [get_ue_context],
    parallel_tool_calls=False,
)
```

### 8.5 `ToolRuntime`

作用：  
在工具内部访问：

- 当前 state
- context
- store
- stream writer
- config
- tool_call_id

示例：

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str

@tool
def read_preferences(runtime: ToolRuntime[Context]) -> str:
    """Read user preferences from runtime store."""
    if runtime.store is None:
        return "No store configured."
    item = runtime.store.get((runtime.context.user_id, "prefs"), "email_style")
    return str(item.value) if item else "No preference found."
```

---

## 9. Agent：`create_agent`

### 9.1 `create_agent(...)`

作用：  
创建生产可用的 LangChain agent。  
它内部运行在 LangGraph runtime 上。

常见参数：

- `model`
- `tools`
- `system_prompt`
- `response_format`
- `middleware`
- `context_schema`
- `checkpointer`
- `store`

基础示例：

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's sunny in {city}."

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Shanghai?"}]}
)
print(result)
```

### 9.2 `response_format=...`

作用：  
要求 agent 返回结构化结果。

示例：

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    name: str
    email: str

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[],
    response_format=ContactInfo,
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Extract: John Doe, john@example.com"}
        ]
    }
)

print(result["structured_response"])
```

### 9.3 `context_schema`

作用：  
为每次 agent 调用注入稳定上下文，比如：

- user_id
- session_id
- snapshot_id
- db connection handle

示例：

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class Context:
    session_id: str
    snapshot_id: str

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[],
    context_schema=Context,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Analyze this request"}]},
    context=Context(session_id="s1", snapshot_id="n42"),
)
```

### 9.4 middleware

作用：  
在模型调用前后做动态 prompt、动态工具过滤、日志、模型切换等。

适合这个项目的用法：

- 根据轮次切换 prompt
- 根据协作阶段限制工具集合
- 在 runtime 中注入 `session_id` / `snapshot_id` / `supi`

---

## 10. Embedding：`OpenAIEmbeddings`

### 10.1 `OpenAIEmbeddings`

作用：  
统一 embedding 接口。

常用方法：

- `embed_query(text)`
- `embed_documents(texts)`
- `aembed_query(...)`
- `aembed_documents(...)`

示例：

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
)

query_vec = embeddings.embed_query("low latency slice")
doc_vecs = embeddings.embed_documents(
    [
        "URLLC slice with low delay budget",
        "eMBB slice optimized for throughput",
    ]
)
```

### 10.2 什么时候不该自己直接调 SDK

像你现在 `tools/knowledge_tool.py` 那样直接调用 `openai.OpenAI().embeddings.create(...)` 并不是错，但会失去 LangChain 统一抽象的好处：

- 更难替换 embedding provider
- 更难和 VectorStore/Retriever 直接对接
- 更难复用 async / batch / tracing

---

## 11. 文本切分：`RecursiveCharacterTextSplitter`

### 11.1 `RecursiveCharacterTextSplitter`

作用：  
把长文本切成适合检索和模型上下文的块。  
官方推荐它作为通用默认方案。

常用方法：

- `split_text(text)`
- `create_documents(texts)`
- `split_documents(documents)`
- `from_language(language, ...)`

示例：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = splitter.split_text(long_text)
print(chunks[:2])
```

生成 `Document` 的示例：

```python
docs = splitter.create_documents(
    ["This is a long document."],
    metadatas=[{"source": "manual"}],
)
```

代码场景：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=800,
    chunk_overlap=100,
)
```

---

## 12. 向量库与检索：`VectorStore` / `Retriever` / `PGVector`

### 12.1 `InMemoryVectorStore`

作用：  
快速验证检索逻辑；适合 demo / 单元测试。

示例：

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = InMemoryVectorStore.from_texts(
    [
        "URLLC is for ultra-reliable low-latency communication",
        "eMBB is for enhanced mobile broadband",
    ],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()
docs = retriever.invoke("Which service targets low latency?")
print(docs[0].page_content)
```

### 12.2 `PGVector`

作用：  
把 PostgreSQL + pgvector 包装成标准向量存储。

常用方法：

- `add_documents(docs)`
- `similarity_search(query, k=...)`
- `similarity_search_with_score(query, k=...)`
- `as_retriever(...)`

示例：

```python
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="semantic_knowledge_docs",
    connection="postgresql+psycopg://user:password@localhost:5432/dbname",
    use_jsonb=True,
)

vector_store.add_documents(
    [
        Document(
            page_content="URLLC slice requires low packet delay budget.",
            metadata={"category": "Slice_Profile", "key": "urllc_profile"},
        )
    ]
)

docs = vector_store.similarity_search("low latency slice", k=3)
```

### 12.3 `as_retriever(...)`

作用：  
把 VectorStore 转成统一检索接口。

示例：

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3},
)

docs = retriever.invoke("industrial automation low latency")
```

### 12.4 重要迁移提醒

如果你要按 LangChain 官方当前路线使用 PGVector，要注意：

- 需要新增依赖：`langchain-postgres`
- 需要改驱动到 `psycopg3`
- 连接串通常写成：

```python
postgresql+psycopg://user:password@host:port/db
```

这和当前仓库里基于 `psycopg2-binary` + SQLAlchemy ORM 的用法不是同一套实现。

---

## 13. 记忆：LangGraph `checkpointer` / `store`

这是本项目最值得改的一层。

### 13.1 短期记忆：`checkpointer`

作用：  
保存某个 thread 内的消息和状态，用于多轮对话或多步 agent 继续执行。

#### 13.1.1 `InMemorySaver`

适合：

- demo
- 单元测试
- 临时实验

示例：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1")
checkpointer = InMemorySaver()

def call_model(state: MessagesState):
    return {"messages": model.invoke(state["messages"])}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-1"}}
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Bob"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)
```

#### 13.1.2 `PostgresSaver`

适合：

- 生产环境
- 需要 durable short-term memory
- 需要线程级持久化

示例：

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # 第一次使用需要 setup()
    # checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

### 13.2 长期记忆：`store`

作用：  
跨 session / 跨 thread 保存用户信息、偏好、摘要、业务侧可复用记忆。

#### 13.2.1 `InMemoryStore`

示例：

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(store=store)
```

#### 13.2.2 `PostgresStore`

适合：

- 生产持久化
- 用户维度记忆
- 多 agent 共享 memory namespace

示例：

```python
from langgraph.store.postgres import PostgresStore

with PostgresStore.from_conn_string(DB_URI) as store:
    # 第一次使用需要 setup()
    # store.setup()
    graph = builder.compile(checkpointer=checkpointer, store=store)
```

### 13.3 在节点里访问 memory

示例：

```python
import uuid
from dataclasses import dataclass
from langgraph.runtime import Runtime
from langgraph.graph import MessagesState

@dataclass
class Context:
    user_id: str

def call_model(state: MessagesState, runtime: Runtime[Context]):
    namespace = (runtime.context.user_id, "memories")

    memories = runtime.store.search(
        namespace,
        query=str(state["messages"][-1].content),
    ) if runtime.store else []

    memory_text = "\n".join(item.value["data"] for item in memories)

    if "remember" in str(state["messages"][-1].content).lower() and runtime.store:
        runtime.store.put(namespace, str(uuid.uuid4()), {"data": "User prefers low latency over throughput"})

    prompt = [
        {"role": "system", "content": f"Relevant memory:\n{memory_text}"},
        *state["messages"],
    ]
    return {"messages": model.invoke(prompt)}
```

### 13.4 为什么它比当前 `MemoryManager.py` 更适合

当前 `MemoryManager.py` 的问题：

- 短期记忆只在进程内 `deque`
- 长期记忆存在 JSON 文件
- 检索手写 numpy 余弦相似度
- 没有 thread / namespace / user 维度
- 并发和持久化都比较弱

LangGraph memory 的优势：

- 短期 / 长期记忆职责更清晰
- 支持 thread_id
- 支持生产持久化后端
- 和 agent runtime 原生打通
- 不需要你在 coordinator 里自己拼那么多 memory 上下文字符串

---

## 14. 一个完整的最小示例

下面给一个“prompt + tool + structured output + memory”的简化版示例。

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    session_id: str
    user_id: str


class IntentResult(BaseModel):
    supi: str = Field(description="UE identifier")
    operation_type: str = Field(description="modify/add/delete")
    target: str = Field(description="policy or flow target")


@tool
def get_latest_snapshot(runtime: ToolRuntime[Context]) -> str:
    """Return current snapshot metadata."""
    return f"session={runtime.context.session_id}, snapshot=latest"


checkpointer = InMemorySaver()
store = InMemoryStore()

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_latest_snapshot],
    system_prompt="You are a telecom intent extraction assistant.",
    response_format=IntentResult,
    context_schema=Context,
    checkpointer=checkpointer,
    store=store,
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Please modify the video flow bandwidth for supi imsi-20893002",
            }
        ]
    },
    context=Context(session_id="session-1", user_id="user-1"),
)

print(result["structured_response"])
```

这个例子虽然小，但已经覆盖了你项目里现在最常手写的四件事：

- 上下文注入
- 工具绑定
- 结构化输出
- 会话级状态保存

---

## 15. 针对本项目的简化建议

这一节最重要。

### 15.1 可以简化的点一：`MemoryManager.py`

当前文件：`agents/MemoryManager.py`

建议：

- 删除手工 `deque + JSON + numpy 相似度` 方案
- 用：
  - `checkpointer` 管短期记忆
  - `store` 管长期记忆
- 如果要语义检索长期记忆：
  - 不要继续把向量塞进 JSON 文件
  - 改为 Postgres/VectorStore 方案

推荐替换方向：

1. 短期记忆  
   `system_coordinator.py` 每轮传 `thread_id=session_id`，不再手工拼最近 5 条消息。

2. 长期记忆  
   按 `user_id` 或 `supi` 建 namespace，例如：
   - `(supi, "episodic")`
   - `(supi, "preferences")`
   - `(supi, "planning_history")`

3. 记忆摘要  
   摘要仍然可以保留，但应该是“写入 store 的一类数据”，不是“先写 JSON 再自己算相似度”。

结论：  
`MemoryManager` 是最适合被 LangGraph memory 直接替换的模块。

### 15.2 可以简化的点二：`tools/knowledge_tool.py`

当前文件：`tools/knowledge_tool.py`

现在做法：

- 自己生成 embedding
- 自己排序查询
- 自己组织字符串结果

建议：

- 如果目标是“语义知识检索”，改成：
  - `OpenAIEmbeddings`
  - `PGVector`
  - `retriever.invoke(query)`
- 如果仍然要沿用当前业务表 `semantic_knowledge`，那就继续保持 SQLAlchemy 写法，不必为了 LangChain 强行改

这里要分两种方案：

#### 方案 A：最稳妥

保留当前 `semantic_knowledge` 业务表，只把 embedding 调用统一成 `OpenAIEmbeddings`。  
这属于“小改动，低风险”。

#### 方案 B：LangChain 标准化

为知识库单独建一套 `langchain-postgres` collection，不直接复用当前业务表。  
这属于“更标准，但要迁移数据和驱动”。

我更推荐：

- 业务表继续用 ORM
- 检索型知识库单独建 PGVector collection

原因：  
`业务数据存储` 和 `RAG/语义检索存储` 不应该硬绑在一张表上。

### 15.3 可以简化的点三：IEA / OSA 的工具循环

当前文件：

- `agents/intent_encoding/agent.py`
- `agents/optimization_strategy/agent.py`

现在做法：

- `llm.bind_tools(...)`
- 手写 `for iteration in range(5)`
- 手写处理 `response.tool_calls`
- 手写 `ToolMessage`
- 手写最终输出解析

这套逻辑能工作，但重复度很高。

建议方向：

#### 方向 A：继续保留当前类结构，但抽出公共 AgentRunner

适合你们现在这个仓库。  
做法：

- 抽一个公共 helper
- 统一处理：
  - tool loop
  - tool map
  - iteration limit
  - output parser
  - 日志

这不一定非得用 `create_agent`，但会立刻减少重复代码。

#### 方向 B：单 agent 内部改为 `create_agent`

适合 IEA 这种“工具调用 + 结构化输出”的场景。  
IEA 最容易迁过去。

#### 方向 C：整个多 agent 闭环改成 LangGraph 图

适合长期演进，但改动最大。  
把：

- IEA
- OSA
- PDA
- Feedback loop
- session context
- handoff history

都建成图节点和状态边。

我的判断：

- 短期最值得做：A + IEA 局部尝试 B
- 中期可做：整体向 LangGraph workflow 收敛

### 15.4 可以简化的点四：结构化输出解析

当前文件：

- `agents/intent_encoding/agent.py`
- `agents/optimization_strategy/agent.py`

现在有很多手工代码：

- 清理 ```json 代码块
- `parser.parse(output_str)`
- parse 失败后写错误日志

建议：

- 如果模型支持可靠 structured output：
  - 优先换成 `with_structured_output(...)`
- 如果场景已经用 `create_agent`：
  - 优先换成 `response_format=...`

这样可以直接减少：

- 手工字符串清洗
- prompt 里多余的“必须返回 JSON”约束文本
- 输出解析失败的不稳定性

### 15.5 可以简化的点五：运行时上下文传递

当前文件：`system_coordinator.py`

现在很多上下文是通过：

- prompt 字符串拼接
- 额外函数参数
- 手工 JSON 序列化

建议：

- 用 `context_schema`
- 把以下信息作为 runtime context 注入：
  - `session_id`
  - `snapshot_id`
  - `round_index`
  - `supi`
  - 数据库访问对象或 service handle

这样好处是：

- 工具内部不必再依赖全局变量
- prompt 不必背太多运行时元数据
- 单元测试更容易写

---

## 16. 不建议用 LangChain 简化的地方

这部分同样重要，否则容易把问题复杂化。

### 16.1 不要用 LangChain 替代业务 ORM

当前数据库表：

- `SessionContext`
- `EpisodicExperience`
- `SemanticKnowledge`
- `NetworkStatusSnapshot`
- `UeContextRecord`

这些是业务数据模型，不是 LangChain 的职责。  
继续保留 SQLAlchemy 是对的。

### 16.2 不要把策略编译逻辑交给 agent

像这些模块：

- `domain/policy_compiler.py`
- `domain/policy_plan.py`
- `agents/optimization_strategy/agent.py` 里的归一化逻辑
- `workflows/execution_controller.py`
- `workflows/assurance_evaluator.py`

本质是确定性业务规则。  
这里应该继续写死、写清楚、写可测，不要交给 LLM 或 agent 自由发挥。

### 16.3 不要为了“统一”而把所有检索都抽成 Retriever

如果某些查询本来就是：

- 按主键查
- 按 SUPI 查
- 按 snapshot_id 查

那就继续走 SQLAlchemy。  
Retriever 只适合“语义近似检索”，不适合拿来替代普通业务查询。

---

## 17. 推荐的落地路线

按收益/风险比排序，我建议这样做。

### 第一步：先收敛结构化输出

目标：

- IEA / OSA 少写解析胶水代码

做法：

- 优先把 `PydanticOutputParser + 手工清理 JSON 代码块` 改成
  - `with_structured_output(...)`
  - 或 `create_agent(..., response_format=...)`

收益：

- 改动小
- 立刻减少脆弱代码

### 第二步：替换 `MemoryManager`

目标：

- 去掉 `deque + JSON + numpy`

做法：

- 短期记忆改 `checkpointer`
- 长期记忆改 `store`
- 真正需要语义检索的长期知识再单独接向量库

收益：

- memory 语义清晰
- 更适合多轮/多 agent

### 第三步：统一 embedding / retriever 抽象

目标：

- 把知识检索层标准化

做法：

- embedding 统一成 `OpenAIEmbeddings`
- 如果要标准化向量检索，再引入 `langchain-postgres`

注意：

- 这一步不是小修，要明确处理 `psycopg2 -> psycopg3` 迁移

### 第四步：评估是否把闭环编排收敛到 LangGraph

目标：

- 用图状态表达 IEA -> OSA -> PDA -> Feedback

收益：

- 状态更规范
- handoff / persistence / thread continuation 更清晰

代价：

- 改动最大

---

## 18. 我对本项目的最终判断

### 18.1 最值得用 LangChain 简化的三个点

1. 结构化输出  
   你现在手工清 JSON 的代码太多，最该先收。

2. Memory  
   当前 `MemoryManager.py` 很像“自己重写了一版简化且更弱的 LangGraph memory”。

3. 知识检索层  
   embedding / vector store / retriever 完全可以统一抽象。

### 18.2 不值得强行改的三个点

1. SQLAlchemy 业务表
2. 策略编译和领域归一化逻辑
3. 快照、UE 上下文、SLA 评估这些确定性模块

### 18.3 一句务实建议

不要把“是否引入 LangChain”理解成“大重构成全 agent 框架”。  
对这个项目更合理的方向是：

- `保留业务核心`
- `替换 LLM 基础设施胶水`

这才是最省成本、最能提升稳定性的改法。

---

## 19. 官方参考资料

以下都是本次整理时核对过的官方资料主入口：

- LangChain overview  
  https://docs.langchain.com/oss/python/langchain/overview
- Agents  
  https://docs.langchain.com/oss/python/langchain/agents
- Models  
  https://docs.langchain.com/oss/python/langchain/models
- Messages  
  https://docs.langchain.com/oss/python/langchain/messages
- Tools  
  https://docs.langchain.com/oss/python/langchain/tools
- Runtime  
  https://docs.langchain.com/oss/python/langchain/runtime
- Structured output  
  https://docs.langchain.com/oss/python/langchain/structured-output
- LangGraph memory  
  https://docs.langchain.com/oss/python/langgraph/add-memory
- OpenAIEmbeddings  
  https://docs.langchain.com/oss/python/integrations/text_embedding/openai
- Text splitters  
  https://docs.langchain.com/oss/python/integrations/splitters
- RecursiveCharacterTextSplitter  
  https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter
- PGVector  
  https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector
- ChatPromptTemplate API  
  https://api.python.langchain.com/en/latest/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
- Runnable API  
  https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.Runnable.html
- PydanticOutputParser API  
  https://api.python.langchain.com/en/latest/core/output_parsers/langchain_core.output_parsers.pydantic.PydanticOutputParser.html

---

## 20. 后续如果你要我继续做

如果你下一步要我继续落代码，我建议优先做下面二选一：

1. 先把 `agents/MemoryManager.py` 替换为 LangGraph memory 版本
2. 先把 `IntentEncodingAgent` 改成 `create_agent + structured output` 版本

我更推荐先做第 2 个，因为改动更小、见效更快。
