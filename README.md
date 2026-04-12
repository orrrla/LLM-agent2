# LLM-MCP 多轮任务型对话 Agent

> 一个面向车载场景的多轮任务型对话 Agent，基于 `LLM + BERT NLU + MCP + Redis + 长期记忆 + Deep Research` 组合实现，支持任务执行、闲聊兜底、长期记忆注入、意图消歧和轻量联网调研。

## 目录

- [项目简介](#项目简介)
- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [Deep Research 与长期记忆](#deep-research-与长期记忆)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [关键配置](#关键配置)
- [项目结构](#项目结构)
- [模型训练与评估](#模型训练与评估)

## 项目简介

本项目最初是一个以车载语音任务为中心的对话 Agent，当前版本在原有 `微服务 + Function Calling + MCP` 架构基础上，继续补入了两条重要能力：

- **长期记忆链**：把对话统一落盘为本地会话事实源，增量蒸馏为结构化对象，并通过 `pgvector + BM25` 做混合检索，为改写、仲裁和 NLU 提供跨轮上下文。
- **Deep Research 最小链路**：在 `chat fallback` 路径中增加“优先联网、失败退化到长期记忆/普通闲聊”的研究型回答能力。

因此，项目现在同时覆盖三类能力：

- **任务型对话**：天气、地图、音乐等 MCP/DM 驱动的车载任务
- **多轮会话**：Redis 短期上下文 + 本地会话持久化 + 长期记忆检索
- **开放域兜底**：普通闲聊、百科问答，以及触发式 deep research

## 核心能力

- **多轮任务对话**：LLM 改写 + BERT 意图召回 + Function Calling 槽位抽取 + DM 执行工具
- **意图消歧**：对 top intent 置信度接近的情况，生成候选列表并等待用户二次确认
- **MCP 工具调用**：通过标准 MCP 协议访问高德地图、QQ 音乐等能力
- **长期记忆 v2**：会话 JSON 落盘、exchange 切分、结构化蒸馏、混合检索、上下文注入
- **Deep Research 最小链路**：Tavily 搜索 + URL 抓取 + LLM 汇总，失败时退回长期记忆/普通 Bot
- **流式输出**：任务答复、闲聊和 research 都复用统一 `SocketIO` 帧协议
- **可审计性**：对话保存在 `memory_store/sessions/`，可回放、可蒸馏、可检索

## 系统架构

```text
用户输入
    │
    ▼
[start.py - WebSocket 主控服务 :8080]
    │
    ├── rewrite.py        → Query 改写（短期历史 + 长期记忆）
    ├── arbitration.py    → 路由仲裁（task / faq / chat）
    ├── reject.py         → 拒识模型
    ├── correlation.py    → 多轮相关性判断
    └── nlu.py            → NLU 微服务

task 路径：
    start.py
      → function_call/chatnlu_infer.py
      → BERT top5 意图召回
      → Function Calling 槽位抽取
      → DM Factory
      → MCP Client
      → amp_server / music_server
      → nlg.py

chat / faq fallback 路径：
    start.py
      → should_use_deep_research()
      → deep_research.py
          → Tavily Search
          → fetch_url_text()
          → LLM 汇总
          → 失败时回退 memory_module_v2 / stream_chat.py

长期记忆路径：
    start.py
      → SessionManager 落盘 memory_store/sessions/*.json
      → 异步 distill_session(session_id)
      → memory_module_v2
      → pgvector + BM25
      → rewrite / arbitration / chatnlu_infer 注入
```

## Deep Research 与长期记忆

### Deep Research 最小链路

当前版本的 deep research 不在任务域里做 MCP 编排，而是先挂在 `chat fallback` 路径，用来处理这类请求：

- “帮我调研一下……”
- “对比一下 A 和 B，给出处”
- “帮我查官网 / 最新信息”
- “总结一下这个方向的资料”

执行顺序：

1. 判断当前 query 是否值得进入 deep research
2. 使用 Tavily 做联网搜索
3. 抓取 1 到 3 个候选 URL 正文
4. 调用 LLM 汇总为中文研究结论
5. 如果联网失败或证据不足，退回长期记忆
6. 如果长期记忆也不可用，再退回普通 Bot 闲聊

### 长期记忆

项目引入了一个裁剪版 `memory_module_v2`，目标不是替代现有任务链，而是给外层对话流程补足跨轮记忆：

- 会话统一写入 `memory_store/sessions/*.json`
- 会话被切分为 `exchange`
- 每个 `exchange` 进一步蒸馏成结构化记忆对象
- 检索时结合 `dense retrieval + keyword retrieval`
- 记忆结果可被注入到：
  - `client/rewrite.py`
  - `client/arbitration.py`
  - `function_call/chatnlu_infer.py`

## 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | Flask + Flask-SocketIO |
| 微服务框架 | FastAPI + Uvicorn |
| LLM 接入 | 豆包 Chat / Bot API |
| NLU 模型 | BERT / BERT-tiny（PyTorch + Transformers） |
| MCP 协议 | `mcp==1.7.0`，FastMCP 工具服务 |
| 外部 API | 高德地图、QQ音乐、Tavily |
| 短期状态 | Redis |
| 长期记忆 | 本地 JSON、`psycopg`、`pgvector`、`rank-bm25` |
| 网页抓取 | `httpx` + `html2text` |
| 并发 | `ThreadPoolExecutor` |

## 快速开始

### 环境要求

- Python 3.10+
- Redis
- PostgreSQL + `pgvector`（启用长期记忆时）
- CUDA（可选，用于 BERT 推理）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础环境变量

最小任务链运行需要：

```bash
export API_KEY="Bearer your_doubao_api_key"
export BASE_URL="https://ark.cn-beijing.volces.com/api/v3/chat/completions"
export BOT_URL="https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions"
export AMAP_MAPS_API_KEY="your_amap_api_key"

export REJECT_URL="http://127.0.0.1:8007/reject-server/v1"
export INTENT_URL="http://127.0.0.1:8008/intent-server/v1"
export NLU_URL="http://127.0.0.1:8009/chatnlu-server/v1"
```

### 启用长期记忆

```bash
export MEMORY_BACKEND="v2"
export MEMORY_V2_INJECT="always"
export POSTGRES_DSN="postgresql://user:password@127.0.0.1:5432/agent"
export MEMORY_EMBEDDING_URL="https://your-embedding-endpoint"
export MEMORY_EMBEDDING_MODEL="your-embedding-model"
export MEMORY_EMBEDDING_API_KEY="Bearer your_embedding_key"
```

### 启用 Deep Research

```bash
export DEEP_RESEARCH_ENABLED="true"
export TAVILY_API_KEY="tvly-xxxx"
export DEEP_RESEARCH_MODEL="ep-xxxx"
export DEEP_RESEARCH_MAX_RESULTS="5"
export DEEP_RESEARCH_FETCH_TOP_N="3"
```

### 启动服务

1. 启动拒识服务

```bash
python train/reject_infer.py
```

2. 启动意图分类服务

```bash
python train/intent_infer.py
```

3. 启动 NLU 微服务

```bash
python function_call/chatnlu_infer.py
```

4. 启动主服务

```bash
python start.py
```

5. 命令行调试

```bash
python dialog.py
```

## 关键配置

### 模型与 API

| 配置项 | 说明 |
|--------|------|
| `API_KEY` | 豆包通用补全接口密钥 |
| `BASE_URL` | 豆包补全接口 |
| `BOT_URL` | 豆包 Bot 接口 |
| `REWRITE_MODEL` | 改写模型 |
| `ARBITRATION_MODEL` | 仲裁模型 |
| `NLU_FC_MODEL` | Function Calling 模型 |
| `NLG_MODEL` | NLG / 汇总模型 |
| `BOT_CHAT_MODEL` | 普通闲聊 Bot 模型 |
| `DEEP_RESEARCH_MODEL` | Deep research 汇总模型 |
| `DEEP_RESEARCH_TRIGGER_MODEL` | 可选的 deep research 触发判断模型 |

### Redis 与微服务

| 配置项 | 说明 |
|--------|------|
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | Redis 连接配置 |
| `REJECT_URL` | 拒识服务地址 |
| `INTENT_URL` | 意图服务地址 |
| `NLU_URL` | NLU 服务地址 |

### 长期记忆

| 配置项 | 说明 |
|--------|------|
| `MEMORY_BACKEND` | `off` / `v2` |
| `MEMORY_V2_INJECT` | `off` / `always` |
| `MEMORY_SESSIONS_DIR` | 本地会话落盘目录 |
| `POSTGRES_DSN` | Postgres DSN |
| `MEMORY_EMBEDDING_URL` | Embedding 接口地址 |
| `MEMORY_EMBEDDING_MODEL` | Embedding 模型名 |
| `MEMORY_EMBEDDING_API_KEY` | Embedding 密钥 |

### Deep Research

| 配置项 | 说明 |
|--------|------|
| `DEEP_RESEARCH_ENABLED` | 是否启用 deep research |
| `TAVILY_API_KEY` | Tavily 搜索密钥 |
| `DEEP_RESEARCH_TIMEOUT` | research 超时 |
| `DEEP_RESEARCH_MAX_RESULTS` | 搜索结果数量 |
| `DEEP_RESEARCH_FETCH_TOP_N` | 抓取正文的 URL 数量 |

## 项目结构

```text
LLM-mcp多轮任务型对话agent/
├── start.py
├── prompts.py
├── dialog.py
├── requirements.txt
├── client/
│   ├── arbitration.py
│   ├── correlation.py
│   ├── deep_research.py
│   ├── nlg.py
│   ├── nlu.py
│   ├── reject.py
│   ├── rewrite.py
│   └── stream_chat.py
├── config/
│   ├── config.ini
│   ├── runtime.py
│   ├── class.txt
│   ├── slot_intent.json
│   └── new_map.json
├── function_call/
│   ├── chatnlu_infer.py
│   ├── function.py
│   ├── slot_process.py
│   └── dm/
├── mcp_core/
│   ├── amp_server.py
│   ├── mcp_client.py
│   └── music_server.py
├── memory_module_v2/
│   ├── api.py
│   ├── config.py
│   ├── distill.py
│   ├── retrieval.py
│   ├── schema.sql
│   ├── segmenter.py
│   ├── session_reader.py
│   ├── store.py
│   └── models.py
├── memory_store/
│   └── sessions/
├── service/
│   └── session_manager.py
├── train/
├── utils/
└── test/
```

## 模型训练与评估

### 训练

```bash
python train/run.py --model bert --task intent
python train/run.py --model bert_tiny --task reject
```

### 评估

```bash
python e2e_score.py
```

## 当前限制

- `faq` 仍然和 `chat fallback` 共用同一条兜底路径，尚未拆成独立 research/faq 服务。
- deep research 当前是“最小链路”，不是通用 Planner，不会做复杂多轮网页代理操作。
- 长期记忆依赖 Postgres 和 embedding 接口，未配置时会自动退化，不影响主任务链路。
