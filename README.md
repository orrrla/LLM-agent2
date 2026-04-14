# LLM-MCP 多轮任务型对话 Agent

> 一个面向车载场景的多轮任务型对话 Agent，基于 `LLM + BERT NLU + MCP + Redis + 长期记忆 + Deep Research + Planner` 组合实现，支持任务执行、闲聊兜底、长期记忆注入、意图消歧、轻量联网调研与复杂任务规划。

## 目录

- [项目简介](#项目简介)
- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [Planner、Deep Research 与长期记忆](#plannerdeep-research-与长期记忆)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [关键配置](#关键配置)
- [项目结构](#项目结构)
- [模型训练与评估](#模型训练与评估)

## 项目简介

本项目最初是一个以车载语音任务为中心的对话 Agent，当前版本在原有 `微服务 + Function Calling + MCP` 架构基础上，继续补入了三条重要能力：

- **长期记忆链**：把对话统一落盘为本地会话事实源，增量蒸馏为结构化对象，并通过 `pgvector + BM25` 做混合检索，为改写、仲裁和 NLU 提供跨轮上下文。
- **Deep Research 最小链路**：在 `chat fallback` 路径中增加“优先联网、失败退化到长期记忆/普通闲聊”的研究型回答能力。
- **分层式 Planner**：为复杂请求增加 `Plan-Act-Observe-Replan` 规划层，按需编排任务执行、联网调研、长期记忆检索和最终总结。

因此，项目现在同时覆盖三类能力：

- **任务型对话**：天气、地图、音乐等 MCP/DM 驱动的车载任务
- **多轮会话**：Redis 短期上下文 + 本地会话持久化 + 长期记忆检索
- **开放域兜底**：普通闲聊、百科问答，以及触发式 deep research
- **复杂请求规划**：多步骤任务拆解、外部资料调研、记忆检索与统一总结

## 核心能力

- **多轮任务对话**：LLM 改写 + BERT 意图召回 + Function Calling 槽位抽取 + DM 执行工具
- **意图消歧**：对 top intent 置信度接近的情况，生成候选列表并等待用户二次确认
- **MCP 工具调用**：通过标准 MCP 协议访问高德地图、QQ 音乐等能力
- **长期记忆 v2**：会话 JSON 落盘、exchange 切分、结构化蒸馏、混合检索、上下文注入
- **Deep Research 最小链路**：Tavily 搜索 + URL 抓取 + LLM 汇总，失败时退回长期记忆/普通 Bot
- **通用 Planner**：基于 `task / research / memory / respond` 四类步骤做受控编排，并支持 `replan`
- **Gemini 兼容调用**：`planner` 与 `deep research` 可通过统一 LLM 客户端切换到 Gemini REST
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
    ├── planner gate      → 复杂任务判定（keyword / optional LLM gate）
    ├── reject.py         → 拒识模型
    ├── correlation.py    → 多轮相关性判断
    └── nlu.py            → NLU 微服务

planner 路径：
    start.py
      → should_use_planner()
      → planner/engine.py
          → plan(task / research / memory / respond)
          → task     → request_nlu() / DM / MCP
          → research → deep_research.py
          → memory   → memory_module_v2
          → replan   → observation 驱动动态调整
          → respond  → 汇总最终回答

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
          → research_engine（迭代式 research loop）
              → plan（生成子问题/检索策略）
              → Tavily Search（多轮）
              → fetch_url_text()（多轮抓取）
              → critique（判断证据缺口/冲突并调整 query）
              → finalize（LLM 汇总 + 来源）
          → 失败时回退 memory_module_v2 / stream_chat.py

长期记忆路径：
    start.py
      → SessionManager 落盘 memory_store/sessions/*.json
      → 异步 distill_session(session_id)
      → memory_module_v2
      → pgvector + BM25
      → rewrite / arbitration / chatnlu_infer 注入
```

## Planner、Deep Research 与长期记忆

### Planner

当前版本的 planner 不是替换原有任务链，而是做一层“复杂任务编排器”：

- 简单任务仍然走既有 `rewrite -> arbitration -> NLU -> DM -> MCP` 链路
- 复杂请求由 `should_use_planner()` 判定后进入 `planner/engine.py`
- Planner 先生成结构化步骤，再按 `task / research / memory / respond` 执行
- 每一步会产出 observation，必要时触发 `replan`
- `task` 复用现有 NLU/DM/MCP，`research` 复用 Deep Research，`memory` 复用长期记忆检索

适合的 query 示例：

- “先帮我查一下明天杭州天气，再推荐适合开车听的歌”
- “帮我调研一下某车型，再总结给我”
- “结合我之前的偏好，帮我规划一个执行方案”

### Deep Research（迭代式 research loop）

当前版本的 deep research 不在任务域里做 MCP 编排，而是挂在 `chat fallback` 路径，主要用于处理这类请求：

- “帮我调研一下……”
- “对比一下 A 和 B，给出处”
- “帮我查官网 / 最新信息”
- “总结一下这个方向的资料”

执行顺序（简化）：

1. 判断当前 query 是否值得进入 deep research（关键词 + 可选 LLM gate）
2. 生成 research plan（子问题 + 初始检索 queries + 停止条件）
3. 多轮执行：Tavily 搜索 → 抓取候选 URL → critique（判断缺口/冲突，必要时给出新 query）
4. 满足停止条件后 finalize：LLM 生成研究结论，并附来源
5. 如果联网或总结失败：回退到 manual summary（返回候选资料与来源）
6. 如果联网失败或证据不足：回退到长期记忆；仍不可用再回退普通闲聊

实现位置：

- 兼容入口仍为 `client/deep_research.py` 的 `request_deep_research()`
- 迭代引擎位于 `client/research_engine.py`（plan / critique / finalize）

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
| LLM 接入 | OpenAI-compatible Chat API + Gemini REST（当前用于 planner / deep research） |
| NLU 模型 | BERT / BERT-tiny（PyTorch + Transformers） |
| MCP 协议 | `mcp==1.7.0`，FastMCP 工具服务 |
| 外部 API | 高德地图、QQ音乐、Tavily |
| 短期状态 | Redis |
| 长期记忆 | 本地 JSON、`psycopg`、`pgvector`、`rank-bm25` |
| 网页抓取 | `httpx` + `html2text` |
| Planner | JSON 计划、`task/research/memory/respond`、Plan-Act-Observe-Replan |
| 并发 | `ThreadPoolExecutor` |

## 快速开始

### 环境要求

- Python 3.10+
- Redis
- PostgreSQL + `pgvector`（启用长期记忆时）
- CUDA（可选，用于 BERT 推理）
- macOS 可直接使用精简依赖文件，无需安装 CUDA 相关包

### 安装依赖

```bash
pip install -r requirements.txt
```

macOS / conda `agent` 环境推荐：

```bash
conda activate agent
pip install -r requirements-macos-agent.txt
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

如果你希望 planner / deep research 直接调用 Gemini，可额外配置：

```bash
export GEMINI_API_KEY="your_gemini_api_key"
export PLANNER_MODEL="gemini-2.0-flash"
export DEEP_RESEARCH_MODEL="gemini-2.0-flash"
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

### 启用 Planner

```bash
export PLANNER_ENABLED="true"
export PLANNER_MODEL="ep-xxxx"
export PLANNER_GATE_MODEL=""
export PLANNER_REPLAN_MODEL="ep-xxxx"
export PLANNER_MAX_STEPS="5"
export PLANNER_MAX_REPLANS="2"
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
| `GEMINI_API_KEY` | Gemini REST 密钥；当模型名以 `gemini` 开头时由统一 LLM 客户端使用 |
| `REWRITE_MODEL` | 改写模型 |
| `ARBITRATION_MODEL` | 仲裁模型 |
| `NLU_FC_MODEL` | Function Calling 模型 |
| `NLG_MODEL` | NLG / 汇总模型 |
| `BOT_CHAT_MODEL` | 普通闲聊 Bot 模型 |
| `DEEP_RESEARCH_MODEL` | Deep research 汇总模型 |
| `DEEP_RESEARCH_TRIGGER_MODEL` | 可选的 deep research 触发判断模型 |
| `PLANNER_MODEL` | Planner 初始规划 / 最终总结模型 |
| `PLANNER_GATE_MODEL` | 可选的 planner 入口判定模型 |
| `PLANNER_REPLAN_MODEL` | Planner 重规划模型 |

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

### Planner

| 配置项 | 说明 |
|--------|------|
| `PLANNER_ENABLED` | 是否启用 planner |
| `PLANNER_TIMEOUT` | planner 单轮 LLM 超时 |
| `PLANNER_MAX_STEPS` | 最大执行步骤数 |
| `PLANNER_MAX_REPLANS` | 最大重规划次数 |

## 项目结构

```text
LLM-mcp多轮任务型对话agent/
├── start.py
├── prompts.py
├── dialog.py
├── requirements.txt
├── requirements-macos-agent.txt
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
├── planner/
│   ├── __init__.py
│   └── engine.py
├── service/
│   └── session_manager.py
├── train/
├── utils/
│   ├── llm_client.py
│   ├── logger.py
│   └── redis_tool.py
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
- deep research 当前仍是“最小链路”，以搜索、抓取和总结为主，不会做复杂浏览器代理操作。
- planner 当前是分层式轻量编排器，工具类型固定为 `task / research / memory / respond`，不等同于通用 autonomous agent 框架。
- 长期记忆依赖 Postgres 和 embedding 接口，未配置时会自动退化，不影响主任务链路。
