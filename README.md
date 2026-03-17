# LLM-MCP 多轮任务型对话 Agent

> 一个基于大语言模型（豆包）+ MCP 协议的车载语音对话 Agent 系统，支持意图识别、槽位抽取、多轮对话管理、工具调用等能力。

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [微服务说明](#微服务说明)
- [项目结构](#项目结构)

---

## 项目简介

本项目是一个**车载语音对话 Agent**，采用 **微服务 + LLM Function Calling + MCP（Model Context Protocol）** 的混合架构。系统支持：

- **任务型对话**：天气查询、音乐播放、地图导航等车载常用场景
- **多轮对话**：基于 Redis 的上下文管理，支持指代消解与信息补全
- **闲聊兜底**：不属于任务域的 Query 路由至豆包 Bot 进行流式闲聊/百科问答
- **拒识能力**：对非车载场景的 Query 进行过滤拒识

---

## 系统架构

```
用户输入
    │
    ▼
[start.py - WebSocket 主控服务 :8080]
    │
    ├──[并行调用]─────────────────────────────────────┐
    │                                                  │
    ▼                                                  ▼
[rewrite.py]                              [arbitration.py]
Query 改写（LLM 指代消解）               仲裁分域（task / faq / chat）
    │                                                  │
    ▼                                     ┌────────────┤
[改写后 Query]                            │            │
    │                                  task 域      chat 域
    ├──[并行]─────────────────────┐       │            │
    ▼                             ▼       ▼            ▼
[nlu.py]                   [reject.py]  [NLU 微服务] [stream_chat.py]
NLU 语义理解               拒识判断     :8009        流式闲聊/百科
                                │
                                ▼
                  [chatnlu_infer.py - FastAPI :8009]
                          │
                          ├── BERT 意图召回 (top5)
                          ├── Function Calling 精确抽取
                          └── DM Factory → [weather / music / maps]
                                                │
                                                ▼
                                    [MCP Client] → MCP 工具服务器
                                           ├── amp_server（高德地图）
                                           └── music_server（QQ音乐）
                                                │
                                                ▼
                                    [nlg.py] → LLM 生成自然语言回复
```

**核心处理流程（task 域）：**

1. LLM 对 Query 进行改写（指代消解、信息补全）
2. BERT 模型快速召回 top5 候选意图类别
3. 豆包 LLM Function Calling 精确匹配意图并抽取槽位
4. DM 层通过 MCP 协议调用外部工具（高德地图 / QQ音乐）
5. NLG 将工具结果转为自然语言回复，推送给用户

**拒识策略（chat 域）：**

1. BERT-tiny 拒识模型判断 Query 是否属于车载场景
2. 若被拒识，用 LLM 相关性判断辅助救回（判断与上轮是否相关）
3. 通过则转入流式闲聊/百科兜底（豆包 Bot 流式输出）

---

## 功能特性

- **多轮对话管理**：基于 Redis 存储对话历史，LLM 负责指代消解与 Query 改写
- **意图识别**：BERT 模型快速召回 + LLM Function Calling 精确匹配，兼顾速度与准确性
- **槽位抽取**：LLM 自动抽取结构化槽位，支持日期/地点/歌曲等多类型后处理规范化
- **MCP 工具调用**：通过标准 MCP 协议调用高德地图（天气/路线/POI）和 QQ音乐
- **流式输出**：闲聊回复按标点分片实时推送，提升交互体验
- **拒识过滤**：BERT-tiny 轻量模型快速过滤非车载场景 Query
- **端到端评估**：内置 `e2e_score.py` 支持多轮对话准确率自动评估

---

## 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | Flask + Flask-SocketIO（WebSocket 实时通信） |
| 微服务框架 | FastAPI + Uvicorn |
| LLM 接入 | 字节跳动豆包（Doubao）API，支持 Function Calling 和 Bot Chat |
| MCP 协议 | `mcp==1.7.0`（Model Context Protocol SDK），FastMCP 构建工具服务器 |
| NLU 模型 | BERT / BERT-tiny（PyTorch + Transformers 微调） |
| 外部 API | 高德地图 REST API、QQ音乐 API（qqmusic-api-python） |
| 缓存/会话 | Redis（存储多轮对话历史、仲裁/改写历史） |
| 并发 | `ThreadPoolExecutor`（线程池并行调用各微服务） |
| 深度学习 | PyTorch 2.6 + Transformers 4.51 |

---

## 快速开始

### 环境要求

- Python 3.10+
- Redis 服务（本地或远程）
- CUDA（可选，用于 BERT 模型推理加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

参考 `config/config.ini`，设置以下环境变量：

```bash
export API_KEY="Bearer your_doubao_api_key"
export BASE_URL="https://ark.cn-beijing.volces.com/api/v3/chat/completions"
export BOT_URL="https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions"
export AMAP_MAPS_API_KEY="your_amap_api_key"

# 微服务地址（默认本地）
export REJECT_URL="http://127.0.0.1:8007/reject-server/v1"
export INTENT_URL="http://127.0.0.1:8008/intent-server/v1"
export NLU_URL="http://127.0.0.1:8009/chatnlu-server/v1"
export ENTRY_URL="http://127.0.0.1:8080/request_nlu"
```

### 启动各微服务

**1. 启动拒识服务（端口 8007）**

```bash
python train/reject_infer.py
```

**2. 启动意图分类服务（端口 8008）**

```bash
python train/intent_infer.py
```

**3. 启动 NLU 微服务（端口 8009）**

```bash
python function_call/chatnlu_infer.py
```

**4. 启动主对话服务（端口 8080）**

```bash
python start.py
```

**5. 命令行测试**

```bash
python dialog.py
```

---

## 配置说明

| 配置项 | 说明 |
|--------|------|
| `API_KEY` | 豆包 LLM API 密钥（用于 Function Calling、改写、仲裁、NLG） |
| `BASE_URL` | 豆包聊天补全接口地址 |
| `BOT_URL` | 豆包 Bot 接口地址（用于流式闲聊） |
| `AMAP_MAPS_API_KEY` | 高德地图 REST API 密钥 |
| `REJECT_URL` | 拒识微服务地址 |
| `INTENT_URL` | 意图分类微服务地址 |
| `NLU_URL` | NLU 语义理解微服务地址 |

---

## 微服务说明

| 服务 | 端口 | 入口文件 | 功能 |
|------|------|----------|------|
| 主对话服务 | 8080 | `start.py` | WebSocket 总调度，协调全流程 |
| 拒识服务 | 8007 | `train/reject_infer.py` | BERT-tiny 二分类，判断是否属于车载场景 |
| 意图分类服务 | 8008 | `train/intent_infer.py` | BERT 多分类，召回 top5 候选意图 |
| NLU 语义理解服务 | 8009 | `function_call/chatnlu_infer.py` | 意图+槽位抽取，触发 DM 执行工具 |

---

## 项目结构

```
LLM-mcp多轮任务型对话agent/
├── start.py                    # 主入口：Flask/SocketIO 对话服务（:8080）
├── dialog.py                   # 命令行测试客户端
├── prompts.py                  # LLM Prompt 模板集中管理
├── e2e_score.py                # 端到端对话准确率评估
├── requirements.txt            # Python 依赖
│
├── config/
│   ├── config.ini              # 环境变量配置（API Key、服务 URL）
│   ├── class.txt               # 意图 ID → 函数名 → 意图名映射表
│   ├── slot_intent.json        # 槽位名映射配置
│   └── new_map.json            # 其他映射配置
│
├── client/                     # 各微服务 HTTP 调用客户端
│   ├── arbitration.py          # 仲裁分域（task / faq / chat）
│   ├── stream_chat.py          # 流式闲聊客户端
│   ├── nlu.py                  # NLU 服务调用
│   ├── reject.py               # 拒识服务调用
│   ├── rewrite.py              # Query 改写（LLM 指代消解）
│   ├── correlation.py          # 多轮相关性判断
│   └── nlg.py                  # 自然语言生成
│
├── function_call/              # NLU 微服务
│   ├── chatnlu_infer.py        # NLU FastAPI 服务（:8009）
│   ├── function.py             # 所有 Function Call 工具定义
│   ├── slot_process.py         # 槽位后处理
│   └── dm/                     # 对话管理层（DM）
│       ├── factory.py          # DM 工厂（按领域分发）
│       ├── weather.py          # 天气领域 DM
│       ├── music.py            # 音乐领域 DM
│       └── maps.py             # 地图领域 DM
│
├── mcp_core/                   # MCP 核心
│   ├── mcp_client.py           # MCP 客户端（连接并调用工具服务器）
│   ├── amp_server.py           # 高德地图 MCP 服务器（天气/路线/POI）
│   └── music_server.py         # QQ音乐 MCP 服务器（歌曲搜索）
│
├── train/                      # 模型训练与推理
│   ├── run.py                  # 训练入口
│   ├── train_eval.py           # 训练 & 评估逻辑
│   ├── data_helper.py          # 数据集加载与处理
│   ├── intent_infer.py         # 意图分类推理服务（:8008）
│   ├── reject_infer.py         # 拒识推理服务（:8007）
│   ├── models/
│   │   ├── bert.py             # BERT 意图分类模型
│   │   └── bert_tiny.py        # BERT-tiny 拒识模型
│   └── core/                   # BERT 核心组件（modeling, tokenization 等）
│
├── utils/
│   ├── logger.py               # 日志工具（带 trace_id）
│   └── redis_tool.py           # Redis 连接池封装
│
└── test/                       # 测试数据与基准测试
    ├── data/                   # 测试集（单轮/多轮）
    ├── result/                 # 测试输出结果
    └── *_benchmark.py          # 各服务基准测试脚本
```

---

## 模型训练

如需重新训练意图分类或拒识模型：

```bash
# 训练意图分类模型（BERT）
python train/run.py --model bert --task intent

# 训练拒识模型（BERT-tiny）
python train/run.py --model bert_tiny --task reject
```

---

## 评估

```bash
# 端到端多轮对话准确率评估
python e2e_score.py
```
