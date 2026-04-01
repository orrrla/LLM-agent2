# -*- coding: utf-8 -*-
import os
import json
import uuid
import random
import numpy as np
import requests
import base64
import time
import uvicorn
import prompts
from slot_process import intent_slot
from function import tools1
from fastapi import FastAPI, Request
from utils import logger
from dm.factory import DMFactory


## 创建FastAPI应用
app = FastAPI()


MAX_CONF = 0.98
TIMEOUT = 5
INTENT_URL = os.environ["INTENT_URL"]
DOUBAO_API_KEY = os.environ["API_KEY"]
DOUBAO_URL = os.environ["BASE_URL"]
DISAMBIG_TOP1_MIN = float(os.getenv("DISAMBIG_TOP1_MIN", "0.70"))
DISAMBIG_MARGIN_MIN = float(os.getenv("DISAMBIG_MARGIN_MIN", "0.08"))
DISAMBIG_TOPK = int(os.getenv("DISAMBIG_TOPK", "3"))


id2func = {}
func2name = {}
name2id = {}
with open("../config/class.txt", 'r', encoding='utf-8') as mapfile:
    for line in mapfile:
        id, name, func = line.strip().split(":")
        id2func[id] = func
        func2name[func] = name
        name2id[name] = id

tool_map = {}
with open("../config/slot_intent.json", "r", encoding="utf-8") as slotfile:
    slot_map = json.load(slotfile)
    for item in tools1:
        name = item["function"]["name"]
        if name not in tool_map.keys():
            lst = [item]
            new_dict = {name: lst}
            tool_map.update(new_dict)
        else:
            tool_map.get(name).append(item)


def send_messages(messages, tool_lst):
    headers = {
        "Authorization": DOUBAO_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "model": "ep-20250106153928-kh8t7",
        "messages": messages,
        "tools": tool_lst,
        "temperature": 1e-6,
        "top_p": 0
    }
    try:
        response = requests.post(
            DOUBAO_URL,
            headers=headers,
            data=json.dumps(data),
            timeout=TIMEOUT
        )
        res = response.content.decode('utf-8')
        res = json.loads(res)
        return res['choices'][0]['message']['tool_calls']
    except Exception as e:
        logger.error(f"Doubao error: {e}")
        return None


def intent_recall(query, trace_id):
    headers = {'Content-Type': 'application/json'}
    data = {"query": query, "trace_id": str(uuid.uuid1())}
    response = requests.post(url=INTENT_URL, headers=headers, data=json.dumps(data))
    return response.json()


def build_candidates(intent_rec, topk=DISAMBIG_TOPK):
    candidate_ids = intent_rec.get("data", "")
    candidate_scores = intent_rec.get("score", "")
    if not candidate_ids or not candidate_scores:
        return []

    ids = candidate_ids.split(",")
    scores = candidate_scores.split(",")
    candidates = []
    for intent_id, score in zip(ids, scores):
        func_name = id2func.get(str(intent_id))
        if not func_name or func_name == "Unknown":
            continue
        display_name = func2name.get(func_name, "")
        try:
            score_value = float(score)
        except Exception:
            score_value = 0.0
        candidates.append({
            "intent": display_name,
            "intent_id": str(intent_id),
            "function": func_name,
            "score": score_value,
            "display_name": display_name
        })

    # 按意图去重，保留最高分
    unique_map = {}
    for item in candidates:
        key = item["intent_id"]
        if key not in unique_map or item["score"] > unique_map[key]["score"]:
            unique_map[key] = item

    ordered = sorted(unique_map.values(), key=lambda x: x["score"], reverse=True)
    return ordered[:max(1, topk)]


def should_disambiguate(intent_rec, candidates):
    score_text = intent_rec.get("score", "")
    if not score_text:
        return False
    raw_scores = []
    for score in score_text.split(","):
        try:
            raw_scores.append(float(score))
        except Exception:
            continue
    if not raw_scores:
        return False

    top1 = raw_scores[0]
    top2 = raw_scores[1] if len(raw_scores) > 1 else 0.0
    if len(candidates) < 2:
        return False
    if top1 < DISAMBIG_TOP1_MIN:
        return True
    if (top1 - top2) < DISAMBIG_MARGIN_MIN:
        return True
    return False


def run_fc(query, now_tool):
    header = [{"role": "system", "content": prompts.NLU_SYSTEM_PROMPT}]
    context = [{"role": "user", "content": query}]
    messages = header + context
    start_time = time.time()
    result = send_messages(messages, now_tool)
    logger.info(f"llm结果：{result}")
    logger.info(f"function调用时间:{time.time() - start_time}")
    if not result:
        return "未知-无"
    return intent_slot(result, func2name, slot_map)


def predict(query, trace_id, force_intent_id=None):
    try:
        if force_intent_id:
            force_intent_id = str(force_intent_id)
            forced_func = id2func.get(force_intent_id)
            if not forced_func:
                return {
                    "nlu": "未知-无",
                    "needs_disambiguation": False,
                    "candidates": []
                }
            now_tool = tool_map.get(forced_func, [])
            nlu = run_fc(query, now_tool) if now_tool else f"{func2name.get(forced_func, '未知')}-无"
            return {
                "nlu": nlu,
                "needs_disambiguation": False,
                "candidates": []
            }

        start = time.time()
        intent_rec = intent_recall(query, trace_id)
        results = intent_rec["data"].split(",")
        max_score = max([float(k) for k in intent_rec["score"].split(",")])
        logger.info(f"top5：{intent_rec['data']}, cost: {time.time() - start}")
        if str(results[0]) == "3" and max_score > MAX_CONF:
            return {
                "nlu": "未知-无",
                "needs_disambiguation": False,
                "candidates": []
            }

        candidates = build_candidates(intent_rec)
        if should_disambiguate(intent_rec, candidates):
            return {
                "nlu": "未知-无",
                "needs_disambiguation": True,
                "candidates": candidates
            }

        now_tool = []
        for t in results:
            func = id2func.get(t)
            lst_a = tool_map.get(func)
            if lst_a:
                for s in lst_a:
                    now_tool.append(s)
            else:
                continue

        nlu = run_fc(query, now_tool)
    except Exception:
        return {
            "nlu": "未知-无",
            "needs_disambiguation": False,
            "candidates": []
        }

    logger.info(f"返回结果：{nlu}")

    return {
        "nlu": nlu,
        "needs_disambiguation": False,
        "candidates": []
    }


@app.post("/chatnlu-server/v1")
async def inference(request: Request):
    json_info = await request.json()

    begin = time.time()
    query = json_info.get("query")
    enable_dm = json_info.get("enable_dm", True)
    trace_id = json_info.get("trace_id", "1")
    force_intent_id = json_info.get("force_intent_id")

    # 抽取意图和槽位
    predict_result = predict(query, trace_id, force_intent_id=force_intent_id)
    nlu = predict_result.get("nlu", "未知-无")
    needs_disambiguation = predict_result.get("needs_disambiguation", False)
    candidates = predict_result.get("candidates", [])

    if needs_disambiguation:
        response = {
            "query": query,
            "tarce_id": trace_id,
            "intent": "未知",
            "intent_id": name2id.get("未知"),
            "function": "Unknown",
            "slots": {},
            "needs_disambiguation": True,
            "candidates": candidates,
            "disambiguation_id": str(uuid.uuid4())
        }
        response["cost"] = time.time() - begin
        return response

    # NLU后处理
    nlu_items = nlu.split("-")
    intent = nlu_items[0]
    if len(nlu_items) > 2:
        slots_str = "-".join(nlu_items[1:])
    else:
        slots_str = nlu_items[1]

    if slots_str != "无":
        slots = {}
        for item in slots_str.split(","):
            if ":" in item:
                if len(item.split(":")) != 2:
                    continue
                k, v = item.split(":")
                slots[k] = v
    else:
        slots = {}
    intent_id = name2id.get(intent)
    func_name = id2func.get(intent_id) 


    response = {
        "query": query,
        "tarce_id": trace_id,
        "intent": intent,
        "intent_id": intent_id,
        "function": func_name,
        "slots": slots,
        "needs_disambiguation": False,
        "candidates": []
    }

    if enable_dm:
        for name in ["weather", "music", "maps"]:
            dm_result = await DMFactory.get(name)(func_name, query, slots)
            if dm_result:
                tool_response, nlg = dm_result
                response["tool"] = tool_response
                response["nlg"] = nlg

    cost = time.time() - begin
    response["cost"] = cost

    return response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8009, workers=1)
