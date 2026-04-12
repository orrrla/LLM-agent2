# -*- coding: utf-8 -*-
import json
import os
import copy
import traceback
import time
import re
from pathlib import Path
import redis
import requests
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, make_response
from flask_socketio import SocketIO, emit

import prompts
from config.runtime import get_app_settings
from memory_module_v2.api import distill_session
from service.session_manager import SessionManager
from utils import logger
from utils.redis_tool import RedisClient
from client.arbitration import request_arbitration
from client.deep_research import process_research, request_deep_research, should_use_deep_research
from client.stream_chat import request_chat, process_chat
from client.reject import request_reject
from client.nlu import request_nlu
from client.rewrite import request_rewrite
from client.correlation import request_correlation


socketio = SocketIO(cors_allowed_origins='*', async_mode='threading')
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
socketio.init_app(app)


TTL = 40
REDIS_KEY = "voice:last_service:{}"
DISAMBIG_KEY = "voice:disambiguation:{}"
DISAMBIG_TTL = 120
redis_client = RedisClient() 
thread_pool = ThreadPoolExecutor(max_workers=10)
session_manager = SessionManager(get_app_settings().sessions_dir)


@app.route("/health", methods=["GET"])
def check():
    response = make_response(
        jsonify(health="healthy"),
        200,
        {'content-type': 'application/json'}
    )
    return response


@socketio.on('connect')
def connected_msg():
    manager = socketio.server.manager
    connections_count = len(manager.rooms['/']) - 1
    logger.info(f'当前连接数: {connections_count}')
    logger.info('client connected.')


@socketio.on('disconnect')
def disconnect_msg():
    logger.info('client disconnected.')


def send_msg(nlu_result, func, frame, seq, cost, status):
    if func == "CHAT":
        intent = "闲聊百科" 
        intent_id = "439"
    else:
        intent = "拒识" 
        intent_id = "440"

    nlu_result["intent"] = intent
    nlu_result["intent_id"] = intent_id
    nlu_result["func"] = func
    nlu_result["frame"] = frame
    nlu_result["seq"] = seq
    nlu_result["cost"] = cost
    nlu_result["status"] = status

    emit(
        "request_nlu",
        json.dumps(nlu_result, ensure_ascii=False),
        broadcast=False
    )


def persist_message(session_id, role, content, **metadata):
    session_manager.save_message(
        session_id,
        role,
        content or "",
        metadata=metadata or None,
    )


def schedule_distill(session_id):
    def _runner():
        try:
            distill_session(session_id)
        except Exception as exc:
            logger.warning(f"async distill failed: {exc}")

    thread_pool.submit(_runner)


def handle_chat(handler_bot, nlu_result, query, sender_id, begin):

    # 开始帧
    seq = 1
    nlu_result_begin = copy.deepcopy(nlu_result)
    send_msg(nlu_result_begin, "CHAT", "", seq, time.time() - begin, status=0)

    # 中间帧
    full_answer = ""
    for value in process_chat(handler_bot.result(), query, sender_id):
        nlu_result_chat = copy.deepcopy(nlu_result)
        send_msg(nlu_result_chat, "CHAT", value, seq, time.time() - begin, status=1)
        seq += 1
        full_answer += value
        logger.info(f"Chat Frame:{seq},content:{value}")

    # 结束帧
    if seq > 1:
        nlu_result_end = copy.deepcopy(nlu_result)
        send_msg(nlu_result_begin, "CHAT", "", seq, time.time() - begin, status=2)
        logger.info(f"Chat cost time: {time.time() - begin}")
        return True, full_answer
    else:
        logger.info(f"Chat cost time: {time.time() - begin}")
        return False, full_answer


def handle_research(handler_research, nlu_result, query, sender_id, begin):

    seq = 1
    nlu_result_begin = copy.deepcopy(nlu_result)
    send_msg(nlu_result_begin, "CHAT", "", seq, time.time() - begin, status=0)

    full_answer = ""
    result = handler_research.result()
    for value in process_research(result, query, sender_id):
        nlu_result_chat = copy.deepcopy(nlu_result)
        send_msg(nlu_result_chat, "CHAT", value, seq, time.time() - begin, status=1)
        seq += 1
        full_answer += value
        logger.info(f"Research Frame:{seq},content:{value}")

    if seq > 1:
        send_msg(nlu_result_begin, "CHAT", "", seq, time.time() - begin, status=2)
        logger.info(f"Research cost time: {time.time() - begin}")
        return True, full_answer, result.get("mode", "research"), result.get("sources", [])

    logger.info(f"Research cost time: {time.time() - begin}")
    return False, full_answer, result.get("mode", "empty"), result.get("sources", [])


def clear_disambiguation_state(sender_id):
    redis_client.set(DISAMBIG_KEY.format(sender_id), "", ex=1)


def parse_user_selection(query, candidates):
    text = (query or "").strip()
    if not text or not candidates:
        return None

    index_patterns = [
        r"^第?\s*(\d+)\s*(个|项|条)?$",
        r"^(\d+)$"
    ]
    selected_index = None
    for pattern in index_patterns:
        match_obj = re.match(pattern, text)
        if match_obj:
            selected_index = int(match_obj.group(1)) - 1
            break
    if selected_index is not None and 0 <= selected_index < len(candidates):
        return candidates[selected_index]

    normalized = text.replace(" ", "").lower()
    for item in candidates:
        keys = [
            str(item.get("display_name", "")).replace(" ", "").lower(),
            str(item.get("intent", "")).replace(" ", "").lower(),
            str(item.get("function", "")).replace(" ", "").lower()
        ]
        for key in keys:
            if not key:
                continue
            if normalized == key or normalized in key or key in normalized:
                return item
    return None


def build_disambiguation_nlg(candidates):
    lines = [prompts.DISAMBIGUATION_PROMPT]
    for idx, item in enumerate(candidates, start=1):
        name = item.get("display_name", "") or item.get("intent", "")
        score = item.get("score", 0)
        try:
            score_text = f"{float(score):.2f}"
        except Exception:
            score_text = str(score)
        lines.append(f"{idx}. {name} (置信度:{score_text})")
    return "\n".join(lines)


def emit_disambiguation_response(template, query, trace_id, candidates, begin, message=None):
    result = copy.deepcopy(template)
    result["query"] = query
    result["tarce_id"] = trace_id
    result["intent"] = "候选确认"
    result["intent_id"] = ""
    result["function"] = "Disambiguation"
    result["slots"] = {}
    result["needs_disambiguation"] = True
    result["candidates"] = candidates
    result["nlg"] = message or build_disambiguation_nlg(candidates)
    result["cost"] = time.time() - begin
    emit(
        "request_nlu",
        json.dumps(result, ensure_ascii=False),
        broadcast=False
    )



@socketio.on('request_nlu')
def inference(req):
    begin = time.time()
    json_info = json.loads(req)
    query = json_info.get("query")
    enable_dm = json_info.get("enable_dm")
    sender_id = json_info.get("sender_id", "test")
    trace_id = json_info.get("trace_id", "123")
    session_id = json_info.get("session_id") or sender_id

    nlu_template = {
        "query": query,
        "tarce_id": trace_id,
        "intent": "",
        "intent_id": "",
        "function": "",
        "slots": {},
        "cost": time.time() - begin
    }
    try:
        ori_query = query
        logger.session.trace_id = trace_id
        logger.info("Request Params: {}".format(json_info))
        persist_message(
            session_id,
            "user",
            ori_query,
            trace_id=trace_id,
            sender_id=sender_id,
            query=ori_query,
        )

        disamb_raw = redis_client.get(DISAMBIG_KEY.format(sender_id))
        forced_intent_id = None
        disamb_origin_query = ""
        if disamb_raw:
            try:
                disamb_state = json.loads(disamb_raw)
            except Exception:
                disamb_state = {}
            disamb_candidates = disamb_state.get("candidates", [])
            if not disamb_candidates:
                clear_disambiguation_state(sender_id)
                disamb_state = {}
                disamb_candidates = []
            disamb_origin_query = disamb_state.get("origin_query", "")
            selected = parse_user_selection(ori_query, disamb_candidates)
            if selected:
                forced_intent_id = str(selected.get("intent_id", ""))
                if not forced_intent_id:
                    forced_intent_id = None
                clear_disambiguation_state(sender_id)
                logger.info(f"TraceID:{trace_id}, disambiguation selected: {selected}")
            else:
                emit_disambiguation_response(
                    nlu_template,
                    ori_query,
                    trace_id,
                    disamb_candidates,
                    begin,
                    message=prompts.DISAMBIGUATION_RETRY_PROMPT
                )
                persist_message(
                    session_id,
                    "assistant",
                    prompts.DISAMBIGUATION_RETRY_PROMPT,
                    route="task",
                    trace_id=trace_id,
                    intent="候选确认",
                    function="Disambiguation",
                )
                schedule_distill(session_id)
                return

        last_info = redis_client.get(REDIS_KEY.format(sender_id))
        last_domain, last_query, last_reject, last_answer = "", "", "", ""
        if last_info:
            last_domain, last_query, last_reject, last_answer = last_info.split("#")

        # Query改写
        if forced_intent_id and disamb_origin_query:
            query = disamb_origin_query
        else:
            query = request_rewrite(query, last_answer, sender_id, session_id=session_id)

        # 调用nlu语义
        handler_nlu = thread_pool.submit(request_nlu, query, trace_id, enable_dm, forced_intent_id, session_id)

        # 调用仲裁
        handler_arbitration = thread_pool.submit(request_arbitration, ori_query, sender_id, session_id)

        # 调拒识模型
        handler_reject = thread_pool.submit(request_reject, query, trace_id)

        # 调用相关性模型
        handler_correlation = thread_pool.submit(request_correlation, ori_query, sender_id)

        # 获取仲裁结果
        arbitration_result = handler_arbitration.result()
        if forced_intent_id:
            arbitration_result = "task"

        logger.info(
            f"TraceID:{trace_id}, query:{query}, arbitration result: {arbitration_result}, cost time: {time.time() - begin}")

        # 开始仲裁
        if arbitration_result == "task":
            nlu_result = handler_nlu.result()
            if nlu_result.get("needs_disambiguation", False):
                disamb_state = {
                    "origin_query": ori_query,
                    "candidates": nlu_result.get("candidates", [])
                }
                redis_client.set(
                    DISAMBIG_KEY.format(sender_id),
                    json.dumps(disamb_state, ensure_ascii=False),
                    ex=DISAMBIG_TTL
                )
                nlu_result["nlg"] = build_disambiguation_nlg(nlu_result.get("candidates", []))
                emit(
                    "request_nlu",
                    json.dumps(
                        nlu_result,
                        ensure_ascii=False
                    ),
                    broadcast=False
                )
                persist_message(
                    session_id,
                    "assistant",
                    nlu_result["nlg"],
                    route="task",
                    trace_id=trace_id,
                    intent="候选确认",
                    function="Disambiguation",
                    query=query,
                    rewritten_query=query,
                    candidates=nlu_result.get("candidates", []),
                )
                schedule_distill(session_id)
                return
            # 技能
            if nlu_result.get("function", "") not in ["Unknown"]:
                clear_disambiguation_state(sender_id)
                redis_client.set(REDIS_KEY.format(sender_id), f"SKILL#{query}#1#", ex=TTL)
                emit(
                    "request_nlu",
                    json.dumps(
                        nlu_result,
                        ensure_ascii=False
                    ),
                    broadcast=False
                )
                tool_response = nlu_result.get("tool")
                if tool_response:
                    persist_message(
                        session_id,
                        "tool",
                        str(tool_response),
                        route="task",
                        trace_id=trace_id,
                        intent=nlu_result.get("intent", ""),
                        function=nlu_result.get("function", ""),
                        slots=nlu_result.get("slots", {}),
                    )
                persist_message(
                    session_id,
                    "assistant",
                    nlu_result.get("nlg", "") or prompts.DEFAULT_NLG,
                    route="task",
                    trace_id=trace_id,
                    intent=nlu_result.get("intent", ""),
                    function=nlu_result.get("function", ""),
                    slots=nlu_result.get("slots", {}),
                    query=ori_query,
                    rewritten_query=query,
                )
                schedule_distill(session_id)
            else:
                send_msg(nlu_result, "REJECT", prompts.DEFAULT_NLG, 1, time.time() - begin, status=-1)
                logger.info(f"Query {query} has been rejected.")
                persist_message(
                    session_id,
                    "assistant",
                    prompts.DEFAULT_NLG,
                    route="reject",
                    trace_id=trace_id,
                    query=ori_query,
                    rewritten_query=query,
                )
                schedule_distill(session_id)
        else:
            # 拒识
            reject_result = handler_reject.result()
            if reject_result == 0:
                correlation_result = handler_correlation.result()
                if correlation_result == "是":
                    reject_result = 1 
            if reject_result == 0:
                send_msg(nlu_template, "REJECT", "", 1, time.time() - begin, status=-1)
                logger.info(f"Query {query} has been rejected.")
                persist_message(
                    session_id,
                    "assistant",
                    prompts.DEFAULT_NLG,
                    route="reject",
                    trace_id=trace_id,
                    query=ori_query,
                    rewritten_query=query,
                )
                schedule_distill(session_id)
            else:
                if should_use_deep_research(ori_query, sender_id, session_id=session_id):
                    handler_research = thread_pool.submit(request_deep_research, ori_query, sender_id, session_id)
                    is_hit_chat, full_answer, response_mode, sources = handle_research(
                        handler_research, nlu_template, ori_query, sender_id, begin
                    )
                    if is_hit_chat:
                        redis_client.set(REDIS_KEY.format(sender_id), f"CHAT#{query}#{reject_result}#{full_answer}", ex=TTL)
                        persist_message(
                            session_id,
                            "assistant",
                            full_answer,
                            route=response_mode,
                            trace_id=trace_id,
                            query=ori_query,
                            rewritten_query=query,
                            sources=sources,
                        )
                        schedule_distill(session_id)
                    else:
                        handler_bot = thread_pool.submit(request_chat, ori_query, sender_id)
                        is_hit_chat, full_answer = handle_chat(handler_bot, nlu_template, ori_query, sender_id, begin)
                        if is_hit_chat:
                            redis_client.set(REDIS_KEY.format(sender_id), f"CHAT#{query}#{reject_result}#{full_answer}", ex=TTL)
                            persist_message(
                                session_id,
                                "assistant",
                                full_answer,
                                route="chat",
                                trace_id=trace_id,
                                query=ori_query,
                                rewritten_query=query,
                            )
                            schedule_distill(session_id)
                else:
                    # 百科闲聊兜底
                    handler_bot = thread_pool.submit(request_chat, ori_query, sender_id)
                    is_hit_chat, full_answer = handle_chat(handler_bot, nlu_template, ori_query, sender_id, begin)
                    if is_hit_chat:
                        redis_client.set(REDIS_KEY.format(sender_id), f"CHAT#{query}#{reject_result}#{full_answer}", ex=TTL)
                        persist_message(
                            session_id,
                            "assistant",
                            full_answer,
                            route="chat",
                            trace_id=trace_id,
                            query=ori_query,
                            rewritten_query=query,
                        )
                        schedule_distill(session_id)

    except Exception as e:
        logger.error(
            'TraceID:{}, Internal Server Error!'.format(trace_id))
        logger.error('{}'.format(e))
        traceback.print_exc()
        send_msg(nlu_template, "REJECT", "", 1, time.time() - begin, status=-1)
        persist_message(
            session_id,
            "assistant",
            prompts.DEFAULT_NLG,
            route="error",
            trace_id=trace_id,
            query=query,
        )
        schedule_distill(session_id)

if __name__ == "__main__":
    socketio.run(
        app,
        allow_unsafe_werkzeug=True,
        host='0.0.0.0',
        port=os.getenv("FLASK_SERVER_PORT", 8080)
    )

