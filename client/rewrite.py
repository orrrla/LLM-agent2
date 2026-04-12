# -*- coding: utf-8 -*-
import os
import json
import re
import requests
import prompts
from config.runtime import get_app_settings, get_model_settings
from memory_module_v2.api import build_memory_context
from utils import logger
from utils.redis_tool import RedisClient


TTL = 40
MAX_HISTORY = 6
REDIS_KEY = "voice:rewrite_history:{}"
_redis_client = RedisClient() 


def request_rewrite(query, last_answer, sender_id, session_id=None):
    settings = get_app_settings()
    models = get_model_settings()
    session_id = session_id or sender_id

    headers = {
        "Authorization": settings.api_key,
        "Content-Type": "application/json"
    }
    history = _redis_client.get(REDIS_KEY.format(sender_id))
    if history:
        history = json.loads(history)
    else:
        history = []
    history = history[-MAX_HISTORY:]

    if history and last_answer:
        history[-1]["content"] = last_answer

    messages_header = [
        {"role": "system", "content": prompts.REWRITE_SYSTEM_PROMPT}
    ]
    memory_context = build_memory_context(query, session_id)
    if not history and not memory_context:
        result = "否"
    else:
        split_history = [history[i:i+2] for i in range(0, len(history), 2)] if history else []
        history_msgs = []
        for item in split_history:
            if len(item) > 1 and item[1]["content"]:
                msg = "A：{}\nB：{}".format(item[0]["content"], item[1]["content"])
            else:
                msg = "A：{}".format(item[0]["content"])
            history_msgs.append(msg)
        prompt = "#对话历史#\n{}\n".format("\n".join(history_msgs)) if history_msgs else ""
        if memory_context:
            prompt += "#长期记忆#\n{}\n".format(memory_context)
        prompt += "A：{}\n".format(query)
        logger.info(f"对话历史：{prompt}")
        messages_now = [
            {"role": "user", "content": prompt}
        ]
        messages = messages_header + messages_now

        data = {
            "model": models.rewrite_model,
            "messages": messages,
            "temperature": 0.001,
            "top_p": 0,
        }

        response = requests.post(
            settings.base_url,
            headers=headers,
            data=json.dumps(data),
        )
        res = response.content.decode('utf-8')
        res = json.loads(res)
        result = res['choices'][0]['message']['content']

        # 防止误改
        if len(set(result).intersection(query)) < len(query) / 4:
            result = "否"

    if result == "否":
        result = query

    logger.info("改写后：{}".format(result))

    history.append({"role": "user", "content": result})
    history.append({"role": "assistant", "content": ""})

    _redis_client.set(REDIS_KEY.format(sender_id), json.dumps(history, ensure_ascii=False), ex=TTL)

    return result
