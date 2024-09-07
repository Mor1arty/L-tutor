# encoding:utf-8

import time

from toolhouse import Toolhouse
from groq import Groq

from bot.bot import Bot
from bot.l_tutor.l_tutor_session import LTutorSession
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf

user_session = dict()


# OpenAI对话模型API (可用)
class LTutorBot(Bot):
    def __init__(self):
        super().__init__()

        self.groq_client = Groq(api_key=conf().get("groq_api_key"))
        # self.th = Toolhouse(access_token=conf().get("toolhouse_api_key"))

        self.sessions = SessionManager(LTutorSession,
                                       model=conf().get("l-tutor-model") or "llama3-groq-70b-8192-tool-use-preview")
        self.args = {
            "model": conf().get("l-tutor-model") or "llama3-groq-70b-8192-tool-use-preview",  # 对话模型的名称
            "temperature": conf().get("temperature", 0.9),  # 值在[0,1]之间，越大表示回复越具有不确定性
            "max_tokens": 1200,  # 回复最大的字符数
            "top_p": 1,
            "frequency_penalty": conf().get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "presence_penalty": conf().get("presence_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "timeout": conf().get("request_timeout", None),  # 重试超时时间，在这个时间内，将会自动重试
            "stop": ["\n\n\n"],
        }

    def reply(self, query, context=None):
        # acquire reply content
        if context and context.type:
            if context.type == ContextType.TEXT:
                logger.info("[L-tutor] query={}".format(query))
                session_id = context["session_id"]
                reply = None
                if query == "#清除记忆":
                    self.sessions.clear_session(session_id)
                    reply = Reply(ReplyType.INFO, "记忆已清除")
                elif query == "#清除所有":
                    self.sessions.clear_all_session()
                    reply = Reply(ReplyType.INFO, "所有人记忆已清除")
                else:
                    session = self.sessions.session_query(query, session_id)
                    result = self.reply_text(session)
                    total_tokens, completion_tokens, reply_content = (
                        result["total_tokens"],
                        result["completion_tokens"],
                        result["content"],
                    )
                    logger.debug(
                        "[L-tutor] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(str(session), session_id, reply_content, completion_tokens)
                    )

                    if total_tokens == 0:
                        reply = Reply(ReplyType.ERROR, reply_content)
                    else:
                        self.sessions.session_reply(reply_content, session_id, total_tokens)
                        reply = Reply(ReplyType.TEXT, reply_content)
                return reply

    def reply_text(self, session: LTutorSession, retry_count=0):
        try:
            # response = self.groq_client.chat.completions.create(prompt=str(session), **self.args)
            # self.th.set_metadata("session_id", session.session_id)
            response = self.groq_client.chat.completions.create(messages=session.messages, **self.args)
            # tool_run_res = self.th.run_tools(response)
            # session.messages.extend(tool_run_res)
            # response = self.groq_client.chat.completions.create(messages=session.messages, **self.args)

            logger.info("[L-tutor] response={}".format(response))
            res_content = response.choices[0].message.content.strip().replace("<|endoftext|>", "")
            total_tokens = response.usage.total_tokens
            completion_tokens = response.usage.completion_tokens
            logger.info("[L-tutor] reply={}".format(res_content))
            return {
                "total_tokens": total_tokens,
                "completion_tokens": completion_tokens,
                "content": res_content,
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            logger.warn("[L-tutor] Exception: {}".format(e))
            # if isinstance(e, openai.error.RateLimitError):
            #     logger.warn("[L-tutor] RateLimitError: {}".format(e))
            #     result["content"] = "提问太快啦，请休息一下再问我吧"
            #     if need_retry:
            #         time.sleep(20)
            # elif isinstance(e, openai.error.Timeout):
            #     logger.warn("[OPEN_AI] Timeout: {}".format(e))
            #     result["content"] = "我没有收到你的消息"
            #     if need_retry:
            #         time.sleep(5)
            # elif isinstance(e, openai.error.APIConnectionError):
            #     logger.warn("[OPEN_AI] APIConnectionError: {}".format(e))
            #     need_retry = False
            #     result["content"] = "我连接不到你的网络"
            # else:
            #     logger.warn("[OPEN_AI] Exception: {}".format(e))
            #     need_retry = False
            #     self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[OPEN_AI] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, retry_count + 1)
            else:
                return result
