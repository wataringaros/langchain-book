import json
import logging
import os
import re
import time
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import MomentoChatMessageHistory
from langchain.schema import HumanMessage, LLMResult, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
# import langchain_core.pro

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler

CHAT_UPDATE_INTERVAL_SEC = 1

load_dotenv()

# ログ
SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ボットトークンを使ってアプリを初期化します
app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    process_before_response=True,
)


class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC
        # 投稿を更新した累計回数カウンタ
        self.update_count = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.message += token

        now = time.time()
        if now - self.last_send_time > self.interval:
            app.client.chat_update(
                channel=self.channel, ts=self.ts, text=f"{self.message}\n\nTyping..."
            )
            self.last_send_time = now
            self.update_count += 1

            # update_countが現在の更新間隔X10より多くなるたびに更新間隔を2倍にする
            if self.update_count / 10 > self.interval:
                self.interval = self.interval * 2

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        message_context = "OpenAI APIで生成される情報は不正確または不適切な場合がありますが、当社の見解を述べるものではありません。"
        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},
            {"type": "divider"},
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": message_context}],
            },
        ]
        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message,
            blocks=message_blocks,
        )


# @app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    message = re.sub("<@.*>", "", event["text"])

    # 投稿のキー(=Momentoキー)：初回=event["ts"],2回目以降=event["thread_ts"]
    id_ts = event["ts"]
    if "thread_ts" in event:
        id_ts = event["thread_ts"]

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"]

    history = MomentoChatMessageHistory.from_client_params(
        id_ts,
        os.environ["MOMENTO_CACHE"],
        timedelta(hours=int(os.environ["MOMENTO_TTL"])),
    )

    sys_prompts = "You are a good assistant."
    # human_prompts = ""
    # human_prompts.extend(history.messages)

    # 検索
    index_name = "langchain-book"
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    vectorstore_from_docs = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    docs = vectorstore_from_docs.similarity_search(message)

    # 検索結果の結合
    # search_result = ""
    # for i in range(len(search_result)):
    #     message += docs[i].page_content
    search_result = "\n".join(doc.page_content for doc in docs)
    prompts = [("system", sys_prompts), ("human", f"以下の内容を{message}を基に次の内容をまとめてください。###{search_result}###")]
    logger.info(f"prompts: {prompts}")
    # prompts = f"You are a good assistant. Below is the content based on your message '{message}'. Please summarize the following content: ###{search_result}###"

    # prompts.append(HumanMessage(content=f"以下の内容を{message}まとめてください。###{search_result}###"))

    # history.add_user_message(message)

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)
    llm = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        streaming=True,
        callbacks=[callback],
    )

    ai_message = llm.invoke(prompts)
    logger.info(f"ai_message: {ai_message}")
    history.add_message(ai_message)


def just_ack(ack):
    ack()

app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

# ソケットモードハンドラーを使ってアプリを起動します
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


def handler(event, context):
    logger.info("handler called")
    header = event["headers"]
    logger.info(json.dumps(header))

    if "x-slack-retry-num" in header:
        logger.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
        return 200

    # AWS Lambda 環境のリクエスト情報を app が処理できるよう変換してくれるアダプター
    slack_handler = SlackRequestHandler(app=app)
    # 応答はそのまま AWS Lambda の戻り値として返せます
    return slack_handler.handle(event, context)
