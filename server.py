from typing import final, AsyncGenerator, Literal, cast

import cohere
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from const import DATABASE_URL, OPENAI_API_KEY, COHERE_API_KEY, EMBEDDING_MODEL
from model.document import DocumentChunk

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)
co = cohere.ClientV2(api_key=COHERE_API_KEY)

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[text[:4000]]
    )
    return response.data[0].embedding


def get_chunk_similarity(
    document_id: str,
    query_embedding: list[float],
    limit: int = 10,
) -> list[DocumentChunk]:
    with SessionLocal() as session:
        cosine_distance = DocumentChunk.embedding.cosine_distance(
            query_embedding
        ).label("distance")

        query = (
            select(DocumentChunk, cosine_distance)
            .where(DocumentChunk.document_id == document_id)
            .order_by(cosine_distance)
            .limit(limit)
        )

        results = session.execute(query).unique().all()
        return [cast(DocumentChunk, row[0]) for row in results]


@final
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


@final
class ChatCompletionPayload(BaseModel):
    messages: list[Message]
    document_id: str


async def _chat_completion_stream(
    payload: ChatCompletionPayload,
) -> AsyncGenerator[str, None]:
    query = payload.messages[-1].content

    query_embedding = get_embedding(query)
    chunks = get_chunk_similarity(payload.document_id, query_embedding)
    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[chunk.text for chunk in chunks],
        top_n=5,
    )
    reranked_indices = [r.index for r in rerank_response.results]
    sorted_results = [chunks[i] for i in reranked_indices]

    context_text = "\n\n".join(
        [f"情報{i+1}: {chunk.text}" for i, chunk in enumerate(sorted_results)]
    )
    system_message = ChatCompletionSystemMessageParam(
        role="system",
        content=(
            "以下はコンテキスト情報です。質問に回答する際、必ず以下の情報を参照してください。:\n\n"
            "コンテキスト情報以外は参照しないようにしてください。:\n\n"
            "\n\n"
            "以下コンテキスト: \n"
            f"{context_text}"
        ),
    )
    print(context_text)

    messages: list[ChatCompletionMessageParam] = [system_message]
    for message in payload.messages:
        if message.role == "assistant":
            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=message.content,
                )
            )
        if message.role == "user":
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=message.content,
                )
            )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


@app.post("/chat_completion")
async def _chat_completion(
    payload: ChatCompletionPayload,
) -> StreamingResponse:
    return StreamingResponse(_chat_completion_stream(payload), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
