from __future__ import annotations

import pickle
import re
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from const import (
    DATABASE_URL,
    AZURE_DOCUMENT_INTELLIGENCE_KEY,
    AZURE_DOCUMENT_INTELLIGENCE_URL,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
)
from model.document import Document, DocumentChunk

assert AZURE_DOCUMENT_INTELLIGENCE_KEY, "AZURE_DOCUMENT_INTELLIGENCE_KEY is not set"
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOCUMENT_INTELLIGENCE_URL,
    credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY),
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def create_document_table_df() -> pd.DataFrame:
    columns = ["id", "name"]
    return pd.DataFrame(columns=columns)


def create_chunk_table_df() -> pd.DataFrame:
    columns = ["id", "document_id", "index", "text", "embedding"]
    return pd.DataFrame(columns=columns)


def append_to_document_table(
    df: pd.DataFrame, _id: str, file_path: Path
) -> pd.DataFrame:
    new_data = {
        "id": [_id],
        "name": [file_path.name],
    }
    return pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)


def append_to_chunk_table(
    df: pd.DataFrame, _id: str, document_id: str, index: int, text: str
) -> pd.DataFrame:
    new_data = {
        "id": [_id],
        "document_id": [document_id],
        "index": [index],
        "text": [text],
    }
    return pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)


def exec_azure_document_intelligence(pdf_path: Path, cache_path: Path) -> str:
    if cache_pdf_text_path.exists():
        print("✅ キャッシュからpdfの情報を読み込みました。")
        return cache_pdf_text_path.read_text(encoding="utf-8")

    with open(pdf_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", f, output_content_format="markdown"
        )
    result = poller.result()

    markdown_text = result.content
    with open(cache_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_text)

    return markdown_text


def preprocess_text(text: str) -> str:
    preprocessed = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    preprocessed = "\n".join(
        line for line in preprocessed.splitlines() if line.strip() != "M"
    )
    preprocessed = re.sub(r"\n\s*\n+", "\n\n", preprocessed)
    return preprocessed.strip()


def split_into_sections(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^#\s+.*$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    matches = list(pattern.finditer(text))

    if matches:
        first_match = matches[0]
        if first_match.start() > 0:
            intro = text[: first_match.start()].strip()
            if intro:
                sections.append(("Introduction", intro))
        for i, match in enumerate(matches):
            heading = match.group(0).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            sections.append((heading, section_text))
    else:
        sections.append(("Whole Document", text.strip()))
    return sections


def split_section_into_chunks(
    section_text: str, chunk_size: int = 300, overlap: int = 50
) -> list[str]:
    words = section_text.split()
    chunks = []
    if len(words) <= chunk_size:
        chunks.append(section_text)
    else:
        start = 0
        while start < len(words):
            end = start + chunk_size
            this_chunk = " ".join(words[start:end])
            chunks.append(this_chunk)
            # 次のチャンクは、overlap分戻して開始
            start = end - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[text[:4000]]
    )
    return response.data[0].embedding


def append_to_chunk_table_embedding(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached_df: pd.DataFrame = pickle.load(f)
        print("✅ キャッシュからEmbeddingデータを読み込みました。")
        return cached_df

    print("⏳ Embeddingを取得中...")
    embeddings: list[list[float]] = []
    for index, (_, row) in enumerate(df.iterrows()):
        input_text = str(row["text"])

        if not input_text.strip():
            embeddings.append([])
            continue

        embedding = get_embedding(input_text)
        embeddings.append(embedding)
        print(f"Processing {index + 1}/{len(df)}")

    df["embedding"] = embeddings

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print("✅ Embedding結果をキャッシュに保存しました。")

    return df


def insert_data_to_db(document_df: pd.DataFrame, chunk_df: pd.DataFrame) -> None:
    def remove_null_bytes(v: str) -> str:
        return v.replace("\x00", "")

    try:
        with SessionLocal() as session:
            for _, row in document_df.iterrows():
                document = Document(
                    id=uuid.UUID(str(row["id"])),
                    name=str(row["name"]),
                    created_at=datetime.now(),
                )
                session.add(document)
                session.flush()

            for _, row in chunk_df.iterrows():
                chunk = DocumentChunk(
                    id=str(row["id"]),
                    document_id=uuid.UUID(str(row["document_id"])),
                    index=int(row["index"]),
                    text=remove_null_bytes(str(row["text"])),
                    embedding=row["embedding"],
                )
                session.add(chunk)
                session.flush()

            session.commit()
    except Exception as e:
        print(f"❌ エラー発生: {e}")

    print("✅ データベースへのインサートが完了しました。")


if __name__ == "__main__":
    input_dir = Path("./input")
    output_dir = Path("./output")
    cache_dir = Path("./cache")

    assert input_dir.exists(), f"{input_dir} does not exist"
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    document_table_cache_path = cache_dir / "document_table.pkl"
    chunk_table_cache_path = cache_dir / "chunk_table.pkl"
    chunk_table_with_embedding_cache_path = cache_dir / "chunk_table_with_embedding.pkl"

    if document_table_cache_path.exists() and chunk_table_cache_path.exists():
        with open(document_table_cache_path, "rb") as f:
            document_table_df = pickle.load(f)
            print(
                f"✅ データをキャッシュから読み込みました: {document_table_cache_path.name}"
            )
        with open(chunk_table_cache_path, "rb") as f:
            chunk_table_df = pickle.load(f)
            print(
                f"✅ データをキャッシュから読み込みました: {chunk_table_cache_path.name}"
            )
    else:
        document_table_df = create_document_table_df()
        chunk_table_df = create_chunk_table_df()
        pdf_files = list(input_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            pdf_file: Path = pdf_file
            file_id = str(uuid.uuid4())
            document_table_df = append_to_document_table(
                document_table_df, file_id, pdf_file
            )

            cache_pdf_text_path = cache_dir / f"{pdf_file.name}_text.md"
            org_text = exec_azure_document_intelligence(pdf_file, cache_pdf_text_path)

            preprocessed_text = preprocess_text(org_text)
            all_sections = split_into_sections(preprocessed_text)
            all_chunks: list[str] = []
            for section_idx, (heading, text) in enumerate(all_sections, start=1):
                chunks = split_section_into_chunks(text, chunk_size=300, overlap=50)
                print(f"✅ {heading} は {len(chunks)} 個のチャンクに分割されました。")
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    metadata = f"<!-- Section: {heading} | Chunk: {chunk_idx}/{len(chunks)} | Section Index: {section_idx} -->\n"
                    all_chunks.append(metadata + chunk)

            output_dir = output_dir / pdf_file.name
            output_dir.mkdir(exist_ok=True)
            for chunk_idx, chunk_text in enumerate(all_chunks):
                output_path = output_dir / f"${chunk_idx}_chunks.md"
                with open(output_path, "w", encoding="utf-8") as f:  # type: ignore
                    f.write(chunk_text)  # type: ignore

                chunk_table_df = append_to_chunk_table(
                    chunk_table_df, str(uuid.uuid4()), file_id, chunk_idx, chunk_text
                )

        with open(document_table_cache_path, "wb") as f:  # type: ignore
            pickle.dump(document_table_df, f)
            print(
                f"✅ データをキャッシュに保存しました: {document_table_cache_path.name}"
            )
        with open(chunk_table_cache_path, "wb") as f:  # type: ignore
            pickle.dump(chunk_table_df, f)
            print(f"✅ データをキャッシュに保存しました: {chunk_table_cache_path.name}")

    if chunk_table_with_embedding_cache_path.exists():
        with open(chunk_table_with_embedding_cache_path, "rb") as f:
            chunk_table_df = pickle.load(f)
            print(
                f"✅ データをキャッシュから読み込みました: {chunk_table_with_embedding_cache_path.name}"
            )
    else:
        chunk_table_df = append_to_chunk_table_embedding(
            chunk_table_df, chunk_table_with_embedding_cache_path
        )

    document_table_df.to_csv(output_dir / "document_table.csv", index=False)
    chunk_table_df.to_csv(output_dir / "chunk_table.csv", index=False)

    insert_data_to_db(document_table_df, chunk_table_df)
