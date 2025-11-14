import argparse, textwrap
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DB_DIR = "chroma_db"
COL_NAME = "docs"

# 논문이 영어이고 평가 섹션(Q1~Q4)이 가까운 청크로 많이 잡힘 > 모델이 그 패턴을 따라감
# def build_prompt(question, contexts, max_ctx_chars=2800):
#     joined = ""
#     for i, c in enumerate(contexts, 1):
#         block = f"[Doc {i}] {c}\n"
#         if len(joined) + len(block) > max_ctx_chars:
#             break
#         joined += block
#     return (
#         "Use the following context to answer the question. "
#         "If unknown from context, say you don't know.\n\n"
#         f"{joined}\nQuestion: {question}\nAnswer:"
#     )

# 성능을 올리기 위한 시도.
def build_prompt(question, contexts, max_ctx_chars=3200):
    joined = ""
    for i, c in enumerate(contexts, 1):
        block = f"[문맥 {i}] {c}\n"
        if len(joined) + len(block) > max_ctx_chars:
            break
        joined += block
    instr = (
        "당신은 엄격하게 근거에만 기반해 답하는 어시스턴트입니다.\n"
        "아래 문맥에서 답을 찾고, 문맥에 없으면 '문맥에 없음'이라고만 말하세요.\n"
        "최대 3문장, 한국어로 간결하게.\n\n"
        f"{joined}\n질문: {question}\n정답:"
    )
    return instr



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    question = " ".join(args.question)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection(COL_NAME)

    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    res = col.query(query_embeddings=[q_emb], n_results=args.k)

    contexts = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []

    prompt = build_prompt(question, contexts)

    model_name = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3072)
    output = model.generate(**inputs, max_new_tokens=256)
    answer = tok.decode(output[0], skip_special_tokens=True)

    print("\n=== Answer ===")
    print(textwrap.fill(answer, 100))
    print("\n=== Top Contexts ===")
    for i, (ctx, m) in enumerate(zip(contexts, metas), 1):
        src = m.get("source"); pg = m.get("page")
        print(f"[{i}] {src} (p.{pg})")
        print(textwrap.shorten(ctx, width=180, placeholder=" ..."))
        print("-"*80)

if __name__ == "__main__":
    main()