from fastapi import FastAPI
from typing import Optional
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# 加载预训练的RAG模型、tokenizer和retriever
tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nli')
retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nli', index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-nli')

app = FastAPI()

@app.post("/combine")
def retrieve_and_combine_rag(query: str, context: str):
    """
    将用户的提问与相关信息结合，通过RAG模型进行检索，生成一个融合后的字符串。

    参数:
    query (str): 用户的提问字符串。
    context (str): 信息文本字符串。

    返回:
    str: 融合后的字符串。
    """
    # Tokenize the input query and context
    inputs = tokenizer(query, return_tensors="pt")
    
    # Retrieve relevant passages using the context
    retrieved_docs = retriever(input_ids=inputs["input_ids"], context=context, return_tensors="pt")

    # Decode the retrieved documents
    retrieved_texts = tokenizer.batch_decode(retrieved_docs["input_ids"], skip_special_tokens=True)

    # Combine the query with the retrieved texts
    combined_input = f"问题: {query}\n相关信息: {' '.join(retrieved_texts)}\n请基于以上信息回答问题。"

    return {"combined_text": combined_input}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)