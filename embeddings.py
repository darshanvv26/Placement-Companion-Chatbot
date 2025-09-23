from sentence_transformers import SentenceTransformer
from chunking import get_chunks

def main():
    # Load chunks produced by chunking.py
    chunks_by_type = get_chunks()
    chunks = chunks_by_type["all"]

    # Only embed the content
    texts = [c["content"] for c in chunks]

    # Create embeddings
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(texts, convert_to_tensor=False)

    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    # print("Sample chunk:", chunks[0])
    print("Sample embedding vector:", embeddings[0][:10])


if __name__ == "__main__":
    main()
