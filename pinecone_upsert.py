import os
from sentence_transformers import SentenceTransformer
from chunking import get_chunks
from pinecone import Pinecone, ServerlessSpec

def main():
    # Prepare data
    chunks_by_type = get_chunks()
    chunks = chunks_by_type["all"]

    texts = [c["content"] for c in chunks]
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(texts, convert_to_tensor=False)

    # Pinecone config (v5 SDK)
    api_key = os.getenv("PINECONE_API_KEY", "pcsk_3ypWCf_3SQ8dW2aZbAvQvNnEtH2RMUikTbV3xPSMEHMb3WzhXxkaevKf56EKzotDEKDcuD")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    pc = Pinecone(api_key=api_key)

    # Create or connect to index
    index_name = "placement-companion"
    vector_dim = len(embeddings[0]) if len(embeddings) > 0 else 768

    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=vector_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    index = pc.Index(index_name)

    # Upsert chunks with IDs and metadata
    vectors = [
        (
            f"id-{i}",
            embeddings[i].tolist(),
            {
                "section": chunks[i]["section"],
                "content": chunks[i]["content"]
            }
        )
        for i in range(len(chunks))
    ]
    index.upsert(vectors)

    print(f"Upserted {len(vectors)} vectors to index '{index_name}' (dim={vector_dim}).")

    # Example query
    query = "What is the cgpa requirement to attend this drive?"
    query_emb = model.encode([query])[0].tolist()
    result = index.query(vector=query_emb, top_k=1, include_metadata=True)

    print("\nQuery Results:")
    for match in result['matches']:
        section = match['metadata'].get('section', 'Unknown')
        content = match['metadata'].get('content', '[No content]')
        print(f"[{section}] {content} (score={match['score']:.4f})")


if __name__ == "__main__":
    main()
