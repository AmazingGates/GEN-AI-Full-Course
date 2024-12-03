# This is an exampe of pinecone usecse directly from the website.

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="c1489634-2122-4dc3-a641-e16b3399b247")

pc.create_index(
    name="sample-movies",
    dimension=1536, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)