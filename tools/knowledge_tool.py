from typing import Optional
import os
from langchain_core.tools import tool
from openai import OpenAI
from sqlalchemy import or_
import json

from database.connection import SessionLocal
from database.models import SemanticKnowledge

# Global embedder instance (lazy init)
_embedder_client = None
_model_name = None

def get_embedder_client():
    global _embedder_client, _model_name
    if _embedder_client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        _embedder_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        _model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    return _embedder_client, _model_name

def embed_text(text: str):
    client, model = get_embedder_client()
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise e

@tool
def search_semantic_knowledge(query: str, category: Optional[str] = None) -> str:
    """
    Search for 6G domain knowledge using vector similarity.
    Useful for finding slice profiles, QoS configs, or standard definitions when exact names are unknown.
    
    Args:
        query: The search term (e.g., "low latency slice", "industrial automation").
        category: Optional context category (e.g., "Slice_Profile", "QoS_Config").
        
    Returns:
        Top matching knowledge items with keys, descriptions and values.
    """
    session = SessionLocal()
    try:
        # Construct search text
        search_text = query
        if category:
            search_text = f"{category}: {query}"
            
        # Generate embedding
        query_vector = embed_text(search_text)
        
        # Vector search using L2 distance (adjust operator if needed based on DB)
        # Assuming pgvector is installed and <-> operator is mapped to l2_distance or similar
        items = session.query(SemanticKnowledge).order_by(
            SemanticKnowledge.embedding.l2_distance(query_vector)
        ).limit(3).all()
        
        if not items:
            return f"No relevant knowledge found for '{query}'."
            
        output = []
        for item in items:
            try:
                value_str = json.dumps(item.value, ensure_ascii=False)
            except:
                value_str = str(item.value)
                
            entry = (
                f"Key: {item.key}\n"
                f"Category: {item.category}\n"
                f"Description: {item.description}\n"
                f"Value: {value_str}\n"
            )
            output.append(entry)
            
        return "\n---\n".join(output)
        
    except Exception as e:
        return f"Error executing vector search: {str(e)}"
    finally:
        session.close()

@tool
def get_knowledge_by_key(key: str) -> str:
    """
    Retrieve specific 6G knowledge. Tries exact match first, then falls back to vector similarity search
    to handle typos or approximate keys (e.g. '5qi_urllec' -> '5qi_urllc').
    
    Args:
        key: The key of the knowledge item.
        
    Returns:
        The value associated with the key or best match.
    """
    session = SessionLocal()
    try:
        # 1. Try Exact Match
        item = session.query(SemanticKnowledge).filter(SemanticKnowledge.key == key).first()
        if item:
            return json.dumps(item.value, ensure_ascii=False)
            
        # 2. Fallback to Vector Search
        query_vector = embed_text(key)
        
        # Find closest match
        closest = session.query(SemanticKnowledge).order_by(
            SemanticKnowledge.embedding.l2_distance(query_vector)
        ).first()
        
        if closest:
            val_str = json.dumps(closest.value, ensure_ascii=False)
            return f"[Approximate Match: {closest.key}] {val_str}"
        else:
            return f"Knowledge item not found for key: {key}"
            
    except Exception as e:
        return f"Error retrieving key {key}: {str(e)}"
    finally:
        session.close()
