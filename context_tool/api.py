"""
Context Tool API
Explicit context management for long-horizon tasks

Reference: "Context as a Tool" - Managing context explicitly in agent loops

MIT-level engineering: Production-grade context management
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """Represents a chunk of context."""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    importance: float  # 0.0-1.0
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ContextTool:
    """
    Context Tool API for long-horizon context management.
    
    Provides:
    - Context summarization
    - Context pinning
    - Context retrieval
    - Context compression
    """
    
    def __init__(self, max_context_length: int = 32768, compression_ratio: float = 0.5):
        """
        Initialize Context Tool.
        
        Args:
            max_context_length: Maximum context length
            compression_ratio: Target compression ratio (0.0-1.0)
        """
        self.max_context_length = max_context_length
        self.compression_ratio = compression_ratio
        
        # Context storage
        self.pinned_contexts: Dict[str, ContextChunk] = {}
        self.context_history: List[ContextChunk] = []
        self.compressed_contexts: Dict[str, str] = {}
    
    def summarize(self, context: str, max_length: Optional[int] = None) -> str:
        """
        Summarize context to reduce length.
        
        Args:
            context: Context to summarize
            max_length: Maximum length for summary (uses compression_ratio if None)
            
        Returns:
            Summarized context
        """
        if max_length is None:
            max_length = int(len(context) * self.compression_ratio)
        
        if len(context) <= max_length:
            return context
        
        # Simple summarization (in production, use LLM-based summarization)
        # For now, take first and last portions
        chunk_size = max_length // 2
        summary = context[:chunk_size] + "\n\n[... truncated ...]\n\n" + context[-chunk_size:]
        
        logger.info(f"Summarized context: {len(context)} -> {len(summary)} chars")
        return summary
    
    def pin(self, context: str, metadata: Optional[Dict[str, Any]] = None, 
            importance: float = 0.5, tags: Optional[List[str]] = None) -> str:
        """
        Pin context for later retrieval.
        
        Args:
            context: Context to pin
            metadata: Optional metadata
            importance: Importance score (0.0-1.0)
            tags: Optional tags for retrieval
            
        Returns:
            Pinned context ID
        """
        # Generate ID
        context_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
        context_id = f"pinned_{context_hash}"
        
        chunk = ContextChunk(
            id=context_id,
            content=context,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            importance=importance,
            tags=tags or []
        )
        
        self.pinned_contexts[context_id] = chunk
        logger.info(f"Pinned context: {context_id} (importance={importance:.2f})")
        
        return context_id
    
    def retrieve(self, query: str, top_k: int = 5) -> List[ContextChunk]:
        """
        Retrieve relevant contexts based on query.
        
        Args:
            query: Search query
            top_k: Number of contexts to retrieve
            
        Returns:
            List of relevant context chunks
        """
        # Simple retrieval (in production, use semantic search/embeddings)
        results = []
        
        # Search pinned contexts
        for chunk in self.pinned_contexts.values():
            score = self._compute_relevance_score(query, chunk)
            results.append((score, chunk))
        
        # Search history
        for chunk in self.context_history[-100:]:  # Last 100 chunks
            score = self._compute_relevance_score(query, chunk)
            results.append((score, chunk))
        
        # Sort by relevance and importance
        results.sort(key=lambda x: x[0] * x[1].importance, reverse=True)
        
        return [chunk for _, chunk in results[:top_k]]
    
    def compress(self, contexts: List[str], target_length: Optional[int] = None) -> str:
        """
        Compress multiple contexts into single context.
        
        Args:
            contexts: List of contexts to compress
            target_length: Target length (uses max_context_length if None)
            
        Returns:
            Compressed context
        """
        if target_length is None:
            target_length = self.max_context_length
        
        # Combine contexts
        combined = "\n\n---\n\n".join(contexts)
        
        # Compress if needed
        if len(combined) > target_length:
            combined = self.summarize(combined, max_length=target_length)
        
        logger.info(f"Compressed {len(contexts)} contexts to {len(combined)} chars")
        return combined
    
    def add_to_history(self, context: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add context to history.
        
        Args:
            context: Context to add
            metadata: Optional metadata
        """
        context_id = f"hist_{len(self.context_history)}"
        chunk = ContextChunk(
            id=context_id,
            content=context,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            importance=0.5,
            tags=[]
        )
        
        self.context_history.append(chunk)
        
        # Limit history size
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-1000:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get summary of current context state.
        
        Returns:
            Summary dictionary
        """
        return {
            "pinned_count": len(self.pinned_contexts),
            "history_count": len(self.context_history),
            "total_pinned_length": sum(len(c.content) for c in self.pinned_contexts.values()),
            "total_history_length": sum(len(c.content) for c in self.context_history)
        }
    
    def _compute_relevance_score(self, query: str, chunk: ContextChunk) -> float:
        """Compute relevance score between query and chunk."""
        # Simple keyword matching (in production, use embeddings)
        query_lower = query.lower()
        content_lower = chunk.content.lower()
        
        # Count matching words
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        matches = len(query_words & content_words)
        total = len(query_words)
        
        score = matches / total if total > 0 else 0.0
        
        # Boost if tags match
        for tag in chunk.tags:
            if tag.lower() in query_lower:
                score += 0.2
        
        return min(1.0, score)
    
    def clear_pinned(self, context_id: Optional[str] = None):
        """
        Clear pinned contexts.
        
        Args:
            context_id: Specific context ID to clear (clears all if None)
        """
        if context_id:
            if context_id in self.pinned_contexts:
                del self.pinned_contexts[context_id]
                logger.info(f"Cleared pinned context: {context_id}")
        else:
            self.pinned_contexts.clear()
            logger.info("Cleared all pinned contexts")
    
    def export_context(self) -> Dict[str, Any]:
        """Export all context for persistence."""
        return {
            "pinned": {cid: chunk.to_dict() for cid, chunk in self.pinned_contexts.items()},
            "history": [chunk.to_dict() for chunk in self.context_history[-100:]],  # Last 100
            "metadata": {
                "max_context_length": self.max_context_length,
                "compression_ratio": self.compression_ratio
            }
        }
    
    def import_context(self, data: Dict[str, Any]):
        """Import context from exported data."""
        # Import pinned
        for cid, chunk_data in data.get("pinned", {}).items():
            chunk = ContextChunk(**chunk_data)
            self.pinned_contexts[cid] = chunk
        
        # Import history
        for chunk_data in data.get("history", []):
            chunk = ContextChunk(**chunk_data)
            self.context_history.append(chunk)
        
        logger.info(f"Imported {len(self.pinned_contexts)} pinned and {len(self.context_history)} history contexts")


class ContextToolAPI:
    """
    High-level API for context management in agent loops.
    """
    
    def __init__(self, context_tool: Optional[ContextTool] = None):
        """
        Initialize Context Tool API.
        
        Args:
            context_tool: Optional ContextTool instance (creates new if None)
        """
        self.context_tool = context_tool or ContextTool()
    
    def call(self, action: str, **kwargs) -> Any:
        """
        Call context tool action.
        
        Supported actions:
        - summarize: Summarize context
        - pin: Pin context
        - retrieve: Retrieve contexts
        - compress: Compress contexts
        - add_to_history: Add to history
        - get_summary: Get context summary
        - clear_pinned: Clear pinned contexts
        
        Args:
            action: Action name
            **kwargs: Action arguments
            
        Returns:
            Action result
        """
        if action == "summarize":
            return self.context_tool.summarize(
                kwargs.get("context", ""),
                max_length=kwargs.get("max_length")
            )
        elif action == "pin":
            return self.context_tool.pin(
                kwargs.get("context", ""),
                metadata=kwargs.get("metadata"),
                importance=kwargs.get("importance", 0.5),
                tags=kwargs.get("tags")
            )
        elif action == "retrieve":
            chunks = self.context_tool.retrieve(
                kwargs.get("query", ""),
                top_k=kwargs.get("top_k", 5)
            )
            return [chunk.to_dict() for chunk in chunks]
        elif action == "compress":
            return self.context_tool.compress(
                kwargs.get("contexts", []),
                target_length=kwargs.get("target_length")
            )
        elif action == "add_to_history":
            self.context_tool.add_to_history(
                kwargs.get("context", ""),
                metadata=kwargs.get("metadata")
            )
            return {"status": "added"}
        elif action == "get_summary":
            return self.context_tool.get_context_summary()
        elif action == "clear_pinned":
            self.context_tool.clear_pinned(kwargs.get("context_id"))
            return {"status": "cleared"}
        else:
            raise ValueError(f"Unknown action: {action}")

