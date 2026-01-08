"""
A2A (Agent-to-Agent) Communication
Agent-to-agent communication protocol

2025/2026 capability: Multi-agent coordination

MIT-level engineering: Production-grade agent communication
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Agent-to-agent message."""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: str
    message_id: str


class A2AProtocol:
    """
    Agent-to-Agent communication protocol.
    
    Enables agents to communicate, coordinate, and collaborate.
    """
    
    def __init__(self):
        """Initialize A2A protocol."""
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.message_queue: List[AgentMessage] = []
        self.message_history: List[AgentMessage] = []
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        endpoint: Optional[str] = None
    ):
        """
        Register an agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (e.g., "planner", "executor", "verifier")
            capabilities: List of agent capabilities
            endpoint: Optional communication endpoint
        """
        self.agents[agent_id] = {
            "id": agent_id,
            "type": agent_type,
            "capabilities": capabilities,
            "endpoint": endpoint,
            "registered_at": datetime.now().isoformat()
        }
        logger.info(f"Registered agent: {agent_id} ({agent_type})")
    
    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any]
    ) -> str:
        """
        Send message from one agent to another.
        
        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            message_type: Type of message
            content: Message content
            
        Returns:
            Message ID
        """
        if sender_id not in self.agents:
            raise ValueError(f"Sender agent {sender_id} not registered")
        if receiver_id not in self.agents:
            raise ValueError(f"Receiver agent {receiver_id} not registered")
        
        message = AgentMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            message_id=f"{sender_id}_{receiver_id}_{datetime.now().timestamp()}"
        )
        
        self.message_queue.append(message)
        self.message_history.append(message)
        
        logger.info(f"Message sent: {sender_id} -> {receiver_id} ({message_type})")
        
        return message.message_id
    
    def receive_messages(self, agent_id: str) -> List[AgentMessage]:
        """
        Receive messages for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of messages for the agent
        """
        messages = [msg for msg in self.message_queue if msg.receiver_id == agent_id]
        # Remove from queue
        self.message_queue = [msg for msg in self.message_queue if msg.receiver_id != agent_id]
        return messages
    
    def broadcast(
        self,
        sender_id: str,
        message_type: str,
        content: Dict[str, Any],
        agent_types: Optional[List[str]] = None
    ):
        """
        Broadcast message to multiple agents.
        
        Args:
            sender_id: Sender agent ID
            message_type: Message type
            content: Message content
            agent_types: Optional filter by agent types
        """
        receivers = [
            agent_id for agent_id, agent in self.agents.items()
            if agent_id != sender_id and (
                agent_types is None or agent["type"] in agent_types
            )
        ]
        
        for receiver_id in receivers:
            self.send_message(sender_id, receiver_id, message_type, content)
    
    def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get capabilities of an agent."""
        if agent_id not in self.agents:
            return []
        return self.agents[agent_id]["capabilities"]
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents with a specific capability."""
        return [
            agent_id for agent_id, agent in self.agents.items()
            if capability in agent["capabilities"]
        ]

