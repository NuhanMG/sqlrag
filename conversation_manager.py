"""
Conversation State Management for Multi-Turn Analytics
Handles conversation history, data context, and visualization tracking
"""
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid
import pandas as pd


@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sql_query: Optional[str] = None
    dataframe_snapshot: Optional[Dict] = None  # Store metadata, not full DF
    visualization: Optional[str] = None  # Chart type
    figure_json: Optional[str] = None  # Store Plotly figure as JSON string
    metadata: Dict = field(default_factory=dict)


@dataclass
class DataContext:
    """Represents a dataset context in the conversation"""
    query: str
    columns: List[str]
    row_count: int
    sample_data: Dict
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VisualizationRecord:
    """Records a visualization created during conversation"""
    question: str
    chart_type: str
    data_summary: str
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationState:
    """Manages conversation state across multiple turns"""
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages: List[Message] = []
        self.data_contexts: List[DataContext] = []
        self.visualizations: List[VisualizationRecord] = []
        self.start_time = datetime.now()
    
    def add_message(self, message: Message):
        """Add a message to conversation history"""
        self.messages.append(message)
    
    def add_data_context(self, context: DataContext):
        """Add a data context (query result) to history"""
        self.data_contexts.append(context)
        # Keep only last N contexts to manage memory
        if len(self.data_contexts) > 10:
            self.data_contexts = self.data_contexts[-10:]
    
    def add_visualization(self, viz: VisualizationRecord):
        """Add a visualization record"""
        self.visualizations.append(viz)
    
    def get_recent_messages(self, n: int = 5) -> List[Message]:
        """Get last N messages"""
        return self.messages[-n:] if len(self.messages) >= n else self.messages
    
    def get_latest_dataframe(self) -> Optional[pd.DataFrame]:
        """Try to reconstruct latest DataFrame from context"""
        if not self.messages:
            return None
        
        # Look for most recent assistant message with dataframe snapshot
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.dataframe_snapshot:
                # For now, return None as we only store metadata
                # In full implementation, this could reconstruct from snapshot
                return None
        
        return None
    
    def clear_old_contexts(self, max_keep: int = 3):
        """Clear old data contexts to save memory"""
        if len(self.data_contexts) > max_keep:
            self.data_contexts = self.data_contexts[-max_keep:]
    
    def export_conversation(self) -> Dict:
        """Export conversation state to dict for persistence"""
        return {
            "conversation_id": self.conversation_id,
            "start_time": self.start_time.isoformat(),
            "message_count": len(self.messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "sql_query": msg.sql_query,
                    "has_dataframe": msg.dataframe_snapshot is not None,
                    "visualization": msg.visualization,
                    "figure_json": msg.figure_json,
                    "metadata": msg.metadata
                }
                for msg in self.messages
            ],
            "data_contexts": [
                {
                    "query": ctx.query,
                    "columns": ctx.columns,
                    "row_count": ctx.row_count,
                    "timestamp": ctx.timestamp.isoformat()
                }
                for ctx in self.data_contexts
            ],
            "visualizations": [
                {
                    "question": viz.question,
                    "chart_type": viz.chart_type,
                    "data_summary": viz.data_summary,
                    "timestamp": viz.timestamp.isoformat()
                }
                for viz in self.visualizations
            ]
        }
    
    def import_conversation(self, data: Dict):
        """Import conversation state from dict"""
        self.conversation_id = data.get("conversation_id", str(uuid.uuid4()))
        self.start_time = datetime.fromisoformat(data.get("start_time", datetime.now().isoformat()))
        
        # Import messages
        self.messages = [
            Message(
                role=msg["role"],
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                sql_query=msg.get("sql_query"),
                dataframe_snapshot=None,  # Cannot restore full dataframe
                visualization=msg.get("visualization"),
                figure_json=msg.get("figure_json"),
                metadata=msg.get("metadata", {})
            )
            for msg in data.get("messages", [])
        ]
        
        # Import data contexts
        self.data_contexts = [
            DataContext(
                query=ctx["query"],
                columns=ctx["columns"],
                row_count=ctx["row_count"],
                sample_data={},
                timestamp=datetime.fromisoformat(ctx["timestamp"])
            )
            for ctx in data.get("data_contexts", [])
        ]
        
        # Import visualizations
        self.visualizations = [
            VisualizationRecord(
                question=viz["question"],
                chart_type=viz["chart_type"],
                data_summary=viz["data_summary"],
                timestamp=datetime.fromisoformat(viz["timestamp"])
            )
            for viz in data.get("visualizations", [])
        ]
