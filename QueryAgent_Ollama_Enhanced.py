import os
import re
import sqlite3
import logging
from typing import Any, Dict, List, Optional, Tuple
import json
import requests
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dask integration for large datasets
from dataframe_factory import (
    from_sql, 
    ensure_pandas, 
    DataFrameFactory, 
    UnifiedDataFrame,
    get_backend_name
)

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_community.tools import QuerySQLDatabaseTool as _QuerySQLTool
import chromadb
from chromadb.config import Settings 

from conversation_manager import ConversationState, Message, DataContext, VisualizationRecord

load_dotenv()

# Configure logging to output info level logs with timestamp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuestionIntent(str, Enum):
    """Types of question intents"""
    NEW_QUERY = "new_query"
    RE_VISUALIZE = "re_visualize"
    TRANSFORM = "transform"
    COMBINE = "combine"
    COMPARE = "compare"
    CLARIFY = "clarify"


class VisualizationResponse(BaseModel):
    """Visualization recommendation model"""
    should_visualize: bool = Field(description="Whether to create visualization")
    chart_types: List[str] = Field(default_factory=list, description="Recommended chart types")
    primary_chart: str = Field(description="Primary recommended chart type")
    x_axis: Optional[str] = Field(default=None, description="Recommended X-axis column")
    y_axis: Optional[str] = Field(default=None, description="Recommended Y-axis column")
    color_by: Optional[str] = Field(default=None, description="Column to color/group by")
    title: str = Field(default="", description="Chart title")
    visualization_rationale: str = Field(description="Why this visualization is recommended")


class IntentAnalysis(BaseModel):
    """Intent analysis model"""
    intent: str = Field(description="Detected intent: new_query, re_visualize, transform, combine, compare, clarify")
    references_previous: bool = Field(description="Whether question references previous results")
    referenced_concepts: List[str] = Field(default_factory=list, description="Concepts referenced from history")
    needs_context: bool = Field(description="Whether context from history is needed")
    confidence: float = Field(description="Confidence score 0-1")


class QueryAgentEnhanced:
    """Enhanced Query agent with multi-turn conversation support"""

    def __init__(
        self,
        source_db_path: str,
        vector_db_path: str = "./chroma_db_768dim",
        llm_model: str = "qwen2.5:7b",
        conversation_state: Optional[ConversationState] = None,
        max_context_messages: int = 10,
        max_data_contexts: int = 20,
        temperature: float = 0.0,
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize the QueryAgentEnhanced.

        Args:
            source_db_path: Path to the SQLite database file.
            vector_db_path: Path to the ChromaDB vector database directory.
            llm_model: Name of the Ollama model to use for generation.
            conversation_state: Optional existing conversation state.
            max_context_messages: Maximum number of previous messages to include in context.
            max_data_contexts: Maximum number of data contexts to keep.
            temperature: Temperature setting for the LLM.
            ollama_base_url: URL of the Ollama API.
            embedding_model: Name of the embedding model to use.
        """
        
        
        self.source_db_path = source_db_path
        self.vector_db_path = vector_db_path
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        
        # Conversation management
        self.conversation_state = conversation_state or ConversationState()
        self.max_context_messages = 10  
        self.max_data_contexts = max_data_contexts
        
        # Performance caches
        self._metadata_cache = {}  # Metadata retrieval cache
        self._embedding_cache = {}  # Embedding cache (semantic)
        self._sql_validation_cache = {}  # SQL validation cache
        self._ollama_connection_verified = False
        
        # Parallel processing setup
        self._executor = ThreadPoolExecutor(max_workers=3)
        logger.info("✅ Parallel processing enabled (3 workers)")
        
        # Initialize connections
        self.conn = sqlite3.connect(source_db_path, check_same_thread=False)
        
        # Create SQLAlchemy engine for LangChain SQLDatabase
        # Use sample_rows_in_table_info=0 to prevent datetime parsing errors
        # when SQLite has DATE columns with space-separated datetime values
        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{source_db_path}")
        self.db = SQLDatabase(engine=engine, sample_rows_in_table_info=0)
        
        # Initialize ChromaDB and vector database
        self._setup_vector_database()
        
        # Load metadata from vector database
        self.table_metadata = self._load_table_metadata()
        self.column_metadata = self._load_column_metadata()
        
        # Initialize LLM
        try:
            self.llm = ChatOllama(
                model=llm_model,
                base_url=ollama_base_url,
                temperature=temperature,
                num_ctx=4096,
                num_predict=1024, # (sql + answer)
                repeat_penalty=1.1,
                timeout=120
            )
            logger.info(f"Successfully initialized Ollama with model: {llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.error("Please ensure Ollama is installed and running with: ollama serve")
            raise
        
        # Create SQL query chain
        self.query_chain = create_sql_query_chain(self.llm, self.db)
        
        # Initialize parsers
        self.viz_parser = PydanticOutputParser(pydantic_object=VisualizationResponse)
        
        logger.info(f"QueryAgentEnhanced initialized with model: {llm_model}")
        logger.info(f"Using embedding model: {self.embedding_model}")
        logger.info(f"Loaded metadata for {len(self.table_metadata)} table(s)")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        try:
            import requests
            logger.info(f"🔍 Requesting embedding from Ollama using model: {self.embedding_model}")
            
            payload = {"model": self.embedding_model, "prompt": text}
            logger.info(f"📤 Payload: model={self.embedding_model}")
            
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result["embedding"]
            
            logger.info(f"📥 Received embedding with dimension: {len(embedding)} (expected 768 for nomic-embed-text)")
            
            # Verify dimension matches expectation
            expected_dims = {"mxbai-embed-large": 1024, "nomic-embed-text": 768, "all-minilm": 384}
            expected_dim = expected_dims.get(self.embedding_model, 768)
            
            if len(embedding) != expected_dim:
                logger.warning(f"⚠️ DIMENSION MISMATCH: Expected {expected_dim} for '{self.embedding_model}', got {len(embedding)}")
                logger.warning("⚠️ This suggests Ollama may be using a different model than requested!")
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding with model {self.embedding_model}: {e}")
            return [0.0] * 768  # nomic-embed-text fallback - zero vector

    def _setup_vector_database(self):
        """Initialize vector database connection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Check if vector database exists
            if not os.path.exists(self.vector_db_path):
                logger.warning(f"Vector database not found: {self.vector_db_path}")
                logger.warning("Please run analyze_existing_db.py first to create the vector database")
                self.chroma_client = None
                self.table_collection = None
                self.column_collection = None
                return
            
            # Connect to ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.vector_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get existing collections
            self.table_collection = self.chroma_client.get_collection(name="table_metadata")
            self.column_collection = self.chroma_client.get_collection(name="column_metadata")
            
            logger.info(f"✅ Connected to vector database at: {self.vector_db_path}")
            logger.info(f"Table collection: {self.table_collection.count()} items")
            logger.info(f"Column collection: {self.column_collection.count()} items")
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {e}")
            self.chroma_client = None
            self.table_collection = None
            self.column_collection = None

    def _load_table_metadata(self) -> Dict:
        """Load table metadata from vector database"""
        if not hasattr(self, 'table_collection') or not self.table_collection:
            logger.warning("Table collection not available")
            return {}
        
        try:
            results = self.table_collection.get()
            
            metadata = {}
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    table_name = meta.get('table_name')
                    if table_name:
                        metadata[table_name] = meta
            
            logger.info(f"Loaded metadata for {len(metadata)} table(s) from vector database")
            return metadata
        except Exception as e:
            logger.error(f"Error loading table metadata from vector database: {e}")
            return {}
    
    def _load_column_metadata(self) -> Dict:
        """Load column metadata from vector database"""
        if not hasattr(self, 'column_collection') or not self.column_collection:
            logger.warning("Column collection not available")
            return {}
        
        try:
            # Retrieve all items from column collection
            results = self.column_collection.get()
            
            metadata = {}
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    table_name = meta.get('table_name')
                    column_name = meta.get('column_name')
                    if table_name and column_name:
                        if table_name not in metadata:
                            metadata[table_name] = {}
                        metadata[table_name][column_name] = meta
            
            total_columns = sum(len(cols) for cols in metadata.values())
            logger.info(f"Loaded metadata for {total_columns} column(s) from vector database")
            return metadata
        except Exception as e:
            logger.error(f"Error loading column metadata from vector database: {e}")
            return {}
    
    def search_tables(self, query: str, n_results: int = 5) -> Dict:
        """Semantic search for tables"""
        if not hasattr(self, 'table_collection') or not self.table_collection:
            return {"error": "Vector database not available"}
        
        try:
            # Generate embedding for the query
            query_embedding = self._get_embedding(query)
            # Query the table collection using the embedding
            results = self.table_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error searching tables: {e}")
            return {"error": str(e)}
    
    def search_columns(self, query: str, n_results: int = 10) -> Dict:
        """Semantic search for columns"""
        if not hasattr(self, 'column_collection') or not self.column_collection:
            return {"error": "Vector database not available"}
        
        try:
            # Generate embedding for the query
            query_embedding = self._get_embedding(query)
            # Query the column collection using the embedding
            results = self.column_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error searching columns: {e}")
            return {"error": str(e)}

    def test_ollama_connection(self) -> bool:
        """Test if Ollama is running and accessible (cached for performance)"""
        
        # Return cached result if already verified
        if self._ollama_connection_verified:
            return True
        
        try:
            import requests
            # Check if Ollama API is reachable
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # Verify if the requested model is available
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if self.llm_model in model_names or any(self.llm_model in name for name in model_names):
                    logger.info(f"✅ Ollama is running and model '{self.llm_model}' is available")
                    self._ollama_connection_verified = True  # Cache the result
                    return True
                else:
                    logger.warning(f"Model '{self.llm_model}' not found. Available: {model_names}")
                    return False
            else:
                logger.error(f"Ollama responded with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            logger.error("Please ensure Ollama is running with: ollama serve")
            return False

    def _execute_parallel(self, tasks: List[callable]) -> List[Any]:
        """
        Execute multiple tasks in parallel using ThreadPoolExecutor.
        
        Args:
            tasks: List of callable functions to execute
            
        Returns:
            List of results in same order as tasks
        """
        try:
            futures = [self._executor.submit(task) for task in tasks]
            results = [future.result() for future in futures]
            return results
        except Exception as e:
            logger.error(f"Parallel execution error: {e}")
            # Fallback to sequential execution
            return [task() for task in tasks]
    
    def _parse_with_fallback(self, response_text: str, parser, fallback_data: dict = None):
        """Parse LLM response with fallback mechanism for better JSON handling"""
        try:
            return parser.parse(response_text)
        except Exception as e:
            logger.warning(f"Pydantic parsing failed: {e}")
            
            try:
                import json
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response_text.strip()
                
                parsed_json = json.loads(json_str)
                return parser.pydantic_object(**parsed_json)
            except Exception as json_error:
                logger.warning(f"JSON parsing also failed: {json_error}")
            
            if fallback_data:
                logger.info("Using fallback data")
                return fallback_data
            
            return {"result": response_text, "success": False, "parsing_error": str(e)}

    def _analyze_question_intent(self, question: str) -> IntentAnalysis:
        """Analyze the intent of the user's question using heuristics to save LLM calls"""
        
        q_lower = question.lower()
        
        # Check for references to previous context
        reference_keywords = ["that", "this", "these", "those", "previous", "last", "earlier", "above", "it", "them"]
        references_previous = any(word in q_lower for word in reference_keywords)
        
        # Default intent
        intent = "new_query"
        
        # Heuristic intent detection based on keywords
        if any(w in q_lower for w in ["show again", "visualize", "again with", "chart", "graph", "plot", "draw", "re draw"]):
            intent = "re_visualize"
        elif any(w in q_lower for w in ["calculate", "add", "filter", "sort", "group by", "arrange", "order by"]):
            intent = "transform"
        elif any(w in q_lower for w in ["compare", "difference", "vs", "versus", "diff"]):
            intent = "compare"
        elif any(w in q_lower for w in ["explain", "mean", "why", "clarify"]):
            intent = "clarify"
        elif any(w in q_lower for w in ["combine", "join", "merge", "union"]):
            intent = "combine"
            
        logger.info(f"Heuristic intent detection: {intent} (references_previous={references_previous})")

        return IntentAnalysis(
            intent=intent,
            references_previous=references_previous,
            referenced_concepts=[], # Optional for heuristics
            needs_context=references_previous,
            confidence=0.8
        )

    def _build_context_prompt(self, question: str, intent: IntentAnalysis) -> str:
        """Build context-aware prompt including relevant conversation history and dashboard hints"""
        
        base_context = []
        
        # Add recent successful queries
        if intent.needs_context or intent.references_previous:
            recent_messages = self.conversation_state.get_recent_messages(self.max_context_messages)
            for msg in recent_messages:
                if msg.role == "assistant" and msg.sql_query:
                    base_context.append(f"Previous query: {msg.sql_query}")
                    if msg.dataframe_snapshot:
                        columns = ", ".join(msg.dataframe_snapshot.get("columns", []))
                        base_context.append(f"  Returned columns: {columns}")
        
        # Add data context information
        if self.conversation_state.data_contexts:
            latest_context = self.conversation_state.data_contexts[-1]
            base_context.append("\nCurrent data context:")
            base_context.append(f"  Columns available: {', '.join(latest_context.columns)}")
            base_context.append(f"  Row count: {latest_context.row_count}")
        
        # Build context string
        context_str = "\n".join(base_context) if base_context else "No previous context"
        
        # 🎯 Smart Prompt Engineering: Dashboard-friendly query hints
        viz_keywords = ['show', 'visualize', 'chart', 'graph', 'plot', 'compare', 'trend', 'dashboard', 'display']
        is_viz_request = any(kw in question.lower() for kw in viz_keywords)
        
        dashboard_hints = ""
        if is_viz_request:
            dashboard_hints = """

🎯 IMPORTANT - Dashboard Visualization Guidelines:
- For charts/visualizations, use aggregation (GROUP BY, SUM, COUNT, AVG) to produce summarized data
- Avoid SELECT * - always select specific columns needed for visualization
- Use LIMIT if returning raw records (max 10000 rows for charts)
- Prefer aggregating by: region, category, time_period, department, or other meaningful dimensions
- Good dashboard queries return 5-100 rows of aggregated/summarized data"""
        
        return f"""Context from conversation:
{context_str}
{dashboard_hints}

Current question with intent [{intent.intent}]:
{question}

Generate appropriate SQL query considering the conversation context."""

    def _check_previous_reference(self, question: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Check if question references previous data and return it if available"""
        
        reference_keywords = [
            "that", "this", "these", "those", "previous", "last", 
            "earlier", "above", "same", "it", "them"
        ]
        
        question_lower = question.lower()
        references_previous = any(keyword in question_lower for keyword in reference_keywords)
        
        if references_previous:
            # Try to get the latest dataframe
            latest_df = self.conversation_state.get_latest_dataframe()
            if latest_df is not None:
                logger.info("Found reference to previous data, reusing DataFrame")
                return True, latest_df
        
        return False, None

    def _get_table_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Load table schema information from SQLite once and cache it"""
        if hasattr(self, "_table_schema_cache") and self._table_schema_cache:
            return self._table_schema_cache
        schema: Dict[str, List[Dict[str, str]]] = {}
        try:
            cursor = self.conn.cursor()
            tables = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (table_name,) in tables:
                cols = cursor.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                schema[table_name] = [
                    {"name": col[1], "type": col[2] or ""}
                    for col in cols
                ]
            self._table_schema_cache = schema
        except Exception as e:
            logger.error(f"Failed to load table schema: {e}")
            self._table_schema_cache = {}
        return self._table_schema_cache

    def _find_columns_in_question(self, question: str, columns: List[str]) -> List[Tuple[str, int]]:
        """Return columns whose names appear in the question along with their positions"""
        q_lower = question.lower()
        matches = []
        for col in columns:
            col_lower = col.lower()
            idx = q_lower.find(col_lower)
            if idx != -1:
                matches.append((col, idx))
        return sorted(matches, key=lambda item: item[1])

    def _maybe_handle_top_category_region_breakdown(self, question: str) -> Optional[Dict[str, Any]]:
        """General handler for "top dimension" requests followed by percentage breakdown by another dimension"""
        q_lower = question.lower()
        percentage_terms = any(term in q_lower for term in ["percentage", "percent", "%"])
        winner_terms = any(term in q_lower for term in ["most", "highest", "top", "biggest", "largest", "winner"])
        if not (percentage_terms and winner_terms):
            return None

        schema = self._get_table_schema()
        if not schema:
            return None

        numeric_types = {"integer", "real", "numeric", "float", "double", "decimal"}
        candidate_table = None
        candidate_columns: List[Dict[str, str]] = []
        best_match_count = 0
        for table_name, cols in schema.items():
            column_names = [col["name"] for col in cols]
            mentioned = self._find_columns_in_question(question, column_names)
            if len(mentioned) > best_match_count and len(mentioned) >= 2:
                candidate_table = table_name
                candidate_columns = cols
                best_match_count = len(mentioned)
        if not candidate_table:
            return None

        logger.info(
            "Detected top-dimension breakdown request for table %s using specialized handler",
            candidate_table
        )
        column_names = [col["name"] for col in candidate_columns]
        mentioned_sorted = self._find_columns_in_question(question, column_names)
        categorical_candidates = []
        numeric_candidates = []
        for col in candidate_columns:
            col_type_lower = (col["type"] or "").lower()
            if any(ntype in col_type_lower for ntype in numeric_types):
                numeric_candidates.append(col["name"])
            else:
                categorical_candidates.append(col["name"])

        mentioned_cats = [col for col, _ in mentioned_sorted if col in categorical_candidates]
        mentioned_nums = [col for col, _ in mentioned_sorted if col in numeric_candidates]

        if len(mentioned_cats) < 2:
            return None

        measure_column = mentioned_nums[0] if mentioned_nums else (numeric_candidates[0] if numeric_candidates else None)
        if not measure_column:
            return None

        primary_dim, secondary_dim = mentioned_cats[0], mentioned_cats[1]

        top_sql = (
            f"SELECT {primary_dim}, SUM({measure_column}) AS total_value "
            f"FROM {candidate_table} GROUP BY {primary_dim} "
            f"ORDER BY total_value DESC LIMIT 1"
        )
        try:
            top_df = pd.read_sql_query(top_sql, self.conn)
            if top_df.empty:
                return None
            top_value = top_df.iloc[0][primary_dim]
            total_amount = float(top_df.iloc[0]["total_value"] or 0.0)
            breakdown_sql = (
                f"SELECT {secondary_dim}, SUM({measure_column}) AS segment_total "
                f"FROM {candidate_table} WHERE {primary_dim} = ? "
                f"GROUP BY {secondary_dim} ORDER BY segment_total DESC"
            )
            breakdown_df = pd.read_sql_query(breakdown_sql, self.conn, params=(top_value,))
            if breakdown_df.empty:
                return None
            if total_amount > 0:
                breakdown_df["percentage_of_total"] = (breakdown_df["segment_total"] / total_amount) * 100
            else:
                breakdown_df["percentage_of_total"] = 0.0
            breakdown_df[primary_dim] = top_value
            ordered_cols = [primary_dim, secondary_dim, "segment_total", "percentage_of_total"]
            breakdown_df = breakdown_df[ordered_cols]

            viz_response = VisualizationResponse(
                should_visualize=True,
                chart_types=["bar"],
                primary_chart="bar",
                x_axis=secondary_dim,
                y_axis="percentage_of_total",
                color_by=None,
                title=f"Share of {top_value} ({primary_dim}) by {secondary_dim}",
                visualization_rationale="Bar chart highlights each group's percentage contribution"
            )
            chart = self._create_visualization(breakdown_df, viz_response)
            contribution_parts = [
                f"{getattr(row, secondary_dim)}: {row.percentage_of_total:.1f}%"
                for row in breakdown_df.itertuples()
            ]
            contributions = ", ".join(contribution_parts)
            total_formatted = f"{total_amount:,.2f}"
            answer = (
                f"{primary_dim}='{top_value}' has the highest total ({total_formatted}). "
                f"Percentage contribution by {secondary_dim}: {contributions}."
            )
            human_breakdown_sql = breakdown_sql.replace("?", f"'{top_value}'")
            combined_sql = (
                f"{top_sql}\n\n-- Breakdown for {primary_dim}='{top_value}'\n{human_breakdown_sql}"
            )
            result = {
                "success": True,
                "question": question,
                "sql_query": combined_sql,
                "answer": answer,
                "data": breakdown_df,
                "visualization": None
            }
            if chart is not None:
                result["visualization"] = {
                    "chart": chart,
                    "type": viz_response.primary_chart,
                    "rationale": viz_response.visualization_rationale
                }
            return result
        except Exception as e:
            logger.error(f"Specialized top-dimension handler failed: {e}")
            return None

    def answer_question_with_context(
        self,
        question: str,
        reuse_data: bool = False
    ) -> Dict[str, Any]:
        """
        Answer question with conversation context awareness
        
        Args:
            question: User's natural language question
            reuse_data: If True, try to reuse previous DataFrame instead of querying
        
        Returns:
            Dictionary containing answer, data, visualization, and context info
        """
        
        try:
            # Step 1: Analyze intent
            intent = self._analyze_question_intent(question)
            logger.info(f"Detected intent: {intent.intent} (confidence: {intent.confidence})")
            
            # Step 2: Check for previous data reference
            references_prev, prev_df = self._check_previous_reference(question)
            
            # Step 3: Determine if we can reuse data
            should_reuse = reuse_data or (references_prev and prev_df is not None)
            
            if should_reuse and prev_df is not None:
                # Handle re-visualization or transformation on existing data
                if intent.intent == "re_visualize":
                    result = self._handle_revisualization(question, prev_df)
                elif intent.intent == "transform":
                    result = self._handle_transformation(question, prev_df)
                else:
                    # Use previous data but generate new analysis
                    result = self._process_with_existing_data(question, prev_df)
            else:
                # Generate new query
                result = self._process_new_query(question, intent)
            
            # Step 4: Update conversation state
            self._update_conversation_state(question, result, intent)
            
            # Step 5: Add metadata
            result["intent"] = intent.intent
            result["reused_data"] = should_reuse
            result["conversation_id"] = self.conversation_state.conversation_id
            
            return result
            
        except Exception as e:
            logger.error(f"Error in answer_question_with_context: {e}", exc_info=True)
            error_result = {
                "success": False,
                "error": str(e),
                "intent": "unknown",
                "reused_data": False
            }
            
            # Still log the failed attempt
            self.conversation_state.add_message(Message(
                role="user",
                content=question,
                metadata={"error": str(e)}
            ))
            
            return error_result

    def _process_new_query(self, question: str, intent: IntentAnalysis) -> Dict[str, Any]:
        """Process a new database query with retry logic"""
        
        # Test Ollama connection first
        if not self.test_ollama_connection():
            return {
                "success": False,
                "error": "Ollama is not running or model is not available",
                "question": question
            }

        specialized_result = self._maybe_handle_top_category_region_breakdown(question)
        if specialized_result:
            return specialized_result
        
        # Build context-aware prompt
        context_prompt = self._build_context_prompt(question, intent)
        
        # Get metadata from vector DB
        metadata_str = self._retrieve_metadata(question)
        
        # Generate SQL with retry logic
        try:
            sql_query = self._generate_sql_with_retry(question, context_prompt, metadata_str)
        except Exception as e:
            logger.error(f"SQL generation failed after retries: {e}")
            return {
                "success": False,
                "error": f"Failed to generate SQL query: {str(e)}",
                "question": question
            }
        
        # Clean SQL
        cleaned_sql = self._clean_sql(sql_query)
        
        # Remove unwanted LIMIT clauses
        final_sql = self._remove_unwanted_limit(cleaned_sql, question)
        
        # Validate and fix tables
        final_sql = self._validate_and_fix_tables(final_sql)
        
        # 🛡️ Apply safety guardrails for large datasets
        final_sql = self._add_safety_limits(final_sql, question)
        
        # 🛡️ Validate result size before execution - with smart regeneration
        validation = self._validate_result_size(final_sql)
        if not validation['safe']:
            # Try to regenerate with aggregation hints instead of failing
            logger.warning(f"⚠️ Query too large ({validation['estimated_rows']:,} rows), attempting regeneration with aggregation")
            
            aggregation_prompt = f"""The previous SQL query will return {validation['estimated_rows']:,} rows, which is too large for a dashboard visualization.

Original question: {question}
Previous SQL: {final_sql}

Please regenerate the SQL with aggregation (GROUP BY, SUM, COUNT, AVG) to reduce result size.
For dashboard visualizations, aggregate by time period, category, product, region, or other meaningful dimensions.
Return ONLY the SQL query, no explanations."""
            
            try:
                response = self.llm.invoke(aggregation_prompt)
                regenerated_sql = self._clean_sql(response.content)
                regenerated_sql = self._validate_and_fix_tables(regenerated_sql)
                
                # Validate regenerated query
                revalidation = self._validate_result_size(regenerated_sql)
                if revalidation['safe']:
                    final_sql = regenerated_sql
                    logger.info(f"✅ Regenerated SQL reduces result to {revalidation['estimated_rows']:,} rows")
                else:
                    logger.warning("⚠️ Regeneration still too large, adding LIMIT 100000 as fallback")
                    final_sql = final_sql.rstrip(';') + ' LIMIT 100000'
            except Exception as e:
                logger.error(f"Failed to regenerate SQL with aggregation: {e}")
                # Fallback: add a LIMIT to prevent memory crash
                final_sql = final_sql.rstrip(';') + ' LIMIT 100000'
                logger.info("📌 Added LIMIT 100000 as fallback safety measure")
        
        if final_sql != sql_query:
            logger.info("SQL was modified during validation")
            logger.info(f"Final SQL: {final_sql}")
        
        logger.info(f"Generated SQL: {final_sql}")
        
        # Execute query with automatic Pandas/Dask routing based on data size
        df = None
        execution_error = None
        for exec_attempt in range(2):
            try:
                df = from_sql(final_sql, self.conn)
                logger.info(f"📊 DataFrame backend: {get_backend_name(df)}")
                
                #  OPTIMIZATION: Materialize Dask DataFrames once to avoid multiple .compute() calls
                if DataFrameFactory.is_dask(df):
                    logger.info(" Materializing Dask DataFrame once for reuse")
                    df = df.compute()  # Convert to Pandas once
                    logger.info(f" Materialized to Pandas: {len(df):,} rows")
                
                break
            except Exception as e:
                execution_error = str(e)
                logger.error(f"SQL execution failed (attempt {exec_attempt + 1}): {execution_error}")
                if exec_attempt == 1:
                    return {
                        "success": False,
                        "error": f"Query execution failed: {execution_error}",
                        "sql_query": final_sql,
                        "question": question
                    }
                logger.info("Attempting to regenerate SQL using execution error feedback")
                alias_summary = self._describe_query_aliases(final_sql)
                repair_prompt = self._augment_prompt_with_error(
                    context_prompt,
                    final_sql,
                    execution_error,
                    alias_summary
                )
                try:
                    regenerated_sql = self._generate_sql_with_retry(question, repair_prompt, metadata_str)
                except Exception as regen_error:
                    logger.error(f"SQL regeneration after execution failure failed: {regen_error}")
                    return {
                        "success": False,
                        "error": f"Query execution failed: {execution_error}",
                        "sql_query": final_sql,
                        "question": question
                    }
                cleaned_sql = self._clean_sql(regenerated_sql)
                cleaned_sql = self._remove_unwanted_limit(cleaned_sql, question)
                final_sql = self._validate_and_fix_tables(cleaned_sql)
                logger.info(f"Retry SQL after execution error: {final_sql}")
        
        #  Parallel execution: Generate answer and check visualization simultaneously
        logger.info("  Starting parallel execution: answer generation + visualization check")
        
        def task_answer():
            return self._generate_answer(question, df, final_sql)
        
        def task_viz_check():
            return self._should_visualize(question, df)
        
        # Execute both tasks in parallel
        answer, viz_response = self._execute_parallel([task_answer, task_viz_check])
        
        logger.info("  Parallel execution completed")
        
        result = {
            "success": True,
            "question": question,
            "sql_query": final_sql,
            "answer": answer,
            "data": df,
            "visualization": None
        }
        
        if viz_response.should_visualize:
            chart = self._create_visualization(df, viz_response)
            result["visualization"] = {
                "chart": chart,
                "type": viz_response.primary_chart,
                "rationale": viz_response.visualization_rationale
            }
        
        return result

    def _generate_sql_with_retry(self, question: str, context_prompt: str, metadata_str: str, max_retries: int = 3) -> str:
        """Generate SQL with retries for robustness"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"SQL generation attempt {attempt + 1}/{max_retries}")
                
                # Combine metadata and context prompt
                full_prompt = f"{metadata_str}\n\n{context_prompt}"
                
                # For retries, simplify the prompt to avoid confusion
                if attempt > 0:
                    # Simplified prompt for retry
                    full_prompt = f"""Generate a valid SQLite query for the following question.
Return ONLY the SQL query, nothing else.

Question: {question}

Available tables and columns:
{metadata_str}"""
                
                # Invoke the query chain to generate SQL
                sql_query = self.query_chain.invoke({"question": full_prompt})
                
                if sql_query and len(sql_query.strip()) > 0:
                    # Pre-validate the response before returning
                    cleaned = self._clean_sql(sql_query)
                    if cleaned and re.match(r"(?i)^(SELECT|WITH)\b", cleaned):
                        return sql_query
                    else:
                        logger.warning(f"Generated SQL failed validation on attempt {attempt + 1}")
                        last_error = "Generated response did not contain valid SQL"
                else:
                    logger.warning(f"Empty SQL generated on attempt {attempt + 1}")
                    last_error = "Empty SQL response"
                    
            except Exception as e:
                logger.error(f"SQL generation attempt {attempt + 1} failed: {e}")
                last_error = str(e)
                if attempt == max_retries - 1:
                    raise
        
        raise Exception(f"Failed to generate SQL after all retries: {last_error}")
    
    
    def _augment_prompt_with_error(
        self,
        base_prompt: str,
        sql_query: str,
        error_message: str,
        alias_summary: Optional[str] = None
    ) -> str:
        """Provide execution error feedback to the LLM for regeneration"""
        alias_hint = (
            f"\nCurrent FROM/JOIN aliases: {alias_summary}."
            " Avoid referencing tables or aliases that are not listed unless you explicitly join them."
        ) if alias_summary else ""

        return (
            f"{base_prompt}\n\n"
            f"Previous SQL attempt:\n{sql_query}\n\n"
            f"SQLite error: {error_message}.{alias_hint}\n"
            "Rewrite the SQL so it runs successfully and still answers the user's question."
        )

    def _add_safety_limits(self, sql: str, question: str) -> str:
        """
        Add automatic limits to prevent memory issues for dashboard queries.
        
        Strategy:
        1. If query has aggregation (GROUP BY, SUM, etc.) -> No limit needed
        2. If no aggregation + visualization requested -> LIMIT 10000
        3. If no aggregation + raw data -> LIMIT 1000
        """
        sql_upper = sql.upper()
        
        # Check if query has aggregation (safe for dashboards)
        has_aggregation = any(keyword in sql_upper for keyword in [
            'GROUP BY', 'COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('
        ])
        
        if has_aggregation:
            logger.info("✅ Query has aggregation - safe for large data")
            return sql
        
        # Check if LIMIT already exists
        if 'LIMIT' not in sql_upper:
            # Check if visualization is requested
            viz_keywords = ['show', 'visualize', 'chart', 'graph', 'plot', 'compare', 'trend']
            is_viz_request = any(kw in question.lower() for kw in viz_keywords)
            
            if is_viz_request:
                logger.warning(" Visualization query without aggregation - adding LIMIT 10000")
                sql = sql.rstrip(';') + ' LIMIT 10000'
            else:
                logger.warning(" Raw data query - adding LIMIT 1000")
                sql = sql.rstrip(';') + ' LIMIT 1000'
        
        return sql
    
    def _validate_result_size(self, sql: str) -> Dict[str, Any]:
        """
        Validate query result size before execution.
        
        Returns:
            Dict with 'safe' (bool), 'estimated_rows' (int), and 'suggestion' (str)
        """
        estimated_rows = DataFrameFactory.estimate_row_count(sql, self.conn)
        MAX_DASHBOARD_ROWS = 100_000
        
        if estimated_rows > MAX_DASHBOARD_ROWS:
            logger.error(f"🚨 Query will return {estimated_rows:,} rows - too large for dashboard")
            
            sql_upper = sql.upper()
            has_aggregation = any(kw in sql_upper for kw in ['GROUP BY', 'COUNT(', 'SUM(', 'AVG('])
            
            if not has_aggregation:
                return {
                    'safe': False,
                    'estimated_rows': estimated_rows,
                    'suggestion': 'Add aggregation (GROUP BY) or filters (WHERE) to reduce result size'
                }
        
        return {
            'safe': True,
            'estimated_rows': estimated_rows
        }
    
    def _extract_sql_filters(self, sql_query: str) -> Dict[str, str]:
        """Extract WHERE clause conditions from SQL query"""
        try:
            if not sql_query:
                return {}
            
            where_match = re.search(r'\bWHERE\s+(.+?)(?:\bGROUP\s+BY|\bORDER\s+BY|\bLIMIT|\)|$)', 
                                    sql_query, re.IGNORECASE | re.DOTALL)
            if where_match:
                where_clause = where_match.group(1).strip()
                conditions = {}
                
                # Parse simple equality conditions 
                for match in re.finditer(r'(\w+)\s*=\s*[\'"](.*?)[\'"]', where_clause):
                    col_name = match.group(1)
                    col_value = match.group(2)
                    conditions[col_name] = col_value
                    logger.info(f"Extracted filter: {col_name} = {col_value}")
                
                return conditions
            return {}
        except Exception as e:
            logger.error(f"Failed to extract filters from SQL: {e}")
            return {}

    def _handle_revisualization(self, question: str, df: UnifiedDataFrame) -> Dict[str, Any]:
        """Handle re-visualization of existing data with filter preservation"""
        
        logger.info("Handling re-visualization request")
        
        # Check if previous query had filters we should preserve
        previous_sql = None
        previous_filters = {}
        
        recent_messages = self.conversation_state.get_recent_messages(2)
        for msg in reversed(recent_messages):
            if msg.role == "assistant" and msg.sql_query:
                previous_sql = msg.sql_query
                previous_filters = self._extract_sql_filters(previous_sql)
                break
        
        # Apply filters to dataframe if they exist
        filtered_df = DataFrameFactory.copy(df)
        if previous_filters:
            logger.info(f"Applying preserved filters: {previous_filters}")
            for col_name, col_value in previous_filters.items():
                if col_name in filtered_df.columns:
                    # Case-insensitive string matching
                    if filtered_df[col_name].dtype == 'object':
                        filtered_df = filtered_df[filtered_df[col_name].str.lower() == col_value.lower()]
                    else:
                        filtered_df = filtered_df[filtered_df[col_name] == col_value]
                    logger.info(f"Applied filter {col_name}={col_value}, remaining rows: {len(filtered_df)}")
        
        # Analyze what visualization is requested
        viz_response = self._should_visualize(question, filtered_df)
        
        # Generate new answer focusing on visualization
        if previous_filters:
            filter_desc = ", ".join([f"{k}='{v}'" for k, v in previous_filters.items()])
            answer = f"I've created a {viz_response.primary_chart} visualization of the filtered data ({filter_desc}). {viz_response.visualization_rationale}"
        else:
            answer = f"I've created a {viz_response.primary_chart} visualization of the previous data. {viz_response.visualization_rationale}"
        
        result = {
            "success": True,
            "question": question,
            "sql_query": previous_sql,  # Preserve original SQL context
            "answer": answer,
            "data": filtered_df,
            "visualization": None
        }
        
        if viz_response.should_visualize:
            chart = self._create_visualization(filtered_df, viz_response)
            result["visualization"] = {
                "chart": chart,
                "type": viz_response.primary_chart,
                "rationale": viz_response.visualization_rationale
            }
        
        return result

    def _handle_transformation(self, question: str, df: UnifiedDataFrame) -> Dict[str, Any]:
        """Handle transformation of existing data (filter, sort, calculate)"""
        
        logger.info("Handling data transformation request")
        
        # Convert to Pandas for LLM-generated code execution
        # (LLM generates Pandas code, and exec() is safer with Pandas)
        if DataFrameFactory.is_dask(df):
            logger.info("⚠️ Converting Dask to Pandas for transformation")
            work_df = df.compute()
        else:
            work_df = df.copy()
        
        # Use LLM to determine transformation
        transform_prompt = f"""Given this DataFrame with columns: {list(work_df.columns)}

User request: "{question}"

Suggest pandas transformation code to fulfill this request.
Return ONLY the Python code, no explanations.
Use 'df' as the DataFrame variable name.

Example formats:
- df[df['sales'] > 1000]
- df.sort_values('date', ascending=False)
- df.groupby('category')['sales'].sum()
- df['profit'] = df['revenue'] - df['cost']
"""

        try:
            response = self.llm.invoke(transform_prompt)
            code = response.content.strip()
            
            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            # Execute transformation safely on Pandas DataFrame
            local_vars = {"df": work_df, "pd": pd}
            exec(code, {"__builtins__": {}}, local_vars)
            transformed_df = local_vars["df"]
            
            # Generate answer
            answer = f"I've transformed the data: {code}\n\nResult has {DataFrameFactory.get_length(transformed_df)} rows."
            
            # Check for visualization
            viz_response = self._should_visualize(question, transformed_df)
            
            result = {
                "success": True,
                "question": question,
                "sql_query": None,
                "transformation": code,
                "answer": answer,
                "data": transformed_df,
                "visualization": None
            }
            
            if viz_response.should_visualize:
                chart = self._create_visualization(transformed_df, viz_response)
                result["visualization"] = {
                    "chart": chart,
                    "type": viz_response.primary_chart,
                    "rationale": viz_response.visualization_rationale
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            return {
                "success": False,
                "error": f"Could not transform data: {str(e)}",
                "data": df
            }

    def _process_with_existing_data(self, question: str, df: UnifiedDataFrame) -> Dict[str, Any]:
        """Process question using existing data without new query"""
        
        answer = self._generate_answer(question, df, None)
        viz_response = self._should_visualize(question, df)
        
        result = {
            "success": True,
            "question": question,
            "sql_query": None,
            "answer": answer,
            "data": df,
            "visualization": None
        }
        
        if viz_response.should_visualize:
            chart = self._create_visualization(df, viz_response)
            result["visualization"] = {
                "chart": chart,
                "type": viz_response.primary_chart,
                "rationale": viz_response.visualization_rationale
            }
        
        return result

    def _update_conversation_state(
        self,
        question: str,
        result: Dict[str, Any],
        intent: IntentAnalysis
    ):
        """Update conversation state with new interaction"""
        
        # Add user message
        self.conversation_state.add_message(Message(
            role="user",
            content=question,
            metadata={"intent": intent.intent}
        ))
        
        # Add assistant message
        df_snapshot = None
        if result.get("data") is not None:
            df = result["data"]
            df_snapshot = {
                "columns": list(df.columns),
                "row_count": len(df),
                "sample": df.head(3).to_dict() if len(df) > 0 else {}
            }
        
        viz_info = None
        figure_json = None
        if result.get("visualization"):
            viz_info = result["visualization"].get("type")
            # Store figure as JSON string for later retrieval
            if result["visualization"].get("chart"):
                try:
                    figure_json = result["visualization"]["chart"].to_json()
                except Exception as e:
                    logger.warning(f"Failed to serialize figure to JSON: {e}")
        
        self.conversation_state.add_message(Message(
            role="assistant",
            content=result.get("answer", ""),
            sql_query=result.get("sql_query"),
            dataframe_snapshot=df_snapshot,
            visualization=viz_info,
            figure_json=figure_json,
            metadata={
                "intent": intent.intent,
                "success": result.get("success", False)
            }
        ))
        
        # Add data context if new data was retrieved
        if result.get("data") is not None and result.get("sql_query"):
            df = result["data"]
            self.conversation_state.add_data_context(DataContext(
                query=result["sql_query"],
                columns=list(df.columns),
                row_count=len(df),
                sample_data=df.head(5).to_dict() if len(df) > 0 else {}
            ))
        
        # Add visualization record
        if result.get("visualization"):
            self.conversation_state.add_visualization(VisualizationRecord(
                question=question,
                chart_type=result["visualization"]["type"],
                data_summary=f"{len(result['data'])} rows" if result.get("data") is not None else "N/A"
            ))

    
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding with caching to avoid redundant API calls"""
        import hashlib
        
        # Normalize text for better cache hits
        normalized = ' '.join(text.lower().split())
        cache_key = hashlib.md5(normalized.encode()).hexdigest()
        
        if cache_key in self._embedding_cache:
            logger.info("📦 Using cached embedding")
            return self._embedding_cache[cache_key]
        
        embedding = self._get_embedding(text)
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def _retrieve_metadata(self, question: str, top_k: int = 3) -> str:
        """Retrieve relevant metadata from vector database with improved caching"""
        import hashlib
        
        # Better cache key: hash of normalized question
        normalized_q = ' '.join(sorted(question.lower().split()))
        cache_key = hashlib.md5(normalized_q.encode()).hexdigest()
        
        if cache_key in self._metadata_cache:
            logger.info("📦 Using cached metadata")
            return self._metadata_cache[cache_key]
        
        try:
            if hasattr(self, 'table_collection') and self.table_collection:
                # Use cached embedding function
                query_embedding = self._get_cached_embedding(question)
                
                # Query vector database using embedding 
                results = self.table_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.table_collection.count())
                )
                
                if not results["documents"] or not results["documents"][0]:
                    logger.warning("No metadata found in vector database")
                    return "No metadata available"
                
                # Format retrieved metadata for the prompt
                metadata_parts = []
                for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                    table_info = f"Table: {metadata.get('table_name', 'Unknown')}"
                    metadata_parts.append(f"{table_info}\n{doc}")
                
                result = "\n\n".join(metadata_parts)
                self._metadata_cache[cache_key] = result  # Cache it
                return result
            else:
                logger.warning("Vector database not available, using basic schema")
                return "Use available database schema"
        except Exception as e:
            logger.error(f"Error retrieving metadata: {e}")
            return "Error retrieving metadata"

    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and extract SQL query from LLM response (basic version)"""
        sql_query = sql_query.strip()
        
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
        
        sql_query = re.sub(r"^SQLQuery:\s*", "", sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r"^Answer:\s*", "", sql_query, flags=re.IGNORECASE)
        
        return sql_query.strip()

    def _clean_sql(self, query: str) -> str:
        """Advanced SQL query cleaning with comprehensive sanitization"""
        if not query:
            return query
        
        original = query
        q = query.strip()

        # Early detection of malformed response (LLM echoing prompt)
        # Check if response contains metadata/prompt text without valid SQL
        malformed_indicators = [
            "Table:", "Description:", "Business Context:", "Primary Key:",
            "Data Quality:", "Row Count:", "Column Count:", "Context from conversation:"
        ]
        has_sql_keyword = bool(re.search(r'\b(SELECT|WITH)\b', q, re.IGNORECASE))
        has_malformed_content = any(indicator in q for indicator in malformed_indicators)
        
        # If response looks like echoed prompt without SQL, return empty to trigger retry
        if has_malformed_content and not has_sql_keyword:
            logger.warning("LLM response appears to be echoed prompt text without SQL")
            return ""

        # Remove markdown-style fences
        q = re.sub(r"```\w*", "", q)
        q = q.replace("```", "")

        # Cut anything preceding the last explicit SQL marker
        marker_iter = list(re.finditer(r"(?i)(sqlquery|sql query|sql|query)\s*:", q))
        if marker_iter:
            q = q[marker_iter[-1].end():]

        q = q.strip()

        # Remove explanatory text before SELECT/WITH
        q = re.sub(r"(?i)^.*?(?:here'?s?\s+(?:the\s+)?(?:sql\s+)?query[:\s]+)", "", q)
        q = re.sub(r"(?i)^.*?(?:steps?[:\s]+)", "", q)
        
        # Remove numbered steps before SQL keywords
        if re.search(r'^[\d\.\s]+', q):
            match = re.search(r'\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b', q, re.IGNORECASE)
            if match:
                q = q[match.start():]
        
        # Handle 'Question:' prefix
        q = re.sub(r"(?is)^Question:.*?\b(SELECT|WITH)\b", r"\1", q)

        # Remove leading quotes
        q = q.strip('\"').strip("'")

        # Prefer retaining leading WITH clause if present
        cte_match = re.search(r"(?i)\bwith\s+[A-Za-z_][\w]*\s+as", q)
        select_match = re.search(r"(?i)\bselect\b", q)

        if cte_match and (not select_match or cte_match.start() <= select_match.start()):
            q = q[cte_match.start():]
        elif select_match:
            q = q[select_match.start():]
        elif cte_match:
            q = q[cte_match.start():]

        q = q.strip()

        # Keep only first statement if multiple
        statements = re.split(r";", q)
        for stmt in statements:
            cand = stmt.strip().strip('\"').strip("'")
            if re.match(r"(?i)^(SELECT|WITH)\b", cand):
                q = cand
                break
        else:
            q = q.strip()

        # Collapse excessive whitespace
        q = re.sub(r"\s+", " ", q).strip()
        
        # Fix common SQL syntax issues
        q = self._fix_sql_syntax(q)

        # Balance quotes
        q = self._balance_sql_quotes(q)

        if not re.match(r"(?i)^(SELECT|WITH)\b", q):
            logger.warning(f"Sanitized SQL does not start with SELECT/WITH: {q!r}")

        if original.strip() != q:
            logger.debug(f"Cleaned SQL from:\n{original}\n->\n{q}")
        
        return q

    def _balance_sql_quotes(self, query: str) -> str:
        """Auto-fix unmatched single quotes in SQL string literals"""
        try:
            single_quote_count = query.count("'")
            if single_quote_count % 2 == 1:
                logger.info("Unbalanced quotes detected, attempting to fix")
                if query.rstrip().endswith("'") is False:
                    query = query + "'"
                    logger.info("Added missing closing quote")

            m = re.search(r"(=\s*'[^']+)$", query)
            if m and not query.strip().endswith("''") and query.count("'") % 2 == 1:
                query = query + "'"
                logger.info("Fixed unbalanced quote in WHERE clause")
            
            return query
        except Exception as e:
            logger.debug(f"Quote balancing skipped due to error: {e}")
            return query
    
    def _fix_sql_syntax(self, query: str) -> str:
        """Fix common SQL syntax issues"""
        # Fix broken aliases like AS "Total" "Sales" -> AS "Total Sales"
        query = re.sub(r'AS\s+"([^"]+)"\s+"([^"]+)"', r'AS "\1 \2"', query, flags=re.IGNORECASE)
        
        # Fix other adjacent quoted strings (not in VALUES clause)
        if 'VALUES' not in query.upper():
            query = re.sub(r'"([^"]+)"\s+"([^"]+)"', r'"\1 \2"', query)
        
        # Fix ORDER BY with unquoted aliases containing spaces
        order_by_pattern = r'ORDER BY\s+([A-Za-z][A-Za-z0-9\s]+?)\s+(ASC|DESC|LIMIT)'
        order_by_matches = re.finditer(order_by_pattern, query, re.IGNORECASE)
        for match in order_by_matches:
            col_name = match.group(1).strip()
            if ' ' in col_name and not (col_name.startswith('"') and col_name.endswith('"')):
                quoted_col = f'"{col_name}"'
                query = query.replace(match.group(0), match.group(0).replace(col_name, quoted_col))
        
        # Fix missing closing quotes in GROUP BY
        if "GROUP BY" in query.upper() and query.count('"') % 2 != 0:
            group_by_match = re.search(r'GROUP BY\s+"?(\w+)"?$', query, re.IGNORECASE)
            if group_by_match:
                query = re.sub(r'(GROUP BY\s+)"?(\w+)"?$', r'\1"\2"', query, flags=re.IGNORECASE)
        
        # Fix missing closing quotes in ORDER BY
        if "ORDER BY" in query.upper() and query.count('"') % 2 != 0:
            order_by_match = re.search(r'ORDER BY\s+"?(\w+)"?\s*(ASC|DESC)?$', query, re.IGNORECASE)
            if order_by_match:
                query = re.sub(r'(ORDER BY\s+)"?(\w+)"?(\s*(ASC|DESC))?$', r'\1"\2"\3', query, flags=re.IGNORECASE)
        
        # Fix unwanted quotes
        if query.count('"') % 2 != 0:
            logger.info("Unbalanced quotes detected, fixing SQL Syntax")
            
            query = query.replace('"', '')
            
            sql_keywords = {
                'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'IS', 'NULL', 'LIKE',
                'BETWEEN', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER',
                'ON', 'GROUP', 'BY', 'ORDER', 'HAVING', 'AS', 'DISTINCT', 'UNION',
                'ALL', 'ANY', 'EXISTS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
                'LIMIT', 'OFFSET', 'WITH', 'VALUES', 'INSERT', 'UPDATE', 'DELETE',
                'CREATE', 'TABLE', 'ALTER', 'DROP', 'INDEX'}
                
            words = query.split()
            fixed_words = []
            
            for i, word in enumerate(words):
                clean_word = word.strip('(),')
                if clean_word.upper() in sql_keywords:
                    fixed_words.append(word)
                elif word.replace('.', '').replace('_', '').isalnum():
                    if '.' in word:
                        parts = word.split('.')
                        quoted_parts = [f'"{p}"' if not p.isdigit() else p for p in parts]
                        fixed_words.append('.'.join(quoted_parts))
                    else:
                        fixed_words.append(f'"{word}"')
                else:
                    fixed_words.append(word)
            
            query = ' '.join(fixed_words)
            logger.info("Fixed SQL syntax by re-adding quotes around identifiers")
        
        return query

    def _remove_unwanted_limit(self, query: str, original_question: str) -> str:
        """Remove LIMIT clause unless explicitly requested"""
        question_lower = original_question.lower()
        limit_keywords = ['top ', 'first ', 'limit ', 'maximum ', 'max ', 'bottom ', 'last ', 'lowest ']
        
        has_explicit_limit = any(keyword in question_lower for keyword in limit_keywords)
        
        if not has_explicit_limit:
            query_no_limit = re.sub(r'\s+LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
            if query_no_limit != query:
                logger.info("Removed implicit LIMIT clause")
                return query_no_limit
        
        return query

    def _extract_main_query_section(self, sql_query: str) -> str:
        """Return the portion of the SQL that represents the main query (after CTEs)"""
        stripped = sql_query.lstrip()
        lowered = stripped.lower()
        if not lowered.startswith("with"):
            return sql_query
        idx = lowered.find("with") + 4
        depth = 0
        i = idx
        while i < len(stripped):
            char = stripped[i]
            if char == '(':
                depth += 1
            elif char == ')':
                if depth > 0:
                    depth -= 1
                if depth == 0:
                    j = i + 1
                    while j < len(stripped) and stripped[j].isspace():
                        j += 1
                    if j < len(stripped) and stripped[j] == ',':
                        i = j
                        continue
                    return stripped[j:]
            i += 1
        return sql_query

    def _analyze_query_aliases(self, sql_query: str):
        """Extract alias metadata from the main query section"""
        main_query_section = self._extract_main_query_section(sql_query)
        from_pattern = r"(?i)(?:FROM|JOIN)\s+([a-zA-Z_][\w]*)\s*(?:AS\s+)?([a-zA-Z_][\w]*)?"

        valid_aliases = set()
        base_to_alias = {}
        alias_to_table = {}
        summary_parts = []

        for table_name, alias in re.findall(from_pattern, main_query_section):
            table_clean = table_name.strip()
            alias_clean = alias.strip() if alias else ""
            table_key = table_clean.lower()
            if alias_clean:
                alias_key = alias_clean.lower()
                valid_aliases.add(alias_key)
                base_to_alias[table_key] = alias_key
                alias_to_table[alias_key] = table_clean
                summary_parts.append(f"{alias_clean} -> {table_clean}")
            else:
                valid_aliases.add(table_key)
                base_to_alias[table_key] = table_key
                alias_to_table[table_key] = table_clean
                summary_parts.append(table_clean)

        derived_alias_pattern = r"(?i)\)\s+(?:AS\s+)?([a-zA-Z_][\w]*)"
        for alias in re.findall(derived_alias_pattern, main_query_section):
            alias_key = alias.lower()
            valid_aliases.add(alias_key)
            alias_to_table.setdefault(alias_key, "derived_table")
            summary_parts.append(f"{alias} (derived)")

        if summary_parts:
            unique_summary = []
            for part in summary_parts:
                if part not in unique_summary:
                    unique_summary.append(part)
            summary_text = "; ".join(unique_summary)
        else:
            summary_text = "No FROM/JOIN aliases detected."

        return valid_aliases, base_to_alias, alias_to_table, summary_text

    def _describe_query_aliases(self, sql_query: str) -> str:
        """Return a human-readable alias summary for prompts/logging"""
        _, _, _, summary = self._analyze_query_aliases(sql_query)
        return summary

    def _validate_and_fix_tables(self, sql_query: str) -> str:
        """Validate table and column names in SQL query and fix if needed (with caching)"""
        import hashlib
        
        # Check cache first - same SQL = same validation result
        sql_hash = hashlib.md5(sql_query.encode()).hexdigest()
        if sql_hash in self._sql_validation_cache:
            logger.info("📦 Using cached SQL validation")
            return self._sql_validation_cache[sql_hash]
        
        # Perform validation
        result = self._validate_and_fix_tables_impl(sql_query)
        
        # Cache the result
        self._sql_validation_cache[sql_hash] = result
        
        # Limit cache size to prevent memory bloat
        if len(self._sql_validation_cache) > 100:
            # Remove oldest entries (simple FIFO)
            keys = list(self._sql_validation_cache.keys())
            for key in keys[:50]:  # Remove half
                del self._sql_validation_cache[key]
        
        return result
    
    def _validate_and_fix_tables_impl(self, sql_query: str) -> str:
        """Implementation of SQL validation (uncached)"""
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        valid_tables = {row[0].lower() for row in cursor.fetchall()}

        cte_pattern = r"(?i)with\s+([a-zA-Z_][\w]*)\s+as"
        cte_names = {name.lower() for name in re.findall(cte_pattern, sql_query)}
        valid_tables.update(cte_names)
        
        table_columns = {}
        for table in valid_tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            table_columns[table] = columns
        
        column_map = {}
        for table, columns in table_columns.items():
            for col in columns:
                col_lower = col.lower()
                if col_lower not in column_map:
                    column_map[col_lower] = (table, col)

        valid_aliases, base_to_alias, _, _ = self._analyze_query_aliases(sql_query)
        
        # Extract subquery aliases and their columns
        subquery_pattern = r'\(\s*SELECT\s+(.+?)\s+FROM\s+.+?\)\s+(?:AS\s+)?(\w+)'
        subqueries = re.finditer(subquery_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        subquery_columns = {}
        
        for match in subqueries:
            select_list = match.group(1)
            alias = match.group(2)
            # Extract column names/aliases from SELECT list
            cols = []
            for item in select_list.split(','):
                item = item.strip()
                # Check for "AS alias" pattern
                as_match = re.search(r'\s+AS\s+(\w+)', item, re.IGNORECASE)
                if as_match:
                    cols.append(as_match.group(1).lower())
                else:
                    # Take last word (simple column name)
                    words = item.split()
                    if words:
                        cols.append(words[-1].lower())
            subquery_columns[alias.lower()] = cols
            logger.info(f"Subquery '{alias}' provides columns: {cols}")
        
        query_lower = sql_query.lower()
        
        # Find JOIN patterns with non-existent tables
        join_pattern = r'(?:LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)?'
        joins = re.finditer(join_pattern, query_lower, re.IGNORECASE)
        
        invalid_tables = set()
        for match in joins:
            table_name = match.group(1).lower()
            if table_name not in valid_tables:
                invalid_tables.add(table_name)
                alias = match.group(2) if match.group(2) else table_name
                invalid_tables.add(alias.lower())
        
        if invalid_tables:
            logger.info(f"Found invalid tables in JOINs: {invalid_tables}")
            
            # Remove JOIN clauses with invalid tables
            for invalid_table in invalid_tables:
                sql_query = re.sub(
                    rf'(?:LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|JOIN)\s+{invalid_table}\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s+ON\s+[^\s]+\s*=\s*[^\s]+',
                    '',
                    sql_query,
                    flags=re.IGNORECASE
                )
            
            # Replace column references with invalid table aliases
            for invalid_alias in invalid_tables:
                pattern = rf'{invalid_alias}\.([a-zA-Z_][a-zA-Z0-9_]*)'
                matches = re.finditer(pattern, sql_query, re.IGNORECASE)
                
                for match in matches:
                    column_name = match.group(1).lower()
                    if column_name in column_map:
                        valid_table, valid_col = column_map[column_name]
                        replacement_table = base_to_alias.get(valid_table.lower(), valid_table)
                        replacement = f"{replacement_table}.{valid_col}"
                        sql_query = sql_query.replace(match.group(0), replacement)
                        logger.info(f"Replaced {match.group(0)} with {replacement}")
            
            # Handle aggregate functions
            agg_pattern = r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*\)'
            sql_query = re.sub(
                agg_pattern,
                lambda m: self._fix_aggregate_reference(m, invalid_tables, column_map),
                sql_query,
                flags=re.IGNORECASE
            )
            
            sql_query = self._cleanup_broken_clauses(sql_query)
        
        # Fix column references in window functions (e.g., SUM(alias.column) OVER ())
        window_pattern = r'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*(\w+)\.(\w+)\s*\)'
        
        def fix_window_column(match):
            func = match.group(1)
            table_or_alias = match.group(2).lower()
            column = match.group(3).lower()
            
            # Check if it's a subquery alias
            if table_or_alias in subquery_columns:
                # Verify column exists in subquery output
                if column not in subquery_columns[table_or_alias]:
                    # Find closest match in subquery columns
                    available = subquery_columns[table_or_alias]
                    for available_col in available:
                        if available_col.lower().endswith(column) or column in available_col.lower():
                            logger.info(f"Fixed window function column: {table_or_alias}.{column} -> {table_or_alias}.{available_col}")
                            return f"{func}({table_or_alias}.{available_col})"
                    # If no match, use first numeric-looking column
                    if available:
                        logger.info(f"Fixed window function column: {table_or_alias}.{column} -> {table_or_alias}.{available[0]}")
                        return f"{func}({table_or_alias}.{available[0]})"
            
            return match.group(0)
        
        sql_query = re.sub(window_pattern, fix_window_column, sql_query, flags=re.IGNORECASE)
        
        # Also fix unqualified column references in window functions that should reference subquery columns
        unqualified_window_pattern = r'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*(\w+)\s*\)\s+OVER\s*\('
        
        def fix_unqualified_window(match):
            func = match.group(1)
            column = match.group(2).lower()
            
            # Check all subqueries to see if column exists
            for alias, cols in subquery_columns.items():
                # If column doesn't exist in subquery but similar one does
                if column not in cols:
                    for available_col in cols:
                        if available_col.endswith(column) or column in available_col or 'sales' in available_col.lower():
                            logger.info(f"Fixed unqualified window column: {column} -> {available_col}")
                            return f"{func}({available_col}) OVER ("
            
            return match.group(0)
        
        sql_query = re.sub(unqualified_window_pattern, fix_unqualified_window, sql_query, flags=re.IGNORECASE)
        
        # Replace references to unknown aliases with best-known equivalents
        alias_ref_pattern = r"([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)"

        def _replace_invalid_alias(match):
            alias = match.group(1)
            column = match.group(2)
            alias_lower = alias.lower()
            if alias_lower not in valid_aliases:
                if alias_lower in base_to_alias:
                    replacement_alias = base_to_alias[alias_lower]
                    logger.info(f"Replacing invalid alias reference {alias}.{column} -> {replacement_alias}.{column}")
                    return f"{replacement_alias}.{column}"
                logger.info(f"Leaving unknown alias reference {alias}.{column} untouched for clarity")
                return match.group(0)
            return match.group(0)

        sql_query = re.sub(alias_ref_pattern, _replace_invalid_alias, sql_query)
        
        return sql_query

    def _fix_aggregate_reference(self, match, invalid_tables, column_map):
        """Fix aggregate function references to invalid tables"""
        agg_func = match.group(1)
        full_ref = match.group(2)
        
        parts = full_ref.split('.')
        if len(parts) == 2:
            table_alias, column_name = parts
            if table_alias.lower() in invalid_tables:
                col_lower = column_name.lower()
                if col_lower in column_map:
                    valid_table, valid_col = column_map[col_lower]
                    return f"{agg_func}({valid_table}.{valid_col})"
        
        return match.group(0)

    def _cleanup_broken_clauses(self, sql_query: str) -> str:
        """Clean up SQL clauses after column/table removal"""
        # Remove empty aggregate functions
        sql_query = re.sub(r'(SUM|AVG|COUNT|MIN|MAX|TOTAL)\s*\(\s*\)', 'NULL', sql_query, flags=re.IGNORECASE)
        
        # Remove double commas
        sql_query = re.sub(r',\s*,', ',', sql_query)
        
        # Fix GROUP BY with leading/trailing commas
        sql_query = re.sub(r'GROUP BY\s*,', 'GROUP BY', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'GROUP BY\s+,\s*', 'GROUP BY ', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r',\s*GROUP BY', ' GROUP BY', sql_query, flags=re.IGNORECASE)
        
        # Fix ORDER BY with leading/trailing commas
        sql_query = re.sub(r'ORDER BY\s*,', 'ORDER BY', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'ORDER BY\s+,\s*', 'ORDER BY ', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r',\s*ORDER BY', ' ORDER BY', sql_query, flags=re.IGNORECASE)
        
        # Fix SELECT with trailing commas before FROM
        sql_query = re.sub(r',\s+FROM', ' FROM', sql_query, flags=re.IGNORECASE)
        
        # Fix WHERE with orphaned AND/OR
        sql_query = re.sub(r'WHERE\s+(AND|OR)\s+', 'WHERE ', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'(WHERE|AND|OR)\s+(AND|OR)\s+', r'\1 ', sql_query, flags=re.IGNORECASE)
        
        # Remove empty WHERE clauses
        sql_query = re.sub(r'WHERE\s+(GROUP BY|ORDER BY|LIMIT)', r'\1', sql_query, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        
        logger.debug("Cleaned up broken SQL clauses")
        return sql_query

    def _generate_answer(
        self,
        question: str,
        df: UnifiedDataFrame,
        sql_query: Optional[str]
    ) -> str:
        """Generate natural language answer from results"""
        
        if DataFrameFactory.empty_check(df):
            return "No results found for your question."
        
        row_count = DataFrameFactory.get_length(df)
        columns = DataFrameFactory.get_columns(df)
        data_summary = f"Found {row_count} rows"
        if row_count > 0:
            data_summary += f" with columns: {', '.join(columns)}"
        
        # Convert to Pandas for display (sample if large)
        sample_df = ensure_pandas(df, max_rows=30)
        sample_data = sample_df.to_string(index=False)
        column_types = ", ".join([f"{col}({str(dtype)})" for col, dtype in sample_df.dtypes.items()])

        
        answer_prompt = f"""You are a careful data analyst. Answer using ONLY the data shown below.
    Do NOT mention charts/plots/graphs/visualizations. Do NOT apologize about not plotting. The system renders visuals separately.

    Question:
    {question}

    Data summary:
    {data_summary}

    Columns (name(type)):
    {column_types}

    Sample data (may be truncated):
    {sample_data}

    Answer format requirements:
    Provide a clear, concise answer to the question based on this data. 
    Focus on key insights and patterns.
    Be specific with numbers when relevant.
    1) Start with a direct 1–2 sentence answer.
    2) Then provide 3–6 bullets with the most important quantitative findings.
       - Include totals, counts, and percentages/shares when applicable.
       - For categorical breakdowns (e.g., gender/category/region), list each group with its value and its share of the total.
       - Call out the top group and the gap vs the next group when relevant.
    3) If the provided rows are insufficient to compute an exact metric (e.g., sample is truncated), say so explicitly and avoid guessing.
    4) Plain text only. No SQL. No Python.
    """

        try:
            response = self.llm.invoke(answer_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Query executed successfully. {data_summary}"

    def _should_visualize(self, question: str, df: UnifiedDataFrame) -> VisualizationResponse:
        """Determine if and how to visualize the results"""
        
        row_count = DataFrameFactory.get_length(df)
        if DataFrameFactory.empty_check(df) or row_count < 2:
            return VisualizationResponse(
                should_visualize=False,
                primary_chart="none",
                visualization_rationale="Not enough data for visualization"
            )
        
        # Check if visualization is requested
        viz_keywords = ['visualize', 'plot', 'chart', 'graph', 'show', 'display', 'indicate', 'represent',
                       'draw', 'diagram', 'illustration', 'picture', 'pie', 'bar', 'line', 'scatter', 'histogram', 'box']
        q_lower = question.lower()
        is_viz_requested = any(keyword in q_lower for keyword in viz_keywords)
        
        if not is_viz_requested:
            return VisualizationResponse(
                should_visualize=False,
                primary_chart="none",
                visualization_rationale="No visualization requested"
            )
        
        # Try LLM-based recommendation
        try:
            parser = PydanticOutputParser(pydantic_object=VisualizationResponse)
            
            # Get sample data for LLM (ensure Pandas)
            sample_df = ensure_pandas(df, max_rows=5)
            columns = DataFrameFactory.get_columns(df)
            numeric_cols = sample_df.select_dtypes(include=['number']).columns.tolist()
            
            viz_prompt = f"""Analyze this data and determine if visualization would be helpful.

Question: {question}

Data info:
- Rows: {row_count}
- Columns: {', '.join(columns)}
- Numeric columns: {', '.join(numeric_cols)}

Sample data:
{sample_df.to_string()}

{parser.get_format_instructions()}

Available chart types: bar, line, pie, scatter, histogram, box, multiple_bar"""

            response = self.llm.invoke(viz_prompt)
            content = response.content.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return parser.parse(content)
        except Exception as e:
            logger.error(f"Error in visualization decision: {e}")
            return self._simple_viz_recommendation(question, df)

    def _simple_viz_recommendation(self, question: str, result_df: pd.DataFrame) -> VisualizationResponse:
        """Simple rule-based visualization recommendation as fallback"""
        q_lower = question.lower()
        columns = list(result_df.columns)
        
        # Detect box plot request
        if 'box' in q_lower or 'distribution' in q_lower:
            category_col = None
            numeric_col = None
            
            for col in columns:
                if result_df[col].dtype in ['object', 'string']:
                    category_col = col
                elif result_df[col].dtype in ['int64', 'float64']:
                    numeric_col = col
            
            # Check if we have multiple values per category (needed for box plots)
            if category_col and numeric_col:
                counts = result_df.groupby(category_col).size()
                if (counts >= 4).any():
                    return VisualizationResponse(
                        should_visualize=True,
                        chart_types=['box'],
                        primary_chart='box',
                        x_axis=category_col,
                        y_axis=numeric_col,
                        color_by=category_col,
                        title=f'{numeric_col} Distribution by {category_col}',
                        visualization_rationale='Box plot for distribution analysis'
                    )
            
            # Fallback for box plot
            return VisualizationResponse(
                should_visualize=True,
                chart_types=['box'],
                primary_chart='box',
                x_axis=category_col,
                y_axis=numeric_col,
                color_by=category_col,
                title=f'{numeric_col or "Value"} Distribution by {category_col or "Category"}',
                visualization_rationale='Box plot for distribution analysis'
            )
        
        # Default bar chart
        return VisualizationResponse(
            should_visualize=True,
            chart_types=['bar'],
            primary_chart='bar',
            x_axis=columns[0] if len(columns) > 0 else None,
            y_axis=columns[1] if len(columns) > 1 else None,
            color_by=None,
            title='Data Visualization',
            visualization_rationale='Default bar chart visualization'
        )

    def _resolve_column_name(self, df: pd.DataFrame, column_name: Optional[str]) -> Optional[str]:
        """Resolve a column reference against dataframe columns (case/quote insensitive)"""
        if column_name is None:
            return None
        stripped = str(column_name).strip().strip('"`[]')
        if not stripped:
            return None
        columns_lower = {col.lower(): col for col in df.columns}
        if stripped in df.columns:
            return stripped
        return columns_lower.get(stripped.lower())

    def _select_default_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Return default categorical and numeric columns for visualization fallbacks"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = [col for col in df.columns if col not in numeric_cols]
        default_cat = categorical_cols[0] if categorical_cols else (df.columns[0] if len(df.columns) > 0 else None)
        default_num = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else None)
        return default_cat, default_num

    def _create_visualization(
        self,
        df: UnifiedDataFrame,
        viz_response: VisualizationResponse
    ) -> Optional[go.Figure]:
        """
        Create visualization based on recommendation.
        Enhanced to handle both Pandas and Dask DataFrames.
        
        Args:
            df (UnifiedDataFrame): The data to visualize (Pandas or Dask).
            viz_response (VisualizationResponse): The visualization configuration.
            
        Returns:
            Optional[go.Figure]: The Plotly figure object or None if creation fails.
        """
        
        try:
            # ===== DASK COMPATIBILITY: Convert to Pandas for Plotly =====
            if DataFrameFactory.is_dask(df):
                total_rows = DataFrameFactory.get_length(df)
                if total_rows > 10_000:
                    logger.info(f"📉 Sampling 10,000 rows from {total_rows:,} for visualization")
                    df = df.head(10_000, npartitions=-1)
                else:
                    logger.info(f"🔄 Converting {total_rows:,} Dask rows to Pandas for visualization")
                    df = df.compute()
            # ===== END DASK COMPATIBILITY =====
            
            import plotly.express as px
            from pandas.api.types import is_numeric_dtype
            
            chart_type = viz_response.primary_chart.lower()
            title = viz_response.title or "Data Visualization"
            default_cat, default_num = self._select_default_columns(df)

            def infer_range_category_order(values: pd.Series) -> Optional[List[str]]:
                """Infer a logical ordering for categories like '<20', '20-30', '60+' etc."""
                try:
                    if values is None or len(values) == 0:
                        return None
                    if is_numeric_dtype(values):
                        return None

                    uniques = [str(v).strip() for v in pd.unique(values.dropna())]
                    if len(uniques) < 2:
                        return None

                    parsed: List[Tuple[float, float, str]] = []
                    unknown: List[str] = []

                    re_between = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*$")
                    re_plus = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")
                    re_less = re.compile(r"^\s*(?:<|<=|less\s+than|under)\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
                    re_ge = re.compile(r"^\s*(?:>=|at\s+least)\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE)

                    for label in uniques:
                        m = re_less.match(label)
                        if m:
                            upper = float(m.group(1))
                            parsed.append((-1.0e18, upper, label))
                            continue
                        m = re_between.match(label)
                        if m:
                            lo = float(m.group(1))
                            hi = float(m.group(2))
                            parsed.append((lo, hi, label))
                            continue
                        m = re_plus.match(label)
                        if m:
                            lo = float(m.group(1))
                            parsed.append((lo, 1.0e18, label))
                            continue
                        m = re_ge.match(label)
                        if m:
                            lo = float(m.group(1))
                            parsed.append((lo, 1.0e18, label))
                            continue
                        unknown.append(label)

                    # Only apply custom ordering if we successfully parsed most categories
                    if len(parsed) < 2:
                        return None

                    parsed_sorted = sorted(parsed, key=lambda t: (t[0], t[1]))
                    ordered = [t[2] for t in parsed_sorted]
                    # Append unknown categories at the end, preserving their original order
                    for label in uniques:
                        if label in unknown and label not in ordered:
                            ordered.append(label)
                    return ordered
                except Exception:
                    return None

            def normalize_axis(axis_value, prefer_numeric=False, allow_multiple=False):
                if axis_value is None:
                    return [] if allow_multiple else None
                if allow_multiple and isinstance(axis_value, list):
                    normalized = []
                    for col in axis_value:
                        resolved = self._resolve_column_name(df, col)
                        if resolved:
                            normalized.append(resolved)
                    if normalized:
                        return normalized
                    return [default_num] if default_num else []
                if allow_multiple and not isinstance(axis_value, list):
                    resolved = self._resolve_column_name(df, axis_value)
                    return [resolved] if resolved else ([default_num] if default_num else [])
                resolved = self._resolve_column_name(df, axis_value)
                if resolved:
                    return resolved
                if prefer_numeric:
                    return default_num
                return default_cat if default_cat != resolved else default_num

            x_axis = normalize_axis(viz_response.x_axis, prefer_numeric=False, allow_multiple=False)
            color_by = self._resolve_column_name(df, viz_response.color_by)

            if isinstance(viz_response.y_axis, list):
                y_axis = normalize_axis(viz_response.y_axis, prefer_numeric=True, allow_multiple=True)
            else:
                y_axis = normalize_axis(viz_response.y_axis, prefer_numeric=True, allow_multiple=False)

            # Final fallbacks to ensure axes exist for specific chart types
            if not x_axis:
                x_axis = default_cat or default_num
            if isinstance(y_axis, list):
                if not y_axis and default_num:
                    y_axis = [default_num]
            else:
                if not y_axis:
                    y_axis = default_num
                    if chart_type != "pie" and y_axis is None and len(df.columns) > 1:
                        y_axis = df.columns[1]
            
            # Check if we have NO numeric columns at all - need to compute counts
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            needs_count_aggregation = len(numeric_cols) == 0 and len(df.columns) >= 2
            
            if needs_count_aggregation:
                # All columns are categorical - compute counts for visualization
                cat_cols = df.columns.tolist()
                if len(cat_cols) >= 2:
                    # Use first column as x-axis, second as color/grouping, compute counts
                    x_col = cat_cols[0]
                    color_col = cat_cols[1] if len(cat_cols) > 1 else None
                    
                    # Aggregate to get counts
                    if color_col:
                        df_counts = df.groupby([x_col, color_col]).size().reset_index(name='count')
                        logger.info(f"Computed counts for categorical data: {x_col} by {color_col}")
                    else:
                        df_counts = df.groupby(x_col).size().reset_index(name='count')
                        logger.info(f"Computed counts for categorical data: {x_col}")
                    
                    # Override df and axis settings for the chart
                    df = df_counts
                    x_axis = x_col
                    y_axis = 'count'
                    if color_col and not color_by:
                        color_by = color_col
            
            if chart_type == "bar":
                fig = px.bar(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=color_by,
                    title=title
                )
                # Ensure legend shows when using color grouping
                if color_by:
                    fig.update_layout(showlegend=True)
            
            elif chart_type == "multiple_bar":
                # For multiple bar charts, check if y_axis is a list of columns
                if isinstance(y_axis, list) and len(y_axis) > 1:
                    # Melt the dataframe to long format for grouped bars
                    df_melted = df.melt(
                        id_vars=[x_axis] if x_axis else None,
                        value_vars=y_axis,
                        var_name='Series',
                        value_name='Value'
                    )
                    fig = px.bar(
                        df_melted,
                        x=x_axis,
                        y='Value',
                        color='Series',
                        barmode='group',
                        title=title
                    )
                    fig.update_layout(showlegend=True)
                else:
                    # Single y column with color grouping
                    fig = px.bar(
                        df,
                        x=x_axis,
                        y=y_axis if not isinstance(y_axis, list) else y_axis[0],
                        color=color_by,
                        barmode='group',
                        title=title
                    )
                    if color_by:
                        fig.update_layout(showlegend=True)
            
            elif chart_type == "line":
                fig = px.line(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=color_by,
                    title=title,
                    markers=True
                )
            
            elif chart_type == "pie":
                # Safely select top 10 rows for pie chart
                df_pie = df.copy()
                if len(df) > 10:
                    # Only use nlargest if y_axis is a valid numeric column
                    if y_axis and y_axis in df.columns and is_numeric_dtype(df[y_axis]):
                        df_pie = df.nlargest(10, y_axis)
                    else:
                        df_pie = df.head(10)
                
                fig = px.pie(
                    df_pie,
                    names=x_axis,
                    values=y_axis,
                    title=title
                )
                # Ensure legend is visible
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.05
                    )
                )
            
            elif chart_type == "scatter":
                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=color_by,
                    title=title
                )
            
            elif chart_type == "histogram":
                x_col = x_axis
                y_col: Optional[str] = None
                if isinstance(y_axis, list):
                    y_col = y_axis[0] if y_axis else None
                else:
                    y_col = y_axis

                # If x is categorical (like age ranges) OR y is provided (already aggregated),
                # render a bar chart instead of a numeric histogram.
                if x_col and x_col in df.columns and (not is_numeric_dtype(df[x_col]) or y_col):
                    if y_col and y_col in df.columns and is_numeric_dtype(df[y_col]):
                        fig = px.bar(df, x=x_col, y=y_col, title=title)
                    else:
                        # No usable y column; compute frequency counts
                        counts_df = (
                            df[x_col]
                            .astype(str)
                            .value_counts(dropna=False)
                            .rename_axis(x_col)
                            .reset_index(name="count")
                        )
                        fig = px.bar(counts_df, x=x_col, y="count", title=title)

                    order = infer_range_category_order(df[x_col])
                    if order:
                        fig.update_xaxes(categoryorder='array', categoryarray=order)
                else:
                    fig = px.histogram(
                        df,
                        x=x_col,
                        title=title
                    )
            
            elif chart_type == "box":
                x_col = x_axis
                y_col = y_axis
                
                if x_col and y_col:
                    counts = df.groupby(x_col).size()
                    if (counts < 4).all():
                        logger.warning("Insufficient data for box plot, using bar chart")
                        fig = px.bar(df, x=x_col, y=y_col, title=title)
                    else:
                        fig = px.box(
                            df,
                            x=x_col,
                            y=y_col,
                            color=x_col,
                            title=title,
                            points="outliers"
                        )
                        fig.update_traces(
                            marker=dict(size=4, opacity=0.6),
                            boxmean='sd'
                        )
                else:
                    fig = px.box(df, y=y_col, title=title)
            
            else:
                logger.warning(f"Unknown chart type: {chart_type}, defaulting to bar")
                fig = px.bar(
                    df,
                    x=x_axis,
                    y=y_axis,
                    title=title
                )

            # Apply natural ordering for range-like categorical x axes where possible
            if x_axis and x_axis in df.columns:
                order = infer_range_category_order(df[x_axis])
                if order:
                    fig.update_xaxes(categoryorder='array', categoryarray=order)
            
            fig.update_layout(
                template="plotly",
                title_font_size=16,
                showlegend=True,
                hovermode='closest',
                height=500,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.2)",
                    borderwidth=1
                )
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            return fig
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None 

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            "conversation_id": self.conversation_state.conversation_id,
            "message_count": len(self.conversation_state.messages),
            "data_contexts": len(self.conversation_state.data_contexts),
            "visualizations": len(self.conversation_state.visualizations),
            "latest_context": self.conversation_state.data_contexts[-1].__dict__ if self.conversation_state.data_contexts else None
        }

    def get_summary(self) -> Dict:
        """Get database and metadata summary"""
        summary = {
            'tables': len(self.table_metadata),
            'conversation': self.get_conversation_summary(),
            'table_details': []
        }
        
        for table_name, metadata in self.table_metadata.items():
            summary['table_details'].append({
                'table_name': table_name,
                'description': metadata.get('description', ''),
                'category': metadata.get('category', 'unknown'),
                'rows': metadata.get('row_count', 0),
                'columns': metadata.get('column_count', 0)
            })
        
        return summary

    def export_conversation(self) -> Dict[str, Any]:
        """Export conversation state"""
        return self.conversation_state.export_conversation()

    def close(self):
        """Close database connection and cleanup resources"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
        
        # Shutdown parallel executor
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logger.info("✅ Parallel executor closed")
