import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Tuple

import gradio as gr
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

from QueryAgent_Ollama_Enhanced import QueryAgentEnhanced
from conversation_manager import ConversationState
from dataframe_factory import DataFrameFactory, ensure_pandas, get_backend_name

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instance of the QueryAgent
agent: Optional[QueryAgentEnhanced] = None
# Global instance of the ConversationState
conversation_state: Optional[ConversationState] = None
# Directory to store conversation history
CONVERSATIONS_DIR = "conversations"

# --- Helper Functions ---

def ensure_conversations_dir():
    """
    Ensure the conversations directory exists.
    Creates the directory if it doesn't exist.
    """
    if not os.path.exists(CONVERSATIONS_DIR):
        os.makedirs(CONVERSATIONS_DIR)

def get_conversation_list() -> List[Tuple[str, str]]:
    """
    Get list of saved conversations for the sidebar.
    
    Reads JSON files from the conversations directory, extracts metadata
    (title, timestamp), and returns a list of tuples for the dropdown.
    
    Returns:
        List[Tuple[str, str]]: List of (label, filename) tuples.
    """
    ensure_conversations_dir()
    files = [f for f in os.listdir(CONVERSATIONS_DIR) if f.endswith('.json')]
    # Sort by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CONVERSATIONS_DIR, x)), reverse=True)
    
    choices = []
    for f in files:
        try:
            path = os.path.join(CONVERSATIONS_DIR, f)
            with open(path, 'r') as file:
                data = json.load(file)
                
            # Determine title from first user message
            title = "New Conversation"
            messages = data.get('messages', [])
            for msg in messages:
                if msg['role'] == 'user':
                    title = msg['content'][:30] + "..." if len(msg['content']) > 30 else msg['content']
                    break
            
            # Format date
            ts = data.get('start_time', datetime.now().isoformat())
            try:
                dt = datetime.fromisoformat(ts)
                date_str = dt.strftime("%m/%d %H:%M")
            except Exception:
                date_str = ""
                
            label = f"{date_str} - {title}"
            choices.append((label, f))
        except Exception as e:
            logger.error(f"Error reading conversation file {f}: {e}")
            continue
    return choices

def save_current_conversation():
    """
    Save current conversation state to disk.
    
    Exports the current conversation state to a JSON file in the
    conversations directory.
    """
    if conversation_state and conversation_state.messages:
        ensure_conversations_dir()
        data = conversation_state.export_conversation()
        filename = f"{conversation_state.conversation_id}.json"
        with open(os.path.join(CONVERSATIONS_DIR, filename), 'w') as f:
            json.dump(data, f, indent=2)

def load_conversation(filename: str) -> Tuple[List, str]:
    """
    Load a conversation and reconstruct history with images.
    
    Args:
        filename (str): The filename of the conversation to load.
        
    Returns:
        Tuple[List, str]: A tuple containing the reconstructed chat history
                          and a status message.
    """
    global conversation_state, agent
    
    if not filename:
        return [], ""
        
    filepath = os.path.join(CONVERSATIONS_DIR, filename)
    if not os.path.exists(filepath):
        return [], "File not found"
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Restore state
        conversation_state = ConversationState()
        conversation_state.import_conversation(data)
        if agent:
            agent.conversation_state = conversation_state
            
        # Reconstruct history
        history = []
        
        pending_user = None
        
        for msg in conversation_state.messages:
            if msg.role == "user":
                pending_user = msg.content
            elif msg.role == "assistant":
                # 1. Text Answer
                content = msg.content
                if msg.sql_query:
                    content += f"\n\n**Generated SQL:**\n```sql\n{msg.sql_query}\n```"

                if pending_user:
                    history.append((pending_user, content))
                    pending_user = None
                else:
                    history.append((None, content))
                
                # 2. Interactive Chart
                if msg.figure_json:
                    try:
                        fig = go.Figure(json.loads(msg.figure_json))
                        history.append((None, gr.Plot(value=fig, render=False)))
                    except Exception as e:
                        logger.error(f"Failed to reconstruct chart: {e}")
                        history.append((None, f"<div>Error loading chart: {e}</div>"))
                
                # 3. Data Table (from snapshot)
                if msg.dataframe_snapshot and msg.dataframe_snapshot.get('sample'):
                    try:
                        df_sample = pd.DataFrame.from_dict(msg.dataframe_snapshot['sample'])
                        # Re-order columns if possible
                        if msg.dataframe_snapshot.get('columns'):
                            cols = [c for c in msg.dataframe_snapshot['columns'] if c in df_sample.columns]
                            if cols:
                                df_sample = df_sample[cols]
                                
                        row_count = msg.dataframe_snapshot.get('row_count', len(df_sample))
                        html_table = df_sample.to_html(classes='data-table', index=False, border=0)
                        table_msg = f'<div class="table-container"><details open><summary>Data Preview ({row_count} rows)</summary>{html_table}</details></div>'
                        history.append((None, table_msg))
                    except Exception as e:
                        logger.error(f"Failed to reconstruct table: {e}")
        
        return history, f"Loaded: {filename}"
        
    except Exception as e:
        logger.error(f"Error loading conversation: {e}")
        return [], f"Error: {e}"

def start_new_chat():
    """
    Reset state for new chat.
    
    Creates a new ConversationState and updates the global agent.
    
    Returns:
        Tuple[List, str]: Empty chat history and status message.
    """
    global conversation_state, agent
    conversation_state = ConversationState()
    if agent:
        agent.conversation_state = conversation_state
    return [], "New Chat Started"

def delete_conversation():
    """
    Delete the current conversation.
    
    Removes the JSON file associated with the current conversation
    and resets the state.
    
    Returns:
        Tuple[List, dict, str]: Empty chat history, updated conversation list,
                                and status message.
    """
    global conversation_state, agent
    
    status_msg = ""
    if conversation_state and conversation_state.conversation_id:
        filename = f"{conversation_state.conversation_id}.json"
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                status_msg = "🗑️ Conversation deleted"
            except Exception as e:
                status_msg = f"❌ Error deleting: {e}"
        else:
            status_msg = "⚠️ Conversation not saved yet"
    
    # Reset state
    conversation_state = ConversationState()
    if agent:
        agent.conversation_state = conversation_state
    
    new_list = get_conversation_list()
    return [], gr.update(choices=new_list), status_msg

def initialize_agent(db_path: str, vector_db: str, model: str):
    """
    Initialize the agent and load history list.
    
    Args:
        db_path (str): Path to the SQLite database.
        vector_db (str): Path to the ChromaDB vector store.
        model (str): Name of the Ollama model to use.
        
    Returns:
        Tuple[str, dict]: Status message and updated conversation list.
    """
    global agent, conversation_state
    try:
        if not os.path.exists(db_path):
            return "❌ Database not found", gr.update()
        
        conversation_state = ConversationState()
        agent = QueryAgentEnhanced(
            source_db_path=db_path,
            vector_db_path=vector_db,
            llm_model=model,
            conversation_state=conversation_state
        )
        
        # Get initial history list
        history_list = get_conversation_list()
        
        return "✅ Agent Ready", gr.update(choices=history_list)
    except Exception as e:
        return f"❌ Error: {e}", gr.update()

def process_question(question: str, history: List):
    """
    Process a new question, update history, and save.
    
    Args:
        question (str): The user's question.
        history (List): The current chat history.
        
    Returns:
        Tuple[List, dict]: Updated chat history and updated conversation list.
    """
    global agent
    
    if not agent:
        history.append((question, "⚠️ Please initialize the agent first."))
        return history, gr.update()
    
    if not question.strip():
        return history, gr.update()
        
    try:
        # Get response from agent
        result = agent.answer_question_with_context(question)
        
        # Add backend info to show which processing method was used
        backend_info = ""
        if result.get('data') is not None:
            backend = get_backend_name(result['data'])
            row_count = DataFrameFactory.get_length(result['data'])
            backend_info = f"\n\n_Processing: {backend} ({row_count:,} rows)_"
        
        # 1. Text Answer
        answer_text = result['answer'] + backend_info
        if result.get('sql_query'):
            answer_text += f"\n\n**Generated SQL:**\n```sql\n{result['sql_query']}\n```"
        
        history.append((question, answer_text))
        
        # 2. Interactive Chart
        if result.get("visualization") and result["visualization"].get("chart"):
            try:
                fig = result["visualization"]["chart"]
                history.append((None, gr.Plot(value=fig, render=False)))
            except Exception as e:
                logger.error(f"Viz error: {e}")
                history.append((None, f"<div>Error creating chart: {e}</div>"))
        
        # 3. Data Table
        df = result.get("data")
        if df is not None and not DataFrameFactory.empty_check(df):
            # Convert to Pandas for display and limit rows
            df_display = ensure_pandas(df, max_rows=100)
            html_table = df_display.to_html(classes='data-table', index=False, border=0)
            total_rows = DataFrameFactory.get_length(df)
            table_msg = f'<div class="table-container"><details open><summary>Data Preview ({total_rows:,} rows)</summary>{html_table}</details></div>'
            history.append((None, table_msg))
        
        # Save conversation
        save_current_conversation()
        
        # Update history list
        new_list = get_conversation_list()
        
        return history, gr.update(choices=new_list)
        
    except Exception as e:
        history.append((question, f"❌ Error: {str(e)}"))
        return history, gr.update()

# --- UI Layout ---

custom_css = """
/* 1. Force Full Screen & Remove Outer Margins */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden;
}
.contain {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.app {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 2. Main Row Layout */
#main_row {
    height: 100% !important;
    gap: 0 !important;
}

/* 3. Sidebar Styling */
#sidebar_col {
    height: 100% !important;
    padding: 15px !important;
    border-right: 1px solid var(--border-color-primary);
    display: flex !important;
    flex-direction: column !important;
}

/* 4. Settings Section (Fixed at top) */
#settings_area {
    flex-shrink: 0 !important;
}

/* 5. Chat History List (Takes remaining space & scrolls) */
#history_list {
    flex-grow: 1 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-right: 5px;
    margin-top: 10px;
}

/* Thin scrollbar styling */
#history_list::-webkit-scrollbar {
    width: 6px;
}
#history_list::-webkit-scrollbar-thumb {
    background-color: #4b5563;
    border-radius: 4px;
}
#history_list::-webkit-scrollbar-track {
    background-color: transparent;
}

/* Chat area */
.chat-area { 
    padding: 20px; 
    height: 100vh; 
    overflow-y: auto; 
}
.avatar-image { border-radius: 50%; }
.message-row img { max-width: 100% !important; height: auto !important; border-radius: 8px; border: 1px solid var(--border-color-primary); margin-top: 10px; }
.message-wrap { white-space: pre-wrap !important; word-break: break-word !important; overflow-wrap: break-word !important; max-width: 100%; }
.message-row { overflow-x: hidden; }
.data-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
.data-table th { background-color: var(--background-fill-secondary); padding: 8px; text-align: left; border-bottom: 2px solid var(--border-color-primary); }
.data-table td { padding: 8px; border-bottom: 1px solid var(--border-color-primary); }
.table-container { overflow-x: auto; margin-top: 10px; border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 10px; }
.chart-container { margin-top: 15px; margin-bottom: 15px; max-width: 100%; overflow-x: auto; }
footer { visibility: hidden; }
"""

with gr.Blocks(title="Data Chat", css=custom_css, theme=gr.themes.Soft(), fill_height=True, fill_width=True) as demo:
    
    with gr.Row(elem_id="main_row"):
        
        # --- Sidebar ---
        with gr.Column(scale=1, min_width=300, elem_id="sidebar_col"):
            # Settings at the top (fixed, not scrollable)
            with gr.Column(elem_id="settings_area"):
                with gr.Accordion("⚙️ Settings", open=False):
                    db_input = gr.Textbox(label="Database Path", value="analysis.db")
                    vec_input = gr.Textbox(label="Vector DB", value="./chroma_db_768dim")
                    model_input = gr.Dropdown(
                        label="Model", 
                        choices=[
                            "qwen2.5:7b", 
                            "qwen2.5-coder:7b", 
                            "llama3.2:3b", 
                            "gemma3:4b", 
                            "gemma3:1b",
                            "qwen2.5:3b", 
                            "qwen3:8b", 
                            "qwen3:4b", 
                            "deepseek-r1:8b"
                        ], 
                        value="qwen2.5:7b"
                    )
                    init_btn = gr.Button("Initialize the LLM Agent")
                    init_status = gr.Markdown("Not Connected")
                
                gr.Markdown("### 🗄️ Chat History")
                with gr.Row():
                    new_chat_btn = gr.Button("+ New Chat", variant="primary", scale=3)
                    delete_btn = gr.Button("🗑️", variant="stop", scale=1, min_width=10)
            
            # Chat history list (scrollable)
            with gr.Column(elem_id="history_list"):
                history_list = gr.Radio(
                    label="Recent Conversations",
                    choices=get_conversation_list(),
                    interactive=True,
                    container=False
                )

        # --- Main Chat ---
        with gr.Column(scale=4, elem_classes="chat-area"):
            chatbot = gr.Chatbot(
                height=750, 
                avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"),
                render_markdown=True,
                show_label=False,
                type="tuples",
                sanitize_html=False
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    scale=9, 
                    placeholder="Ask a question about your data...", 
                    show_label=False, 
                    container=False,
                    lines=1
                )
                submit_btn = gr.Button("➤", scale=1, variant="primary")

    # --- Event Wiring ---
    
    # Initialize
    init_btn.click(
        fn=initialize_agent,
        inputs=[db_input, vec_input, model_input],
        outputs=[init_status, history_list]
    )
    
    # New Chat
    new_chat_btn.click(
        fn=start_new_chat,
        outputs=[chatbot, init_status]
    )
    
    # Delete Chat
    delete_btn.click(
        fn=delete_conversation,
        outputs=[chatbot, history_list, init_status]
    )
    
    # Load History
    history_list.select(
        fn=load_conversation,
        inputs=[history_list],
        outputs=[chatbot, init_status]
    )
    
    # Submit Question
    submit_args = {
        "fn": process_question,
        "inputs": [msg_input, chatbot],
        "outputs": [chatbot, history_list]
    }
    
    msg_input.submit(**submit_args).then(lambda: "", outputs=msg_input)
    submit_btn.click(**submit_args).then(lambda: "", outputs=msg_input)

if __name__ == "__main__":
    print("🚀 Starting UI...")
    demo.launch(server_name="0.0.0.0", server_port=6969)