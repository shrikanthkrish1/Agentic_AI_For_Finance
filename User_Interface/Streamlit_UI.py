import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
from typing import Optional
from datetime import datetime

from Financial_Agent import FinancialAgent

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Agentic AI Finance Analyst",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def load_custom_css():
    """Load custom CSS for better UI styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }

    .tool-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .status-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }

    .status-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }

    .status-error {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }

    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }

    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }

    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_main_header():
    """Display the main application header"""
    st.markdown('<div class="main-header">Agentic AI Finance Analyst</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
    <p><strong>Your AI-Powered Financial Intelligence System</strong></p>
    <p>Upload financial data → Auto-clean → Monte Carlo simulation → Financial storytelling</p>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar():
    """Build the application sidebar with controls and status"""
    with st.sidebar:
        st.markdown("## Control Panel")
        
        # Show agent status
        status_icons = {
            'ready': 'Ready', 
            'cleaning': 'Cleaning Data', 
            'simulating': 'Running Simulation',
            'storytelling': 'Creating Stories'
        }
        
        current_status = st.session_state.get('agent_status', 'ready')
        st.markdown(f"**Agent Status:** {status_icons.get(current_status, current_status).title()}")
        
        # Progress tracking
        st.markdown("### Progress Tracker")
        
        progress_steps = [
            ("Data Upload", st.session_state.get('data_uploaded', False)),
            ("Data Cleaning", st.session_state.get('data_cleaned', False)), 
            ("Monte Carlo", st.session_state.get('simulation_completed', False)),
            ("Ready for Stories", st.session_state.get('simulation_completed', False))
        ]
        
        for step_name, completed in progress_steps:
            icon = "✅" if completed else "⏳"
            st.markdown(f"{icon} {step_name}")
        
        st.markdown("---")
        
        # Quick action buttons
        st.markdown("### Quick Actions")
        
        if st.button("Reset Session", help="Clear all data and start over"):
            agent = FinancialAgent()
            agent.reset_session()
            st.rerun()
        
        if st.session_state.get('simulation_completed', False):
            if st.button("Show Results Summary"):
                stats = st.session_state.get('simulation_stats', {})
                summary_msg = f"""Quick Summary:
• Expected Net Income: ${stats.get('ni_mean', 0):,.0f}
• Profit Probability: {stats.get('prob_profit', 0):.1f}%
• Risk Level: {stats.get('prob_loss', 0):.1f}% chance of loss
• Best Case (95%): ${stats.get('var_95', 0):,.0f}
• Worst Case (5%): ${stats.get('var_5', 0):,.0f}"""
                
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'message': summary_msg,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
        
        # Data information panel
        if st.session_state.get('data_uploaded', False):
            st.markdown("### Data Information")
            df = st.session_state.get('raw_data')
            if df is None:
                df = st.session_state.get('cleaned_data')
            if df is not None:
                st.write(f"**Rows:** {df.shape[0]:,}")
                st.write(f"**Columns:** {df.shape[1]}")
                column_preview = ', '.join(df.columns[:3])
                if len(df.columns) > 3:
                    column_preview += '...'
                st.write(f"**Columns:** {column_preview}")


def handle_file_upload():
    """Manage file upload interface and processing"""
    st.markdown("## Data Upload Center")
    
    uploaded_file = st.file_uploader(
        "Upload your financial data (CSV format)", 
        type=['csv'],
        help="Upload a CSV file with financial data including columns like Revenue, Expenses, Net_Income, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Read and store the uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_data = df
            st.session_state.data_uploaded = True
            
            st.success(f"File uploaded successfully! Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Show data preview
            with st.expander("Data Preview", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
                
                # Display column type information
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Numeric Columns:**")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    st.write(", ".join(numeric_cols) if numeric_cols else "None detected")
                
                with col2:
                    st.write("**Text Columns:**")
                    text_cols = df.select_dtypes(include=['object']).columns.tolist() 
                    st.write(", ".join(text_cols) if text_cols else "None detected")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV with proper formatting.")


def build_chat_interface():
    """Create the chat interface for user interaction"""
    st.markdown("## Chat with Your AI Financial Agent")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.get('chat_history'):
            for chat in st.session_state.chat_history:
                if chat['type'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You ({chat['timestamp']}):</strong><br>
                        {chat['message']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>AI Agent ({chat['timestamp']}):</strong><br>
                        <div style="white-space: pre-wrap;">{chat['message']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Start a conversation with your AI agent! Try uploading data first, then ask questions.")
    
    # Chat input field
    user_input = st.chat_input("Ask me anything about your financial data...")
    
    if user_input:
        # Process user input through the agent
        agent = FinancialAgent()
        agent.process_user_query(user_input)
        st.rerun()


def create_dashboard():
    """Build the results dashboard when simulation is complete"""
    if st.session_state.get('simulation_completed', False) and st.session_state.get('simulation_results') is not None:
        st.markdown("## Financial Analysis Dashboard")
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        stats = st.session_state.get('simulation_stats', {})
        
        with col1:
            st.metric(
                "Expected Net Income",
                f"${stats.get('ni_mean', 0):,.0f}",
                f"{stats.get('roi_mean', 0):.1%} ROI"
            )
        
        with col2:
            st.metric(
                "Profit Probability", 
                f"{stats.get('prob_profit', 0):.1f}%",
                f"{stats.get('prob_high_return', 0):.1f}% high return"
            )
        
        with col3:
            st.metric(
                "Risk Level",
                f"{stats.get('prob_loss', 0):.1f}%",
                "Loss probability"
            )
        
        with col4:
            st.metric(
                "Value at Risk (5%)",
                f"${abs(stats.get('var_5', 0)):,.0f}",
                "Worst 5% scenarios"
            )
        
        # Show scenario distribution
        with st.expander("Scenario Distribution", expanded=True):
            scenario_data = st.session_state.get('scenario_outcomes', {})
            if scenario_data:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create bar chart of scenarios
                    scenario_df = pd.DataFrame(list(scenario_data.items()), columns=['Scenario', 'Count'])
                    st.bar_chart(scenario_df.set_index('Scenario'))
                
                with col2:
                    st.write("**Scenario Breakdown:**")
                    for scenario, count in scenario_data.items():
                        percentage = (count / 10)
                        st.write(f"• {scenario.title()}: {count} ({percentage:.1f}%)")


def show_help_section():
    """Display help and usage information"""
    st.markdown("""
    ## How to Use the Agentic AI Finance Analyst
    
    ### Quick Start Guide:
    
    1. **Upload Data**: Upload your financial CSV file with columns like Revenue, Expenses, Net_Income
    2. **Auto-Clean**: The AI agent will automatically clean and prepare your data
    3. **Simulate**: Ask the agent to "run simulation" for Monte Carlo analysis
    4. **Get Stories**: Ask for "financial stories" to get narrative scenario analysis
    
    ### Data Requirements:
    
    Your CSV should ideally contain:
    - **Revenue/Sales** data (different periods)
    - **Expenses/Costs** data 
    - **Net Income/Profit** data
    - Any additional financial metrics
    
    ### Agent Capabilities:
    
    - **Smart Data Cleaning**: Removes formatting, converts currencies, handles missing data
    - **Monte Carlo Simulation**: Generates 1,000 financial scenarios using statistical modeling
    - **Risk Analysis**: Calculates VaR, probability distributions, and risk metrics
    - **Financial Storytelling**: Creates engaging narratives about your business future
    
    ### Sample Questions to Ask:
    
    - "Clean my data and run analysis"
    - "Tell me a financial story about my business"
    - "What are the risks and opportunities?"
    - "Run Monte Carlo simulation"
    - "Show me different scenarios"
    
    ### Understanding Scenarios:
    
    - **Excellent**: >20% ROI - Outstanding performance
    - **Good**: 10-20% ROI - Solid profitable growth  
    - **Neutral**: 0-10% ROI - Modest positive outcomes
    - **Poor**: -10-0% ROI - Challenging but recoverable
    - **Crisis**: <-10% ROI - Significant challenges requiring action
    
    ### Troubleshooting:
    
    - If data cleaning fails, check for proper CSV formatting
    - Ensure your data has recognizable financial column names
    - Use the "Reset Session" button to start over if needed
    - The agent will guide you through each step automatically
    """)


def main():
    """Main application function"""
    # Setup page configuration and styling
    setup_page_config()
    load_custom_css()
    
    # Initialize the financial agent
    agent = FinancialAgent()
    
    # Render main components
    render_main_header()
    create_sidebar()
    
    # Create main tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data & Chat", "Dashboard", "Help"])
    
    with tab1:
        # File upload and chat interface
        handle_file_upload()
        st.markdown("---")
        build_chat_interface()
    
    with tab2:
        # Results dashboard
        create_dashboard()
        
        if not st.session_state.get('simulation_completed', False):
            st.info("Dashboard will be available after completing the Monte Carlo simulation.")
    
    with tab3:
        # Help and documentation
        show_help_section()


if __name__ == "__main__":
    main()
