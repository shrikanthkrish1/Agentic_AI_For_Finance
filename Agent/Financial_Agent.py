import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
from Tools import DataCleaningTool, MonteCarloTool, FinancialStoryteller


class FinancialAgent:
    """Main AI agent that coordinates the financial analysis workflow"""
    
    def __init__(self):
        self.tools = {
            'data_cleaning': DataCleaningTool(),
            'monte_carlo': MonteCarloTool(),
            'storyteller': FinancialStoryteller()
        }
        self.status = 'ready'
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Set up session state variables if they don't exist"""
        defaults = {
            'raw_data': None,
            'cleaned_data': None,
            'simulation_results': None,
            'data_uploaded': False,
            'data_cleaned': False,
            'simulation_completed': False,
            'chat_history': [],
            'cleaning_log': [],
            'scenario_outcomes': {},
            'simulation_stats': {},
            'agent_status': 'ready'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def process_user_query(self, user_query: str) -> str:
        """Main method to process user requests and route to appropriate tools"""
        
        query_lower = user_query.lower()
        
        # Add user message to chat history
        self._add_to_chat('user', user_query)
        
        # Determine what action to take based on current state and user input
        if not st.session_state.data_uploaded:
            response = self._handle_no_data_state()
        elif not st.session_state.data_cleaned:
            response = self._handle_uncleaned_data_state()
        elif not st.session_state.simulation_completed:
            response = self._handle_pre_simulation_state(query_lower)
        else:
            response = self._handle_post_simulation_state(query_lower)
        
        # Add agent response to chat history
        self._add_to_chat('assistant', response)
        
        self.status = 'ready'
        st.session_state.agent_status = 'ready'
        
        return response
    
    def _handle_no_data_state(self) -> str:
        """Handle queries when no data has been uploaded yet"""
        return ("Welcome to the Agentic AI Finance Analyst! Please upload your financial data CSV file first, "
                "then I'll help you clean the data, run Monte Carlo simulations, and tell you fascinating financial "
                "stories about your business future!")
    
    def _handle_uncleaned_data_state(self) -> str:
        """Handle queries when data exists but hasn't been cleaned"""
        response = "I see you've uploaded data! Let me start by cleaning and preparing your financial data for analysis..."
        st.session_state.agent_status = 'cleaning'
        
        # Run data cleaning
        cleaning_result = self.tools['data_cleaning'].clean_financial_data(st.session_state.raw_data)
        
        if cleaning_result['success']:
            st.session_state.cleaned_data = cleaning_result['data']
            st.session_state.cleaning_log = cleaning_result['log']
            st.session_state.data_cleaned = True
            response += "\n\n" + cleaning_result['message']
        else:
            response += "\n\n" + cleaning_result['message']
        
        return response
    
    def _handle_pre_simulation_state(self, query_lower: str) -> str:
        """Handle queries after data is cleaned but simulation hasn't run"""
        if any(keyword in query_lower for keyword in ['simulate', 'monte carlo', 'analysis', 'run', 'calculate']):
            response = "Time for the Monte Carlo simulation! Let me analyze your cleaned data and generate 1,000 different financial scenarios..."
            st.session_state.agent_status = 'simulating'
            
            # Run simulation
            sim_result = self.tools['monte_carlo'].run_simulation(st.session_state.cleaned_data)
            
            if sim_result['success']:
                st.session_state.simulation_results = sim_result['results']
                st.session_state.scenario_outcomes = sim_result['scenarios']
                st.session_state.simulation_stats = sim_result['stats']
                st.session_state.simulation_completed = True
                response += "\n\n" + sim_result['message']
            else:
                response += "\n\n" + sim_result['message']
            
            return response
        else:
            return "Your data is cleaned and ready! Ask me to 'run simulation' or 'analyze my data' to proceed with Monte Carlo analysis."
    
    def _handle_post_simulation_state(self, query_lower: str) -> str:
        """Handle queries after simulation is complete"""
        if any(keyword in query_lower for keyword in ['story', 'tell', 'narrative', 'scenario', 'future']):
            response = "Let me weave the financial scenarios into an engaging story for you..."
            st.session_state.agent_status = 'storytelling'
            
            # Generate story
            story_result = self.tools['storyteller'].create_story(
                st.session_state.simulation_results,
                st.session_state.scenario_outcomes,
                st.session_state.simulation_stats
            )
            
            if story_result['success']:
                response += "\n\n" + story_result['story']
            else:
                response += "\n\n" + story_result['message']
            
            return response
            
        elif any(keyword in query_lower for keyword in ['simulate', 'monte carlo', 'analysis', 'run again']):
            response = "Running Monte Carlo simulation again..."
            st.session_state.agent_status = 'simulating'
            
            sim_result = self.tools['monte_carlo'].run_simulation(st.session_state.cleaned_data)
            
            if sim_result['success']:
                st.session_state.simulation_results = sim_result['results']
                st.session_state.scenario_outcomes = sim_result['scenarios']
                st.session_state.simulation_stats = sim_result['stats']
                response += "\n\n" + sim_result['message']
            else:
                response += "\n\n" + sim_result['message']
            
            return response
            
        elif any(keyword in query_lower for keyword in ['clean', 'prepare', 'fix data']):
            response = "Re-cleaning your data..."
            st.session_state.agent_status = 'cleaning'
            
            cleaning_result = self.tools['data_cleaning'].clean_financial_data(st.session_state.raw_data)
            
            if cleaning_result['success']:
                st.session_state.cleaned_data = cleaning_result['data']
                st.session_state.cleaning_log = cleaning_result['log']
                response += "\n\n" + cleaning_result['message']
            else:
                response += "\n\n" + cleaning_result['message']
            
            return response
            
        elif any(keyword in query_lower for keyword in ['help', 'what can you do', 'commands']):
            return self._get_help_message()
        
        elif any(keyword in query_lower for keyword in ['summary', 'results', 'show']):
            return self._get_quick_summary()
        
        else:
            # Default intelligent response
            return ("I'm your AI Financial Analyst! Here's what I can help you with:\n\n"
                   "Available Actions:\n"
                   "• 'Tell me a financial story' - Get narrative scenarios about your business future\n"
                   "• 'Run simulation again' - Re-run Monte Carlo analysis\n"
                   "• 'Show me the results' - Display current simulation summary\n"
                   "• 'Clean data again' - Re-process your uploaded data\n"
                   "• 'Help' - Show all available commands\n\n"
                   "Pro Tip: Try asking for a 'financial story' to get engaging narratives about your business scenarios!")
    
    def _get_quick_summary(self) -> str:
        """Generate a quick summary of simulation results"""
        if not st.session_state.simulation_completed or not st.session_state.simulation_stats:
            return "No simulation results available yet. Please run a Monte Carlo simulation first."
        
        stats = st.session_state.simulation_stats
        return f"""QUICK SUMMARY:

• Expected Net Income: ${stats.get('ni_mean', 0):,.0f}
• Profit Probability: {stats.get('prob_profit', 0):.1f}%
• Risk Level: {stats.get('prob_loss', 0):.1f}% chance of loss
• Best Case (95%): ${stats.get('var_95', 0):,.0f}
• Worst Case (5%): ${stats.get('var_5', 0):,.0f}"""
    
    def _get_help_message(self) -> str:
        """Provide comprehensive help information"""
        return """AGENTIC AI FINANCE ANALYST - COMMAND CENTER

PRIMARY WORKFLOW:
1. Upload CSV data → 2. Data gets auto-cleaned → 3. Run Monte Carlo simulation → 4. Get financial stories

AVAILABLE COMMANDS:

Data Management:
• 'Clean my data' - Process and prepare financial data
• 'Fix data issues' - Re-run data cleaning process
• 'Show data info' - Display dataset information

Analysis Commands:
• 'Run simulation' - Execute Monte Carlo analysis
• 'Analyze my data' - Comprehensive financial analysis
• 'Calculate scenarios' - Generate probability scenarios
• 'Run Monte Carlo' - Alternative simulation trigger

Storytelling Commands:
• 'Tell me a financial story' - Narrative scenario analysis
• 'Show me scenarios' - Story-based future projections
• 'What's my business future?' - Comprehensive storytelling
• 'Create financial narrative' - Custom story generation

Utility Commands:
• 'Help' - Show this command center
• 'Status' - Check current analysis state
• 'What can you do?' - Capability overview
• 'Reset' - Clear session and start over

QUICK START: Just upload your CSV file and I'll guide you through the entire process automatically!"""
    
    def _add_to_chat(self, message_type: str, message: str):
        """Add message to chat history"""
        st.session_state.chat_history.append({
            'type': message_type,
            'message': message,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and progress"""
        return {
            'status': self.status,
            'data_uploaded': st.session_state.get('data_uploaded', False),
            'data_cleaned': st.session_state.get('data_cleaned', False),
            'simulation_completed': st.session_state.get('simulation_completed', False),
            'chat_count': len(st.session_state.get('chat_history', []))
        }
    
    def reset_session(self):
        """Reset all session state variables"""
        keys_to_clear = [
            'raw_data', 'cleaned_data', 'simulation_results',
            'data_uploaded', 'data_cleaned', 'simulation_completed',
            'chat_history', 'cleaning_log', 'scenario_outcomes', 
            'simulation_stats', 'agent_status'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        self.initialize_session_state()
        self.status = 'ready'
