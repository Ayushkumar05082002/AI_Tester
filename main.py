import asyncio
import os
import streamlit as st
from textwrap import dedent
from dotenv import load_dotenv
import yaml

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Page config
st.set_page_config(page_title="Puppeteer MCP Agent", page_icon="🌐", layout="wide")

# Title and description
st.markdown("<h1 class='main-header'>🌐 Puppeteer MCP Agent</h1>", unsafe_allow_html=True)
st.markdown("Interact with a powerful web browsing agent that can navigate and interact with websites")

# Load environment variables
load_dotenv()

def update_filesystem_config(file_location):
    """Update the mcp_agent.config.yaml file with the new filesystem path"""
    config_path = "mcp_agent.config.yaml"
    
    try:
        # Read current config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update filesystem server args with new path
        if 'mcp' in config and 'servers' in config['mcp'] and 'filesystem' in config['mcp']['servers']:
            # Keep the node command and first arg (the js file path), update the directory path
            current_args = config['mcp']['servers']['filesystem']['args']
            if len(current_args) >= 2:
                # Update only the directory path (second argument)
                config['mcp']['servers']['filesystem']['args'][1] = file_location
            else:
                # If args structure is unexpected, rebuild it
                config['mcp']['servers']['filesystem']['args'] = [
                    current_args[0] if current_args else "C:/Users/Ayush Singh/AppData/Roaming/npm/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js",
                    file_location
                ]
        
        # Write updated config back
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            
        return True
    except Exception as e:
        st.error(f"Failed to update config file: {str(e)}")
        return False

# Model selector and settings in sidebar
with st.sidebar:
    st.markdown("### Model Settings")
    selected_model = st.selectbox(
        "Select OpenAI Model",
        [
            "gpt-4.1-2025-04-14",
            "o4-mini-2025-04-16",
            "gpt-4-1106-preview",  # GPT-4 Turbo
            "gpt-4",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo"
        ],
        help="Choose the OpenAI model to use for responses"
    )
    # Store selected model in session state
    st.session_state['model'] = selected_model
    
    selected_temp = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make the output more creative, lower values make it more focused"
    )
    # Store temperature in session state
    st.session_state['temperature'] = selected_temp
    
    st.markdown("### File System Settings")
    file_location = st.text_input(
        "Working Directory",
        value=os.getcwd(),
        placeholder="Enter the file system path for operations",
        help="Specify the directory where the filesystem server will operate"
    )
    # Store file location in session state
    st.session_state['file_location'] = file_location
    
    # Display current working directory info
    st.caption(f"Current: {os.path.abspath(file_location) if file_location else 'Not set'}")
    
    # Show status of file location
    if st.session_state.get('last_file_location'):
        if st.session_state.last_file_location == file_location:
            st.success(f"✅ Active: {st.session_state.last_file_location}")
        else:
            st.warning("⚠️ Location changed - will update on next run")
    else:
        st.info("🔄 Will be set on first run")

# Query input
query = st.text_area(
    "Your Command",
    placeholder="Ask the agent to navigate to websites and interact with them")

# Initialize app and agent
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mcp_app = MCPApp(name="streamlit_mcp_agent")
    st.session_state.mcp_context = None
    st.session_state.mcp_agent_app = None
    st.session_state.browser_agent = None
    st.session_state.llm = None
    st.session_state.loop = asyncio.new_event_loop()
    st.session_state.last_file_location = None
    asyncio.set_event_loop(st.session_state.loop)

# Setup function that runs only once
async def setup_agent():
    current_file_location = st.session_state.get('file_location', os.getcwd())
    
    # Check if we need to reinitialize due to file location change
    if (st.session_state.initialized and 
        st.session_state.last_file_location != current_file_location):
        # Clean up existing connections
        if st.session_state.mcp_context:
            try:
                await st.session_state.mcp_context.__aexit__(None, None, None)
            except:
                pass
        st.session_state.initialized = False
        st.session_state.mcp_context = None
        st.session_state.mcp_agent_app = None
        st.session_state.browser_agent = None
        st.session_state.llm = None
    
    if not st.session_state.initialized:
        try:
            # Validate and prepare file location
            if not os.path.exists(current_file_location):
                try:
                    os.makedirs(current_file_location, exist_ok=True)
                except Exception as e:
                    return f"Error: Cannot create directory '{current_file_location}': {str(e)}"
            
            # Update config file with new filesystem path
            if not update_filesystem_config(current_file_location):
                return "Error: Failed to update configuration file"
            
            # Store the current file location
            st.session_state.last_file_location = current_file_location
            
            # Note: The filesystem server working directory is now configured in mcp_agent.config.yaml
            # with the user-specified path
            
            # Create context manager and store it in session state
            st.session_state.mcp_context = st.session_state.mcp_app.run()
            st.session_state.mcp_agent_app = await st.session_state.mcp_context.__aenter__()
              # Create and initialize agent            
            st.session_state.browser_agent = Agent(
                name="browser",
                instruction=f"""You are an efficient and precise automated assistant specialized in web automation, file operations, and Excel manipulation. Follow instructions exactly and provide detailed, structured responses.

                    IMPORTANT: The filesystem server is configured to work in: {current_file_location}
                    All file operations will be relative to this directory.

                    Web Automation Protocol:
                    1. Precise Navigation:
                        - Execute exact browser actions (click, type, scroll) with explicit selectors
                        - Validate successful navigation and actions
                        - Handle dynamic content and loading states
                        - Default to www.lastmileai.dev unless specified otherwise
                    
                    2. Data Extraction & Processing:
                        - Extract data using optimized selectors
                        - Capture targeted screenshots of relevant elements
                        - Generate structured summaries with clear hierarchies
                        - Process and validate extracted information
                    
                    File System Operations:
                    1. Core Functions:
                        - Read/write operations with proper error handling
                        - Directory manipulation with validation
                        - File search with multiple criteria
                        - Change monitoring with event tracking
                        - Working directory: {current_file_location}
                    
                    2. Best Practices:
                        - File operations are relative to: {current_file_location}
                        - Implement error handling and recovery
                        - Verify operations before confirming
                        - Maintain data integrity checks
                    
                    Excel Processing:
                    1. Data Operations:
                        - Structured read/write operations
                        - Formula validation and optimization
                        - Multi-sheet coordination
                        - Data validation implementation
                    
                    2. Workbook Management:
                        - Efficient worksheet organization
                        - Template-based creation
                        - Cross-reference handling
                        - Format standardization
                    
                    Response Protocol:
                    1. Always confirm understanding of the task
                    2. Break complex operations into clear steps
                    3. Provide progress updates for long operations
                    4. Report success/failure with specific details
                    5. Suggest optimizations when applicable
                    
                    Error Handling:
                    1. Implement proper try-catch blocks
                    2. Provide detailed error messages
                    3. Attempt recovery when possible
                    4. Log important operations
                    
                    Remember: All file operations are rooted in {current_file_location}, handle errors gracefully, and maintain data integrity.""",
                server_names=["puppeteer", "filesystem", "excel", "playwright"]
            )
            
            # Initialize agent and attach LLM
            await st.session_state.browser_agent.initialize()
            st.session_state.llm = await st.session_state.browser_agent.attach_llm(OpenAIAugmentedLLM)
            
            # List tools once
            logger = st.session_state.mcp_agent_app.logger
            tools = await st.session_state.browser_agent.list_tools()
            logger.info("Tools available:", data=tools)
            
            # Mark as initialized
            st.session_state.initialized = True
                
        except Exception as e:
            return f"Error during initialization: {str(e)}"
    return None

# Main function to run agent
async def run_mcp_agent(message):
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: OpenAI API key not provided"
    
    try:
        # Make sure agent is initialized
        error = await setup_agent()
        if error:
            return error
        
        # Generate response with optimized parameters and user-selected model
        result = await st.session_state.llm.generate_str(
            message=message,
            request_params=RequestParams(
                use_history=True,
                temperature=st.session_state.get('temperature', 0.7),  # Use UI temperature or default
                max_tokens=2000,  # Increased for more detailed responses
                frequency_penalty=0.3,  # Reduce repetition
                presence_penalty=0.3,  # Encourage diverse responses
                model=st.session_state.get('model', 'gpt-4-1106-preview')  # Use UI model or default
            )
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Run button
if st.button("🚀 Run Command", type="primary", use_container_width=True):
    with st.spinner("Processing your request..."):
        result = st.session_state.loop.run_until_complete(run_mcp_agent(query))
    
    # Display results
    st.markdown("### Response")
    st.markdown(result)

# Display help text for first-time users
if 'result' not in locals():
    st.markdown(
        """<div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        </div>""", 
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.write("Note: The agent uses Puppeteer to control a real browser.")