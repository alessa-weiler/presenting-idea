
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
from pyvis.network import Network
import streamlit.components.v1 as components
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Therapy Conversation Network",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #111;
        color: white;
    }
    .stApp {
        background-color: #111;
    }
    .metric-card {
        background-color: #222;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #888;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: white;
        font-size: 42px;
        font-weight: bold;
    }
    .progress-container {
        width: 100%;
        background-color: #333;
        border-radius: 5px;
        margin-top: 10px;
    }
    .progress-blue {
        background-color: #3b82f6;
        height: 6px;
        border-radius: 5px;
    }
    .progress-gold {
        background: linear-gradient(to right, #555, #f0c14b);
        height: 6px;
        border-radius: 5px;
    }
    .insight-card {
        background-color: #222;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 10px;
    }
    .insight-title {
        color: #888;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .insight-content {
        color: white;
        font-size: 14px;
    }
    .top-connections {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }
    h1, h2, h3, p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("Therapy Conversation Network")

# Define data
@st.cache_data
def get_data():
    connections = [
        {"id": "Victoria", "sentiment": 85, "similarity": 72, 
         "summary": "The conversation begins with Victoria sharing news about the Europe-wide Female Founder Office Hours program..."},
        {"id": "Yasmin", "sentiment": 90, "similarity": 85, 
         "summary": "Yasmin shares her career advancement to Principal at 4impact capital and her upcoming attendance at Climate Tech Connect..."},
        {"id": "Eline", "sentiment": 88, "similarity": 80, 
         "summary": "Eline shares her passion for supporting world-changing entrepreneurs through early-stage investments at Rubio..."},
        {"id": "Cesare", "sentiment": 78, "similarity": 68, 
         "summary": "Cesare enthusiastically shares his experience co-organizing the first Road to START Summit Milano..."},
        {"id": "Kristi", "sentiment": 93, "similarity": 88, 
         "summary": "Their conversation begins with Kristi sharing her diverse operational expertise across multiple industries..."},
        {"id": "Marie-Sophie", "sentiment": 87, "similarity": 75, 
         "summary": "Marie-Sophie enthusiastically shares her upcoming speaking engagement at the unlock VC Summit for women in venture capital..."}
    ]
    return connections

connections = get_data()

# Create main layout with columns
col1, col2 = st.columns([7, 3])

# Function to generate the network visualization using pyvis
def generate_network():
    # Create a pyvis network
    net = Network(notebook=True, bgcolor="#111111", font_color="white", height="600px", width="100%")
    
    # Configure physics options
    net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150, spring_strength=0.05, damping=0.09)
    
    # Add central node (Alessa)
    net.add_node("Alessa", label="Alessa", title="Alessa", size=30, color="#ff6633")
    
    # Add connection nodes
    for connection in connections:
        net.add_node(connection["id"], label=connection["id"], title=connection["id"], 
                    size=20, color="#33cc99")
        # Add edge to Alessa with weight based on average score
        weight = (connection["sentiment"] + connection["similarity"]) / 2
        net.add_edge("Alessa", connection["id"], value=weight/10, title=f"Sentiment: {connection['sentiment']}, Similarity: {connection['similarity']}")
    
    # Add background nodes for visual effect
    groups = ['#6666cc', '#66cccc', '#cc66cc', '#cc6666']
    
    # Create 3 clusters of background nodes
    for i in range(100):
        group_color = random.choice(groups)
        node_id = f"node-{i}"
        # Random position from center (Alessa)
        net.add_node(node_id, size=3+random.random()*3, color=group_color, 
                    hidden=False, opacity=0.6)
        # Connect to Alessa with very thin edge
        net.add_edge("Alessa", node_id, width=0.1, color="#333333", opacity=0.2)
        
        # Add some random connections between background nodes
        if random.random() > 0.7:
            target_id = f"node-{random.randint(0, 99)}"
            if target_id != node_id:
                net.add_edge(node_id, target_id, width=0.1, color="#333333", opacity=0.1)
    
    # Generate the HTML file
    path = "network.html"
    net.save_graph(path)
    
    # Read the HTML content
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    
    return html

# Function to display metrics card
def display_metric_card(label, value, percentage, progress_class, info_icon=True):
    html = f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between;">
            <div class="metric-label">{label}</div>
            {"<span style='cursor: pointer;'>ⓘ</span>" if info_icon else ""}
        </div>
        <div class="metric-value">{value}</div>
        <div class="progress-container">
            <div class="{progress_class}" style="width: {percentage}%;"></div>
        </div>
    </div>
    """
    return html

# Function to display insight card
def display_insight_card(title, content):
    html = f"""
    <div class="insight-card">
        <div class="insight-title">{title}</div>
        <div class="insight-content">{content}</div>
    </div>
    """
    return html

# Function to display top connections
def display_top_connections():
    # Sort connections by sentiment
    sorted_connections = sorted(connections, key=lambda x: x["sentiment"], reverse=True)
    top_2 = sorted_connections[:2]
    
    html = """
    <div class="insight-card">
        <div class="insight-title">Top Connections by Sentiment</div>
    """
    
    for conn in top_2:
        html += f"""
        <div class="top-connections">
            <span>{conn['id']}</span>
            <span style="font-weight: bold;">{conn['sentiment']}</span>
        </div>
        """
    
    html += "</div>"
    return html

# Session state for selected node
if 'selected_node' not in st.session_state:
    st.session_state.selected_node = None

# Display network visualization in column 1
with col1:
    # Generate the network HTML
    network_html = generate_network()
    
    # Display the network using components
    components.html(network_html, height=600)
    
# Display metrics and insights in column 2
with col2:
    # Dropdown to simulate node selection
    selected_name = st.selectbox(
        "Select a connection to view details:",
        ["None"] + [conn["id"] for conn in connections]
    )
    
    if selected_name != "None":
        # Find the selected connection data
        selected_node = next((conn for conn in connections if conn["id"] == selected_name), None)
        st.session_state.selected_node = selected_node
    else:
        st.session_state.selected_node = None
    
    # Display sentiment score
    sentiment_value = st.session_state.selected_node["sentiment"] if st.session_state.selected_node else 80
    st.markdown(display_metric_card("Sentiment score", sentiment_value, sentiment_value, "progress-blue"), unsafe_allow_html=True)
    
    # Display similarity score
    similarity_value = st.session_state.selected_node["similarity"] if st.session_state.selected_node else 59
    st.markdown(display_metric_card("Similarity score", similarity_value, similarity_value, "progress-gold"), unsafe_allow_html=True)
    
    # Display relationship insights
    if st.session_state.selected_node:
        node = st.session_state.selected_node
        insight_text = f"Relationship with {node['id']} shows high {'sentiment' if node['sentiment'] > 85 else 'potential'} and {'strong similarity' if node['similarity'] > 80 else 'growing connection'}."
    else:
        insight_text = "Select a connection to see relationship insights."
    
    st.markdown(display_insight_card("Relationship Insights", insight_text), unsafe_allow_html=True)
    
    # Display recommended actions
    if st.session_state.selected_node:
        node = st.session_state.selected_node
        action_text = f"Focus on strengthening communication with {node['id']}. Schedule a follow-up conversation to discuss shared interests."
    else:
        action_text = "Select a connection to see recommended actions."
    
    st.markdown(display_insight_card("Recommended Actions", action_text), unsafe_allow_html=True)
    
    # Display top connections
    st.markdown(display_top_connections(), unsafe_allow_html=True)

# Add a section below the visualization for conversation details
st.subheader("Conversation Summary")

if st.session_state.selected_node:
    st.write(st.session_state.selected_node["summary"])
else:
    st.write("Select a connection to see the conversation summary.")

# Add an expandable section for additional information
with st.expander("About This Tool"):
    st.write("""
    This tool visualizes the relationships and conversation quality between the therapist and potential clients. 
    
    The network graph shows:
    - The central node represents the main person (therapist)
    - Connected nodes represent conversations with different clients
    - The thickness of connections represents the strength of the relationship
    - Metrics include sentiment analysis and similarity scores
    
    This visualization helps identify the best potential matches and provides insights to improve the therapy experience.
    """)

# Add sidebar for additional controls
st.sidebar.title("Visualization Controls")
st.sidebar.subheader("Display Settings")

# Add controls
show_labels = st.sidebar.checkbox("Show Node Labels", value=True)
min_weight = st.sidebar.slider("Minimum Connection Strength", 65, 95, 70, 5)
emphasis = st.sidebar.radio("Emphasize Connections By:", ["Average", "Sentiment", "Similarity"])

st.sidebar.markdown("---")

# Add a download button for example
def to_excel():
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Convert connections to DataFrame
    df = pd.DataFrame(connections)
    df.to_excel(writer, sheet_name='ConnectionData', index=False)
    
    writer.close()
    processed_data = output.getvalue()
    return processed_data

excel_data = to_excel()
b64 = base64.b64encode(excel_data).decode()
href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="network_data.xlsx">Download Data as Excel</a>'
st.sidebar.markdown(href, unsafe_allow_html=True)

# Add a feedback section
st.sidebar.markdown("---")
st.sidebar.subheader("Feedback")
feedback = st.sidebar.text_area("Share your thoughts on this tool:")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")
