
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Conversation Analysis - Bubble Chart", layout="wide")

st.title("Conversation Analysis - Bubble Chart")

# Function to load data
@st.cache_data
def load_data():
    # You can replace this with your file path or upload functionality
    # For this example, I'm using hardcoded data
    data = {
        "person1": ["Alessa", "Alessa", "Alessa", "Alessa", "Alessa", "Alessa"],
        "person2": ["Victoria", "Yasmin", "Eline", "Cesare", "Kristi", "Marie-Sophie"],
        "sentiment": [85, 90, 88, 78, 93, 87],
        "similarity": [72, 85, 80, 68, 88, 75],
        "summary": [
            "The conversation begins with Victoria sharing news about the Europe-wide Female Founder Office Hours program...",
            "Yasmin shares her career advancement to Principal at 4impact capital and her upcoming attendance at Climate Tech Connect...",
            "Eline shares her passion for supporting world-changing entrepreneurs through early-stage investments at Rubio...",
            "Cesare enthusiastically shares his experience co-organizing the first Road to START Summit Milano...",
            "Their conversation begins with Kristi sharing her diverse operational expertise across multiple industries...",
            "Marie-Sophie enthusiastically shares her upcoming speaking engagement at the unlock VC Summit for women in venture capital..."
        ]
    }
    return pd.DataFrame(data)

# Load data
df = load_data()

# Sidebar with options
st.sidebar.header("Network Graph Settings")
weight_option = st.sidebar.selectbox(
    "Edge weight based on:",
    ["sentiment", "similarity", "average"]
)

min_threshold = st.sidebar.slider(
    "Minimum connection strength to show:", 
    min_value=0, 
    max_value=100, 
    value=50
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Network Graph", "Data Table", "Summary View"])

with tab1:
    st.header("Network Visualization")
    
    # Create a network graph
    G = nx.Graph()
    
    # Get all unique persons
    all_persons = list(set(df['person1'].tolist() + df['person2'].tolist()))
    
    # Add nodes
    for person in all_persons:
        G.add_node(person)
    
    # Add edges with weights
    for _, row in df.iterrows():
        if weight_option == "sentiment":
            weight = row['sentiment']
        elif weight_option == "similarity":
            weight = row['similarity']
        else:  # average
            weight = (row['sentiment'] + row['similarity']) / 2
        
        # Only add edges above the threshold
        if weight >= min_threshold:
            G.add_edge(row['person1'], row['person2'], 
                      weight=weight/100,  # Normalize to 0-1
                      sentiment=row['sentiment'],
                      similarity=row['similarity'],
                      summary=row['summary'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get edge weights for coloring
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize colors based on weights
    norm = Normalize(vmin=min_threshold/100, vmax=1)
    cmap = cm.Blues
    
    # Draw edges with varying colors based on weight
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=edges,
        width=[w*5 for w in weights],  # Scale width by weight
        edge_color=[cmap(norm(w)) for w in weights],
        alpha=0.7
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=['Alessa'],  # Central node
        node_size=800,
        node_color='lightgreen',
        alpha=0.9
    )
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n in G.nodes() if n != 'Alessa'],
        node_size=500,
        node_color='skyblue',
        alpha=0.9
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_weight='bold')
    
    # Add edge labels if there are few enough edges
    if len(edges) <= 10:
        if weight_option == "sentiment":
            edge_labels = {(u, v): f"{G[u][v]['sentiment']}" for u, v in edges}
        elif weight_option == "similarity":
            edge_labels = {(u, v): f"{G[u][v]['similarity']}" for u, v in edges}
        else:
            edge_labels = {(u, v): f"{int((G[u][v]['sentiment'] + G[u][v]['similarity'])/2)}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.axis('off')
    st.pyplot(fig)
    
    # Add color bar for reference
    fig2, ax2 = plt.subplots(figsize=(6, 1))
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                    cax=ax2, orientation='horizontal')
    
    if weight_option == "sentiment":
        cb.set_label('Sentiment Score (normalized)')
    elif weight_option == "similarity":
        cb.set_label('Similarity Score (normalized)')
    else:
        cb.set_label('Average of Sentiment and Similarity (normalized)')
    
    st.pyplot(fig2)

with tab2:
    st.header("Conversation Data")
    st.dataframe(df[['person1', 'person2', 'sentiment', 'similarity']], use_container_width=True)

with tab3:
    st.header("Detailed Conversation Summaries")
    
    # Create an expandable section for each conversation
    for _, row in df.iterrows():
        with st.expander(f"{row['person1']} and {row['person2']} (Sentiment: {row['sentiment']}, Similarity: {row['similarity']})"):
            st.markdown(row['summary'])

# Add additional insights
st.header("Network Insights")

# Calculate and display metrics
if G.number_of_edges() > 0:
    strongest_connection = max([(u, v) for u, v in G.edges()], 
                              key=lambda x: G[x[0]][x[1]]['weight'])
    
    p1, p2 = strongest_connection
    
    st.markdown(f"### Strongest Connection: {p1} and {p2}")
    st.markdown(f"- Sentiment: {G[p1][p2]['sentiment']}")
    st.markdown(f"- Similarity: {G[p1][p2]['similarity']}")
    
    # Show summary for strongest connection
    strongest_row = df[(df['person1'] == p1) & (df['person2'] == p2) | 
                      (df['person1'] == p2) & (df['person2'] == p1)].iloc[0]
    
    st.markdown("#### Summary:")
    st.markdown(strongest_row['summary'])

# Add instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
- Use the dropdown to change the metric shown on the edges
- Adjust the strength threshold to filter connections
- Hover over nodes and edges for more information
- View the data table for raw scores
- Expand the summaries for detailed conversation analyses
""")
