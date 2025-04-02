
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
# Load data
df = load_data()

# Sidebar options
st.sidebar.header("Visualization Settings")
x_axis = st.sidebar.selectbox(
    "X-Axis Metric:",
    ["sentiment", "similarity"],
    index=1  # Default to similarity
)

y_axis = st.sidebar.selectbox(
    "Y-Axis Metric:",
    ["sentiment", "similarity"],
    index=0  # Default to sentiment
)

# Calculate size (based on average of sentiment and similarity)
df['bubble_size'] = (df['sentiment'] + df['similarity']) / 2
min_size, max_size = 100, 1000  # Min and max bubble sizes

# Create tabs
tab1, tab2 = st.tabs(["Bubble Chart", "Summary View"])

with tab1:
    st.header("Bubble Chart Visualization")
    st.subheader(f"Conversation compatibility between Alessa and others")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize bubble sizes
    sizes = df['bubble_size'].values
    normalized_sizes = [(size - df['bubble_size'].min()) / (df['bubble_size'].max() - df['bubble_size'].min()) 
                       * (max_size - min_size) + min_size for size in sizes]
    
    # Choose different colors for each bubble
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    
    # Create scatter plot
    scatter = ax.scatter(
        df[x_axis], 
        df[y_axis],
        s=normalized_sizes,
        c=colors,
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    
    # Add labels next to each bubble
    for i, row in df.iterrows():
        ax.annotate(
            row['person2'],
            xy=(row[x_axis], row[y_axis]),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )
    
    # Set titles and labels
    ax.set_title(f'Conversation Analysis: Alessa with Others', fontsize=16)
    ax.set_xlabel(f'{x_axis.capitalize()} Score', fontsize=14)
    ax.set_ylabel(f'{y_axis.capitalize()} Score', fontsize=14)
    
    # Set axis limits with some padding
    x_padding = (df[x_axis].max() - df[x_axis].min()) * 0.1
    y_padding = (df[y_axis].max() - df[y_axis].min()) * 0.1
    
    ax.set_xlim(df[x_axis].min() - x_padding, df[x_axis].max() + x_padding)
    ax.set_ylim(df[y_axis].min() - y_padding, df[y_axis].max() + y_padding)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add reference lines for average values
    ax.axvline(x=df[x_axis].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=df[y_axis].mean(), color='gray', linestyle='--', alpha=0.5)
    
    # Add text to indicate quadrants
    avg_x = df[x_axis].mean()
    avg_y = df[y_axis].mean()
    
    ax.text(
        df[x_axis].min(), 
        df[y_axis].max(), 
        f"High {y_axis}, Low {x_axis}", 
        fontsize=10, 
        ha='left', 
        va='top'
    )
    
    ax.text(
        df[x_axis].max(), 
        df[y_axis].max(), 
        f"High {y_axis}, High {x_axis}", 
        fontsize=10, 
        ha='right', 
        va='top'
    )
    
    ax.text(
        df[x_axis].min(), 
        df[y_axis].min(), 
        f"Low {y_axis}, Low {x_axis}", 
        fontsize=10, 
        ha='left', 
        va='bottom'
    )
    
    ax.text(
        df[x_axis].max(), 
        df[y_axis].min(), 
        f"Low {y_axis}, High {x_axis}", 
        fontsize=10, 
        ha='right', 
        va='bottom'
    )
    
    # Annotate top performers
    if x_axis != y_axis:  # Only if showing different metrics
        best_overall_idx = df[['sentiment', 'similarity']].mean(axis=1).idxmax()
        best_person = df.loc[best_overall_idx, 'person2']
        
        ax.annotate(
            f"Best overall: {best_person}",
            xy=(df.loc[best_overall_idx, x_axis], df.loc[best_overall_idx, y_axis]),
            xytext=(20, 20),
            textcoords='offset points',
            fontsize=12,
            arrowprops=dict(arrowstyle='->', color='red')
        )
    
    st.pyplot(fig)
    
    # Show chart explanation
    st.markdown("""
    ### How to read this chart:
    - **Position**: The x and y coordinates show the chosen metrics (sentiment and similarity)
    - **Bubble Size**: Larger bubbles indicate stronger overall connections (average of sentiment and similarity)
    - **Quadrants**: Divided by average values, showing relative strengths in each metric
    - **Best Overall**: Person with the highest combined scores is highlighted
    """)

with tab2:
    st.header("Detailed Conversation Summaries")
    
    # Get person with highest score for highlight
    best_match = df.loc[df['bubble_size'].idxmax(), 'person2']
    
    # Create an expandable section for each conversation, sorted by score
    sorted_by_score = df.sort_values('bubble_size', ascending=False)
    
    for _, row in sorted_by_score.iterrows():
        person = row['person2']
        is_best = person == best_match
        
        # Add a special indicator for the best match
        title = f"{person} (Sentiment: {row['sentiment']}, Similarity: {row['similarity']})"
        if is_best:
            title = f"🌟 BEST MATCH: {title}"
        
        with st.expander(title, expanded=is_best):
            st.markdown(row['summary'])

# Add insights section
st.header("Key Insights")

# Get best and worst matches
best_match_idx = df['bubble_size'].idxmax()
worst_match_idx = df['bubble_size'].idxmin()

best_match_data = df.iloc[best_match_idx]
worst_match_data = df.iloc[worst_match_idx]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Strongest Connection")
    st.markdown(f"**Person:** {best_match_data['person2']}")
    st.markdown(f"**Sentiment:** {best_match_data['sentiment']}/100")
    st.markdown(f"**Similarity:** {best_match_data['similarity']}/100")
    st.markdown(f"**Overall Score:** {best_match_data['bubble_size']:.1f}")

with col2:
    st.subheader("Lowest Connection")
    st.markdown(f"**Person:** {worst_match_data['person2']}")
    st.markdown(f"**Sentiment:** {worst_match_data['sentiment']}/100")
    st.markdown(f"**Similarity:** {worst_match_data['similarity']}/100")  
    st.markdown(f"**Overall Score:** {worst_match_data['bubble_size']:.1f}")

# Add file upload option for real data
st.sidebar.markdown("---")
st.sidebar.header("Upload Your Own Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV or TSV file", type=["csv", "tsv"])
if uploaded_file is not None:
    try:
        # Try to determine if it's CSV or TSV
        if uploaded_file.name.endswith('.tsv'):
            user_df = pd.read_csv(uploaded_file, sep='\t')
        else:
            user_df = pd.read_csv(uploaded_file)
            
        st.sidebar.success("File uploaded successfully! Refresh the page to see your data.")
        # In a real app, you would update the visualization with this data
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Instructions
- Use the dropdowns to change the metrics shown on each axis
- The bubble size represents the average of sentiment and similarity
- Larger bubbles indicate stronger overall connections
- Check the "Summary View" tab for detailed conversation analyses
""")
