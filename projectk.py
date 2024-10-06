import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from google.generativeai import GenerativeModel, configure
import requests

# Replace with your API key
GOOGLE_API_KEY = "AIzaSyA5HtRnzGruiia-aKtMMLnBjJ0ovTh11nE"

class Tools:
    def __init__(self):  # Fixed: Changed _init_ to __init__
        self.available_tools = {
            "image_generation": {
                "function": self.image_generation_tool,
                "description": "Generates an image based on a given text prompt using Stable Diffusion API."
            },
            "plotly_scatter_plot": {
                "function": self.plotly_scatter_plot,
                "description": "Generates an interactive scatter plot using Plotly."
            },
            "plotly_line_chart": {
                "function": self.plotly_line_chart,
                "description": "Creates an interactive line chart using Plotly."
            },
            "plotly_bar_chart": {
                "function": self.plotly_bar_chart,
                "description": "Generates an interactive bar chart using Plotly."
            },
            "plotly_3d_surface": {
                "function": self.plotly_3d_surface,
                "description": "Generates a 3D surface plot using Plotly."
            },
            "plotly_pie_chart": {
                "function": self.plotly_pie_chart,
                "description": "Creates an interactive pie chart using Plotly."
            },
            "plotly_histogram": {
                "function": self.plotly_histogram,
                "description": "Generates a histogram to display frequency distributions using Plotly."
            },
            "plotly_3d_scatter": {
                "function": self.plotly_3d_scatter,
                "description": "Creates a 3D scatter plot with Plotly."
            },
            "plotly_animated_chart": {
                "function": self.plotly_animated_chart,
                "description": "Generates an animated plot in Plotly to visualize changes over time."
            },
            "matplotlib_scatter_plot": {
                "function": self.matplotlib_scatter_plot,
                "description": "Creates a scatter plot using Matplotlib."
            },
            "matplotlib_line_plot": {
                "function": self.matplotlib_line_plot,
                "description": "Generates a line plot using Matplotlib."
            },
            "matplotlib_bar_chart": {
                "function": self.matplotlib_bar_chart,
                "description": "Creates a bar chart using Matplotlib."
            },
            "matplotlib_histogram": {
                "function": self.matplotlib_histogram,
                "description": "Generates a histogram using Matplotlib."
            },
            "matplotlib_heatmap": {
                "function": self.matplotlib_heatmap,
                "description": "Creates a heatmap using Matplotlib."
            }
        }

    # Image Generation Tool (Existing tool)
    def image_generation_tool(self, prompt):
        API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.content  # Return image bytes
                else:
                    return f"Error: {response.status_code}, {response.text}"
            except Exception as e:
                return f"Exception occurred: {str(e)}"

        image_bytes = query({"inputs": prompt})
        if isinstance(image_bytes, bytes):
            with open(f"generated_image_{prompt[:10]}.png", "wb") as f:
                f.write(image_bytes)
            return f"Image generated and saved as: generated_image_{prompt[:10]}.png"
        else:
            return image_bytes

    # ----------------- Plotly Tools ---------------------
    
    # Plotly Scatter Plot
    def plotly_scatter_plot(self, x, y):
        try:
            fig = px.scatter(x=x, y=y, title="Plotly Scatter Plot", labels={'x': 'X-axis', 'y': 'Y-axis'})
            st.plotly_chart(fig)
            return "Plotly scatter plot displayed successfully."
        except Exception as e:
            return f"Error in creating scatter plot: {str(e)}"

    # Plotly Line Chart
    def plotly_line_chart(self, x, y):
        try:
            fig = px.line(x=x, y=y, title="Plotly Line Chart", labels={'x': 'X-axis', 'y': 'Y-axis'})
            st.plotly_chart(fig)
            return "Plotly line chart displayed successfully."
        except Exception as e:
            return f"Error in creating line chart: {str(e)}"
    
    # Plotly Bar Chart
    def plotly_bar_chart(self, x, y):
        try:
            fig = px.bar(x=x, y=y, title="Plotly Bar Chart", labels={'x': 'Categories', 'y': 'Values'})
            st.plotly_chart(fig)
            return "Plotly bar chart displayed successfully."
        except Exception as e:
            return f"Error in creating bar chart: {str(e)}"
    
    # Plotly Pie Chart
    def plotly_pie_chart(self, values, labels):
        try:
            fig = px.pie(values=values, names=labels, title="Plotly Pie Chart")
            st.plotly_chart(fig)
            return "Plotly pie chart displayed successfully."
        except Exception as e:
            return f"Error in creating pie chart: {str(e)}"
    
    # Plotly Histogram
    def plotly_histogram(self, x, bins=10):
        try:
            fig = px.histogram(x=x, nbins=bins, title="Plotly Histogram", labels={'x': 'Values', 'y': 'Frequency'})
            st.plotly_chart(fig)
            return "Plotly histogram displayed successfully."
        except Exception as e:
            return f"Error in creating histogram: {str(e)}"
    
    # Plotly 3D Surface Plot
    def plotly_3d_surface(self, x, y, z):
        try:
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
            fig.update_layout(title="3D Surface Plot", scene=dict(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis"))
            st.plotly_chart(fig)
            return "3D surface plot displayed successfully."
        except Exception as e:
            return f"Error in creating 3D surface plot: {str(e)}"
    
    # Plotly 3D Scatter Plot
    def plotly_3d_scatter(self, x, y, z):
        try:
            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
            fig.update_layout(title="3D Scatter Plot", scene=dict(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis"))
            st.plotly_chart(fig)
            return "3D scatter plot displayed successfully."
        except Exception as e:
            return f"Error in creating 3D scatter plot: {str(e)}"
    
    # Plotly Animated Chart
    def plotly_animated_chart(self, x, y, frame_data, animation_type="scatter"):
        try:
            if animation_type == "scatter":
                fig = px.scatter(x=x, y=y, animation_frame=frame_data, title="Animated Scatter Plot")
            elif animation_type == "line":
                fig = px.line(x=x, y=y, animation_frame=frame_data, title="Animated Line Plot")
            else:
                return f"Unsupported animation type: {animation_type}"

            st.plotly_chart(fig)
            return f"Plotly {animation_type} animated plot displayed successfully."
        except Exception as e:
            return f"Error in creating animated plot: {str(e)}"

    # ----------------- Matplotlib Tools ---------------------

    # Matplotlib Scatter Plot
    def matplotlib_scatter_plot(self, x, y):
        try:
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            ax.set_title("Matplotlib Scatter Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            st.pyplot(fig)
            return "Matplotlib scatter plot displayed successfully."
        except Exception as e:
            return f"Error in creating scatter plot: {str(e)}"

    # Matplotlib Line Plot
    def matplotlib_line_plot(self, x, y):
        try:
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title("Matplotlib Line Plot")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            st.pyplot(fig)
            return "Matplotlib line plot displayed successfully."
        except Exception as e:
            return f"Error in creating line plot: {str(e)}"

    # Matplotlib Bar Chart
    def matplotlib_bar_chart(self, categories, values):
        try:
            fig, ax = plt.subplots()
            ax.bar(categories, values)
            ax.set_title("Matplotlib Bar Chart")
            ax.set_xlabel("Categories")
            ax.set_ylabel("Values")
            st.pyplot(fig)
            return "Matplotlib bar chart displayed successfully."
        except Exception as e:
            return f"Error in creating bar chart: {str(e)}"

    # Matplotlib Histogram
    def matplotlib_histogram(self, data, bins=10):
        try:
            fig, ax = plt.subplots()
            ax.hist(data, bins=bins)
            ax.set_title("Matplotlib Histogram")
            ax.set_xlabel("Values")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            return "Matplotlib histogram displayed successfully."
        except Exception as e:
            return f"Error in creating histogram: {str(e)}"

    # Matplotlib Heatmap
    def matplotlib_heatmap(self, data):
        try:
            fig, ax = plt.subplots()
            cax = ax.imshow(data, cmap='hot', interpolation='nearest')
            fig.colorbar(cax)
            ax.set_title("Matplotlib Heatmap")
            st.pyplot(fig)
            return "Matplotlib heatmap displayed successfully."
        except Exception as e:
            return f"Error in creating heatmap: {str(e)}"


# You can now call methods from the tools instance
# For example: tools.plotly_scatter_plot([1, 2, 3], [4, 5, 6])

    # Function for agent to use tools
    def use_tool(self, tool_name, input_data):
        tool = self.available_tools.get(tool_name)
        if tool:
            return tool["function"](*input_data)
        else:
            return f"Tool {tool_name} is not available."

    # Function to get tool description (LLM Memory/Reasoning)
    def get_tool_description(self, tool_name):
        tool = self.available_tools.get(tool_name)
        if tool:
            return tool["description"]
        else:
            return f"No description available for tool {tool_name}."

def analyze_and_visualize_data(df, experiment_type, tools):
    """Analyzes and visualizes the given dataset using Gemini API and various tools.
    Args:
      df: Pandas DataFrame.
      experiment_type: String, either "space" or "earth" to indicate the type of experiment.
      tools: Instance of the Tools class.
    """
    try:
        # Generate insights using Gemini API
        prompt = f"Analyze this dataset from a {experiment_type} biological experiment and provide insights:\n{df.head(20).to_string()}"
        model = GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        st.write(f"Gemini AI Summary for {experiment_type} experiment:")
        st.write(response.text)

        # Visualizations
        st.write(f"Data Visualizations for {experiment_type} experiment:")

        # Scatter Plot (Plotly)
        tools.use_tool("plotly_scatter_plot", [df.iloc[:, 0], df.iloc[:, 1]])

        # Line Chart (Plotly)
        tools.use_tool("plotly_line_chart", [df.iloc[:, 0], df.iloc[:, 1]])

        # Bar Chart (Plotly)
        tools.use_tool("plotly_bar_chart", [df.columns, df.iloc[:, 0]])

        # Pie Chart (Plotly)
        tools.use_tool("plotly_pie_chart", [df.iloc[:, 0].value_counts().values, df.iloc[:, 0].value_counts().index])

        # Histogram (Plotly)
        tools.use_tool("plotly_histogram", [df.iloc[:, 0]])

        # 3D Surface Plot (Plotly)
        tools.use_tool("plotly_3d_surface", [np.linspace(0, 10, 20), np.linspace(0, 10, 20), np.random.rand(20, 20)])

        # 3D Scatter Plot (Plotly)
        tools.use_tool("plotly_3d_scatter", [np.random.rand(50), np.random.rand(50), np.random.rand(50)])

        # Animated Scatter Plot (Plotly)
        tools.use_tool("plotly_animated_chart", [np.arange(10), np.random.rand(10), np.arange(10)])

        # Scatter Plot (Matplotlib)
        tools.use_tool("matplotlib_scatter_plot", [df.iloc[:, 0], df.iloc[:, 1]])

        # Line Plot (Matplotlib)
        tools.use_tool("matplotlib_line_plot", [df.iloc[:, 0], df.iloc[:, 1]])

        # Bar Chart (Matplotlib)
        tools.use_tool("matplotlib_bar_chart", [df.columns, df.iloc[:, 0]])

        # Histogram (Matplotlib)
        tools.use_tool("matplotlib_histogram", [df.iloc[:, 0]])

        # Heatmap (Matplotlib)
        tools.use_tool("matplotlib_heatmap", [np.random.rand(10, 10)])

        # Comparison Plots (Space vs. Earth)
        if experiment_type == "space" and "earth_experiment_data" in st.session_state:
            earth_df = st.session_state["earth_experiment_data"]
            st.write("Comparison Plots (Space vs. Earth):")

            # Scatter Plot Comparison
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color="experiment_type", labels={"color": "Experiment Type"})
            fig.add_trace(go.Scatter(x=earth_df.iloc[:, 0], y=earth_df.iloc[:, 1], mode='markers', name='Earth'))
            st.plotly_chart(fig)

            # Box Plot Comparison
            fig = px.box(pd.concat([df, earth_df], ignore_index=True), x="experiment_type", y=df.columns[0])
            st.plotly_chart(fig)

    except Exception as e:
        st.write(f"An error occurred: {e}")

# Streamlit app
st.title("Space Experiment Biological Data Visualization")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Check if API key is defined
        if not GOOGLE_API_KEY:
            st.write("Please set your GOOGLE_API_KEY environment variable.")
        else:
            # Configure the Gemini API
            configure(api_key=GOOGLE_API_KEY)

            # Determine the experiment type
            experiment_type = st.selectbox("Select the experiment type", ["space", "earth"])

            # Save earth experiment data for comparison
            if experiment_type == "earth":
                st.session_state["earth_experiment_data"] = df

            # Initialize the Tools class
            tools = Tools()

            # Analyze and visualize the data
            analyze_and_visualize_data(df, experiment_type, tools)

    except Exception as e:
        st.write(f"An error occurred while loading the file: {e}")