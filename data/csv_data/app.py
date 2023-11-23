import argparse
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, render_template

# Set up argparse to accept a directory path
parser = argparse.ArgumentParser(description='Plot CSV files in a specified directory.')
parser.add_argument('directory', type=str, help='Directory containing CSV files')
args = parser.parse_args()

app = Flask(__name__)

@app.route('/')
def index():
    plots_html = ''
    # Loop through each file in the specified directory
    for filename in os.listdir(args.directory):
        if filename.endswith('.csv'):
            # Read CSV data
            file_path = os.path.join(args.directory, filename)
            df = pd.read_csv(file_path)

            # Print DataFrame for debugging
            print(df.head())  # Print first few rows of the DataFrame

            # Convert all columns except the first one (timestamp) to numeric, handling non-numeric data
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Create a Plotly figure
            fig = px.line(df, x=df.columns[0], y=df.columns[1:], title=f'Plot of {filename}')

            # Print figure for debugging
            print(fig)

            # Convert Plotly figure to HTML and concatenate
            plots_html += pio.to_html(fig, full_html=False) + '<hr>'

    return render_template('index.html', plots_html=plots_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
