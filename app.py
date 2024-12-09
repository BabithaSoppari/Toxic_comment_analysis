from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import joblib
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the saved SVM model and vectorizer
svm = joblib.load('optimized_svm_model.pkl')  # pre-trained SVM model
tfidf = joblib.load('tfidf_vectorizer.pkl')  #  pre-trained TF-IDF vectorizer

# Preprocessing function to clean text
def clean_text(text):
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    text = text.lower()  # Convert to lowercase
    return text

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to check individual comment toxicity
@app.route('/check_text', methods=['POST'])
def check_text():
    comment = request.form['comment_text']
    cleaned_comment = clean_text(comment)
    
    # Vectorize the comment using TF-IDF
    comment_tfidf = tfidf.transform([cleaned_comment])
    svm_prediction = svm.predict(comment_tfidf)[0]
    svm_proba = svm.predict_proba(comment_tfidf)[0][1]  # Toxicity probability

    # Prepare the result to return
    result = {
        "comment": comment,
        "svm_prediction": "Toxic" if svm_prediction == 1 else "Non-Toxic",
        "svm_proba": round(svm_proba * 100, 2)  # Convert to percentage
    }
    return render_template('index.html', result=result)

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Check if the necessary column exists in the file
        if 'comment_text' not in df.columns:
            return jsonify({'error': 'CSV must contain a column named "comment_text"'}), 400

        # Clean the comments
        comments = df['comment_text'].apply(clean_text)
        
        # Vectorize comments using TF-IDF
        comments_tfidf = tfidf.transform(comments)
        
        # Predict toxicity
        svm_predictions = svm.predict(comments_tfidf)
        svm_probabilities = svm.predict_proba(comments_tfidf)[:, 1]  # Toxicity probabilities

        # Add predictions and probabilities to DataFrame
        df['svm_predicted_toxic'] = svm_predictions
        df['svm_toxicity_probability'] = svm_probabilities
        
        # Generate Word Cloud
        text = ' '.join(comments)
        wordcloud = WordCloud(width=800, height=400, max_words=100, background_color="white").generate(text)
        wc_image_path = 'static/wordcloud.png'
        wordcloud.to_file(wc_image_path)
        
        # Generate Summary Statistics
        toxic_count = df['svm_predicted_toxic'].sum()
        non_toxic_count = len(df) - toxic_count
        total_comments = len(df)

        # Ensure the statistics are correct by checking the data type and calculating the counts properly
        toxic_percentage = (toxic_count / total_comments) * 100
        non_toxic_percentage = (non_toxic_count / total_comments) * 100
        
        # Create Toxic vs Non-Toxic chart
        plt.switch_backend('Agg')  # Ensure non-GUI backend for Flask
        fig, ax = plt.subplots()
        ax.bar(['Toxic', 'Non-Toxic'], [toxic_count, non_toxic_count], color=['red', 'green'])
        ax.set_title('Toxic vs Non-Toxic Comments')
        ax.set_xlabel('Comment Type')
        ax.set_ylabel('Count')
        
        # Save the chart as an image
        chart_image_path = 'static/toxic_vs_non_toxic.png'
        plt.savefig(chart_image_path)
        plt.close(fig)  # Close the plot to avoid memory issues
        
        # Save results to CSV
        processed_filename = f"processed_{filename}"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        df.to_csv(output_filepath, index=False)

        # Return results to frontend
        return render_template('index.html', download_link=processed_filename, wordcloud_image=wc_image_path,
                               chart_image=chart_image_path, toxic_count=toxic_count, non_toxic_count=non_toxic_count,
                               total_comments=total_comments, toxic_percentage=toxic_percentage, non_toxic_percentage=non_toxic_percentage)

    return jsonify({'error': 'Invalid file type'}), 400

# Route for downloading the processed file
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

# Main entry point for running the app
if __name__ == '__main__':
    app.run(debug=True)
