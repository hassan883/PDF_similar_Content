from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Create the 'pdfs' directory if it doesn't exist
pdfs_dir = 'pdfs'
if not os.path.exists(pdfs_dir):
    os.makedirs(pdfs_dir)
app.config['UPLOAD_FOLDER'] = pdfs_dir

# Load the SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Store the PDF text and metadata
pdfs = []

# Index PDFs
def index_pdfs(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                pdfs.append({"filename": filename, "text": pdf_text})

# Search PDFs
def search_pdfs(query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([pdf['text'] for pdf in pdfs] + [query])
    similarities = cosine_similarity(tfidf_matrix)

    # Filter and sort results based on similarity
    relevant_results = []
    for i in range(len(pdfs)):
        similarity_score = similarities[i][-1]
        if similarity_score > 0.0001:  # Adjust the threshold as needed
            relevant_results.append((pdfs[i], similarity_score))

    # Sort the relevant results by similarity score in descending order
    relevant_results.sort(key=lambda x: x[1], reverse=True)
    # print("SImilarity ; ",relevant_results[0][1] )
    # print("Type of relevant resulst is", type(relevant_results[0][0]['text']))
    if(len(relevant_results)==0):
        return "Nothing is there"
    print("The Length of ", len(relevant_results))
        
    
    relevant_rs = relevant_results[0][0]['text']
    query_sentences = query.split(".")
    content_sentences = relevant_rs.split(".")

# Vectorize the content and query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(content_sentences + query_sentences)

# Calculate cosine similarities
    similarities = cosine_similarity(tfidf_matrix)

# Get the index of the most similar content
    most_similar_index = similarities[-1].argsort()[-2]

# Retrieve the most similar content
    most_similar_content = content_sentences[most_similar_index]
    dic = {'filename': relevant_results[0][0]['filename'],
           'text':most_similar_content
           }
    tu = (dic, relevant_results[0][1])
    li = [tu]

    return li




@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query')
    results = search_pdfs(query)
    return render_template('index.html', results=results, query=query)
# Create the index on app startup
index_pdfs(pdfs_dir)

@app.route('/')
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        results = search_pdfs(query)
        return render_template('index.html', results=results, query=query)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    pdf_file = request.files['pdf_file']
    if pdf_file and pdf_file.filename.endswith('.pdf'):
        pdf_filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        pdf_file.save(pdf_path)
        index_pdfs(app.config['UPLOAD_FOLDER'])
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
