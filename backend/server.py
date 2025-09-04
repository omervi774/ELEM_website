
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from io import BytesIO
from io import BytesIO
from train_model import train
from pca_analysis import pca
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from bert_embdding import get_bert_embedding
# import plotly.graph_objects as go
# from annonimization import anonymize_text


app = Flask(__name__)
CORS(app)  # Allow React (running on different port) to access Flask

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'result': 'לא נשלח קובץ'}), 400

    try:
        df = pd.read_csv(file)
        model = train(df)
        # pca(df)

       # Save to memory
        output = BytesIO()
        df.to_csv(output, index=False, encoding="utf-8-sig")
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='bert_embeddings.csv'
        )

    except Exception as e:
        print(str(e))
        return jsonify({'result': f'שגיאה: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
