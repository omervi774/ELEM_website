from bert_embdding import get_bert_embedding
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA    
def pca(df):    
    # Get BERT embeddings (don't add to df)
    embeddings = df["text"].apply(lambda x: get_bert_embedding(str(x)))
    embeddings_matrix = pd.DataFrame(embeddings.tolist())

    # Run PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_matrix)

    

    fig, ax = plt.subplots(figsize=(8, 6))

    # First 7 rows — blue
    ax.scatter(pca_result[:38, 0], pca_result[:38, 1], color='blue', label='loneliness')

    # The rest — orange
    ax.scatter(pca_result[38:, 0], pca_result[38:, 1], color='orange', label='substence')

    # Labels & title
    ax.set_title("PCA")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC2")
    ax.legend(loc='best')

    # Save and close
    plt.savefig("pca_plot.png")
    plt.close()