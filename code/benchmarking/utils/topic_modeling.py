import argparse
import os
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from datasets import load_dataset
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from umap import UMAP

# This script performs topic modeling on a dataset using BERTopic.

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    embed_model_id = "BAAI/bge-base-en-v1.5"
    embedding_model = SentenceTransformer(embed_model_id)
    hdbscan_model = HDBSCAN(
        min_cluster_size=50, 
        min_samples=15, 
        prediction_data=True, 
        metric="euclidean"
    )
    umap_model = UMAP(n_neighbors=50, 
                      n_components=10, 
                      min_dist=0.0, 
                      metric="euclidean")
    vectorizer_model = CountVectorizer(
        stop_words="english",
          min_df=5, 
          ngram_range=(1, 2)
    )
    representation_model = KeyBERTInspired(nr_repr_docs=50)
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True, 
        reduce_frequent_words=True
    )

    topic_model = BERTopic(
        calculate_probabilities=True,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        verbose=True,
    )

    ds = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(args.data_root_dir, 
                                  "annotations", 
                                  "train.csv"),
            "val": os.path.join(args.data_root_dir, 
                                "annotations", 
                                "val.csv"),
            "test": os.path.join(args.data_root_dir, 
                                 "annotations", 
                                 "test.csv"),
        },
    )

    docs = (
        ds["train"]["text_content"]
        + ds["val"]["text_content"]
        + ds["test"]["text_content"]
    )
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    # euclidean distance on normalized vectors is equivalent to cosine distance
    embeddings = normalize(embeddings)
    topics, probs = topic_model.fit_transform(docs, 
                                              embeddings=embeddings)

    new_topics = topic_model.reduce_outliers(
        docs, 
        topics=topics, 
        strategy="c-tf-idf", 
        probabilities=probs, 
        threshold=0.7
    )
    new_topics = topic_model.reduce_outliers(
        docs,
        topics=new_topics,
        strategy="embeddings",
        probabilities=probs,
        threshold=0.7,
        embeddings=embeddings,
    )
    new_topics = topic_model.reduce_outliers(
        docs,
        topics=new_topics,
        strategy="distributions",
        probabilities=probs,
        threshold=0.7,
    )

    topic_model.update_topics(
        docs,
        topics=new_topics,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
    )

    topic_model.save(
        args.output_dir,
        serialization="safetensors",
        save_embedding_model=embed_model_id,
    )

    topic_info_df = topic_model.get_topic_info()
    topic_info_df.to_csv(os.path.join(args.output_dir, 
                                      "topic_info.csv"), 
                                      index=False)

    # add topics to dataset
    topic_labels = [topic_model.topic_labels_[idx] for idx in topic_model.topics_]
    ds["train"] = ds["train"].add_column("bertopics", 
                                         topic_labels[: len(ds["train"])])
    ds["val"] = ds["val"].add_column(
        "bertopics", topic_labels[len(ds["train"]) : len(ds["train"]) + len(ds["val"])]
    )
    ds["test"] = ds["test"].add_column("bertopics", 
                                       topic_labels[-len(ds["test"]) :])

    ds["train"].to_csv(os.path.join(args.output_dir, 
                                    "train_bertopic.csv"), 
                                    index=False)
    ds["val"].to_csv(os.path.join(args.output_dir, 
                                  "val_bertopic.csv"), 
                                  index=False)
    ds["test"].to_csv(os.path.join(args.output_dir, 
                                   "test_bertopic.csv"), 
                                   index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)
