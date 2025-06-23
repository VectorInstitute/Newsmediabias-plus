# 📊 Demographics Extraction

This module extracts **demographic attributes** and **readability metrics** from news articles using LLMs (e.g., LLaMA-3). It supports asynchronous processing and outputs structured CSVs.

---

### Components

| File            | Description                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| `demo_utils.py` | Provides prompts and regex-based extractors for various demographic attributes.                                 |
| `demo.py`       | Asynchronous extractor for demographic types like `GENDER`, `RACE`, `RELIGION`, `TARGETED_GROUP`, and `TOPICS`. |
| `ideology.py`   | Specialized script to extract **political ideology** (`IDIOLOGY`) from articles.                                |
| `merge.py`      | Merges multiple demographic CSVs (e.g., gender, topics) into a unified analysis file.                           |
| `readablity.py` | Computes the **Flesch-Kincaid readability index** for article content.                                          |

---

### Supported Demographic Types

* `GENDER`: Male, Female, LGBTQ
* `RACE`: e.g., White, Black, Asian, etc.
* `RELIGION`: Christian, Muslim, Jewish, Hindu, Buddhist
* `TARGETED_GROUP`: Up to 5 focused population segments
* `TOPICS`: Max 3 topics per article
* `IDIOLOGY`: Left-wing, Right-wing, Centrist, or None

---