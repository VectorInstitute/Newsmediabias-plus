# Dataset Details
## News Sources
- CNN, Fox News, CBS News, ABC News, New York Times
- Washington Post, BBC, USA Today, Wall Street Journal
- AP News, Politico, New York Post, Forbes, Reuters
- Bloomberg, Al Jazeera, PBS NewsHour, The Guardian
- Newsmax, HuffPost, CNBC, C-SPAN, The Economist
- Financial Times, Time, Newsweek, The Atlantic
- The New Yorker, The Hill, ProPublica, Axios
- National Review, The Daily Beast, Daily Kos
- Washington Examiner, The Federalist, OANN
- Daily Caller, Breitbart, CBC, Toronto Sun
- Global News, The Globe and Mail, National Post

## Date Range
- Date range : 2023-05-06 till 2024-09-06
Python script using feedparser, newspaper3k, and selenium to scrape articles from multiple news sources, supporting keyword searches, custom date ranges, and outputting data to CSV files, with features for deduplication and image downloading.
## Key Features
- Scrapes articles from multiple news sources
- Supports keyword-based searches
- Downloads and stores article content and images
- Deduplicates articles based on unique identifiers
- Outputs data to CSV files


# Dataset Schema

    -- news_article_analysis (
    unique_id VARCHAR(255) PRIMARY KEY,
    
    -- Bias assessment
    text_label VARCHAR(10) CHECK (text_label IN ('Likely', 'Unlikely')),
    image_label VARCHAR(10) CHECK (image_label IN ('Likely', 'Unlikely')),
    text_label_reason TEXT,
    image_label_reason TEXT,
    news_category VARCHAR(255),
    
    -- Article content
    title VARCHAR(255),
    canonical_link VARCHAR(255),
    first_paragraph TEXT,
    outlet VARCHAR(100),
    source_url VARCHAR(255),
    topics TEXT,
    text_content TEXT,
    date_published TIMESTAMP,
    
    -- Image details
    img_description TEXT,
    image_filename VARCHAR(255),
    
    -- Topic modeling
    bertopics TEXT
)


## Access

Here's the information formatted in MkDocs style:

## Dataset Access

### Train
```
https://example.com/datasets/train_data.csv
```

### Val
```
https://example.com/datasets/train_data.csv```

### Test
```
https://example.com/datasets/train_data.csv```

This format provides clear and organized access to the dataset links for train, validation, and test sets. Users can easily copy the URLs for each dataset split.


## Sample Data
Certainly! I'll create a sample dataset with 3 entries based on the schema provided. Here's how you can present it in MkDocs format:

## Sample Data
## Article 1: Sex trafficking victim says Sen. Katie Britt telling her story during SOTU rebuttal is 'not fair' - CNN

- **Unique ID**: 1098444910
- **Title**: [Sex trafficking victim says Sen. Katie Britt telling her story during SOTU rebuttal is 'not fair' - CNN](https://www.cnn.com/2024/03/10/politics/katie-britt-sex-trafficking-victim-interview/index.html)
- **Text**: CNN — The woman whose story Alabama Sen. Katie Britt appeared to have shared in the Republican response to the State of the Union as an example of President Joe Biden’s failed immigration policies told CNN she was trafficked before Biden’s presidency and said legislators lack empathy when using the issue of human trafficking for political purposes. 
  > "I hardly ever cooperate with politicians, because it seems to me that they only want an image. They only want a photo — and that to me is not fair," Karla Jacinto told CNN on Sunday.
- **Outlet**: CNN
- **Source URL**: [CNN](https://www.cnn.com)
- **Topics**: 5_bipartisan, border, border deal, border policy, border wall
- **Date Published**: 2024-03-10
- **Image Description**: 
  > The image shows a person standing at a podium with a microphone, appearing to be giving a speech or presentation. The individual is wearing a pink blazer with a white shirt underneath. The background is indistinct but suggests an indoor setting with a wooden structure, possibly a room with a high ceiling. There are no visible logos, text, or other identifying features that provide context to the event or the person's identity.
- **Text Label**: Unlikely
- **Text Bias Analysis**:
  > "failed immigration policies", "lack of empathy", "despicable", "almost entirely preventable"
- **Image Label**: Unlikely
- **Image Analysis**:
  > The image alone does not provide enough context to analyze potential biases. The choice of the image could be influenced by the event's significance, the person's role, or the visual impact of the pink blazer. Without additional information, it is not possible to determine if the image is biased or Unbiased. The image does not appear to evoke strong emotions as it is a straightforward depiction of a person at a podium. There are no clear indications of stereotypes or oversimplification of complex issues in the image.

---

## Article 2: LA’s graffiti-tagged skyscraper: a work of art – and symbol of city’s wider failings - The Guardian US

- **Unique ID**: 1148232027
- **Title**: [LA’s graffiti-tagged skyscraper: a work of art – and symbol of city’s wider failings - The Guardian US](https://www.theguardian.com/us-news/2024/mar/17/los-angeles-graffiti-abandoned-skyscraper-downtown)
- **Text**: 
  > An asparagus patch is how the architect Charles Moore described the lackluster skyline of downtown Los Angeles in the 1980s. "The tallest stalk and the shortest stalk are just alike, except that the tallest has shot farther out of the ground." This sprawling city of bungalows has never been known for the quality of its high-rise buildings, and not much has changed since Moore’s day. A 1950s ordinance dictating that every tower must have a flat roof was rescinded in 2014, spawning a handful of clumsy quiffs and crowns atop a fresh crop of swollen glass slabs. It only added further evidence to the notion that architects in this seismic city are probably better suited to staying on the ground.
- **Outlet**: The Guardian US
- **Source URL**: [The Guardian US](https://www.theguardian.com)
- **Topics**: affordable housing, public housing, homeowners, housing crisis
- **Date Published**: 2024-03-17
- **Image Description**: 
  > The image shows a tall, multi-story building with numerous windows. The building is covered in various graffiti tags and symbols, with words like 'READY', 'SHAKA', 'RAKM', 'TOOL', 'TOLT', 'KERZ', 'SMK', 'DZER', 'MSK', and 'OBER' prominently displayed. The building is situated in an urban environment with other structures visible in the background. The sky is clear, suggesting it might be daytime. The image is taken from a high angle, looking down on the building.
- **Text Label**: Likely
- **Text Bias Analysis**:
  > "mind-numbingly generic glass boxes", "abandoned", "doing nothing", "if they ain’t gon finish the job", "This building has needed love for years", "the streets of LA are happy to make something out of it", "the developer had ceased paying"
- **Image Label**: Likely
- **Image Analysis**:
  > The image and accompanying headline from The Guardian US suggest a critical perspective on the state of urban development and the impact of graffiti on architecture. The choice of this image may be intended to highlight the issue of urban decay and the lack of maintenance in certain areas. The graffiti tags could be seen as a form of artistic expression, but within the context of the headline, they are likely to be interpreted as a symbol of the city's wider failings. The image does not provide a balanced view, as it focuses on the negative aspects of the building's appearance. The framing of the image, with the building as the central focus and the surrounding environment in the background, may lead viewers to associate the building's condition with the overall state of the city.
