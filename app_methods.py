import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
from textblob import TextBlob
from collections import Counter
import base64
import time
import re
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from wordcloud import WordCloud
from PyPDF2 import PdfReader
import docx2txt
import spacy
import neattext as nt
import neattext.functions as nfx
import requests

# Configure matplotlib to use the 'Agg' backend
matplotlib_use("Agg")

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Time Format for filename
timestr = time.strftime("%Y%m%d-%H%M%S")


# Functions
def text_analyzer(my_text):
    """
    Analyze the given text using spaCy to extract various token attributes.

    Parameters:
    - my_text (str): Text to be analyzed.

    Returns:
    - DataFrame: DataFrame containing token attributes.
    """

    docx = nlp(my_text)
    all_data = [(token.text, token.shape_, token.pos_, token.tag_, 
                 token.lemma_, token.is_alpha, token.is_stop) for token in docx]
    df = pd.DataFrame(all_data, columns=['Token', 'Shape', 'PoS', 'Tag', 'Lemma', 'Is_Alpha', 'Is_Stopword'])
    return df


def get_entities(my_text):
    """
    Extract named entities from the given text using spaCy.

    Parameters:
    - my_text (str): Text to extract entities from.

    Returns:
    - list: List of tuples containing entity text and label.
    """

    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


def get_most_common_tokens(my_text, num=5):
    """
    Get the most common tokens from the given text.

    Parameters:
    - my_text (str): Text to extract tokens from.
    - num (int): Number of most common tokens to return.

    Returns:
    - dict: Dictionary of most common tokens and their counts.
    """

    word_tokens = Counter(my_text.split(' '))
    most_common_tokens = dict(word_tokens.most_common(num))
    return most_common_tokens


def get_sentiment(my_text):
    """
    Analyze the sentiment of the given text using TextBlob.

    Parameters:
    - my_text (str): Text to analyze sentiment.

    Returns:
    - Sentiment: Sentiment object containing polarity and subjectivity.
    """

    blob = TextBlob(my_text)
    sentiment = blob.sentiment
    return sentiment


def plot_wordcloud(my_text):
    """
    Generate and display a word cloud from the given text.

    Parameters:
    - my_text (str): Text to generate the word cloud from.
    """

    wc = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

def download(data):
    """
    Provide a download link for the given DataFrame.

    Parameters:
    - data (DataFrame): DataFrame to be downloaded as a CSV file.
    """

    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_file_name = f'nlp_result_{timestr}_.csv'
    st.markdown('### ⬇️ Download CSV File ⬇️')
    href = f'<a href="data:file/csv;base64, {b64}" download="{new_file_name}">Click Here !!</a>'
    st.markdown(href, unsafe_allow_html=True)


def read_pdf(file):
    """
    Extract text from a PDF file.

    Parameters:
    - file (UploadedFile): PDF file to extract text from.

    Returns:
    - str: Extracted text from the PDF file.
    """

    pdf_reader = PdfReader(file)
    count = len(pdf_reader.pages)
    all_page_text = ""
    for i in range(count):
        page = pdf_reader.pages[i]
        all_page_text += page.extract_text()
    return all_page_text


def analyze_text(raw_text, num_most_common):
    """
    Perform text analysis and visualization on the given text.

    Parameters:
    - raw_text (str): Text to be analyzed.
    - num_most_common (int): Number of most common tokens to display.
    """

    with st.expander("Original Text"):
        st.write(raw_text)

    with st.expander("Text Analysis"):
        token_res_df = text_analyzer(raw_text)
        st.dataframe(token_res_df)

    with st.expander("Entities"):
        entity_res = get_entities(raw_text)
        st.write(entity_res, height=500, scrolling=True)

    # Design Layout
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Word Stats"):
            st.info("🔬 Word Statistics")
            docx = nt.TextFrame(raw_text)
            st.write(docx.word_stats())

        with st.expander("Top Keywords"):
            st.info("🔬 Top Keywords/Tokens")
            processed_text = nfx.remove_stopwords(raw_text)
            key_words = get_most_common_tokens(processed_text, num_most_common)
            st.write(key_words)

        with st.expander("Sentiment"):
            sent_res = get_sentiment(raw_text)
            st.write(sent_res)

    with col2:
        with st.expander("Plot Word's Frequency"):
            fig = plt.figure()
            top_key_words = get_most_common_tokens(processed_text, num_most_common)
            plt.bar(key_words.keys(), top_key_words.values())
            st.pyplot(fig)

        with st.expander("Plot Part Of Speech"):
            try:
                fig = plt.figure()
                sns.countplot(x=token_res_df['PoS'], palette='viridis')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except:
                st.warning("Insufficient Data")

        with st.expander("Plot WordCloud"):
            plot_wordcloud(raw_text)

    with st.expander("Download Text Analysis Result"):
        download(token_res_df)


def handle_uploaded_file(text_file):
    """
    Handle the uploaded file and extract text based on the file type.

    Parameters:
    - text_file (UploadedFile): Uploaded file to process.

    Returns:
    - str: Extracted text from the uploaded file.
    """

    if text_file.type == 'application/pdf':
        raw_text = read_pdf(text_file)
    elif text_file.type == 'text/plain':
        raw_text = str(text_file.read(), encoding='utf-8')
    else:
        raw_text = docx2txt.process(text_file)
    return raw_text


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_arxiv_research_data(query, max_results=120):
    """
    Fetch research papers from arXiv API in real time.

    Parameters:
    - query (str): Search query to submit to arXiv.
    - max_results (int): Number of papers to fetch.

    Returns:
    - DataFrame: Paper metadata with title, abstract, date, and URL.
    """

    normalized_query = (query or "").strip()
    if not normalized_query:
        return pd.DataFrame(columns=["title", "summary", "published", "updated", "authors", "url"])

    encoded_query = quote_plus(normalized_query)
    api_url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )

    response = requests.get(api_url, timeout=20)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", namespace)

    records = []
    for entry in entries:
        title = (entry.findtext("atom:title", default="", namespaces=namespace) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=namespace) or "").strip().replace("\n", " ")
        published = (entry.findtext("atom:published", default="", namespaces=namespace) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=namespace) or "").strip()
        url = (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip()
        author_nodes = entry.findall("atom:author", namespace)
        authors = ", ".join(
            (author.findtext("atom:name", default="", namespaces=namespace) or "").strip()
            for author in author_nodes
        )

        records.append(
            {
                "title": title,
                "summary": summary,
                "published": pd.to_datetime(published, errors="coerce", utc=True),
                "updated": pd.to_datetime(updated, errors="coerce", utc=True),
                "authors": authors,
                "url": url,
            }
        )

    papers_df = pd.DataFrame(records)
    if papers_df.empty:
        return papers_df

    papers_df = papers_df.dropna(subset=["published"]).sort_values("published", ascending=False).reset_index(drop=True)
    return papers_df


def build_publication_timeseries(papers_df, months_window=24):
    """
    Build monthly publication count time-series.

    Parameters:
    - papers_df (DataFrame): Papers dataset with a published datetime column.
    - months_window (int): Number of recent months to include.

    Returns:
    - DataFrame: Monthly publication counts.
    """

    if papers_df.empty:
        return pd.DataFrame(columns=["month", "paper_count"])

    monthly_counts = (
        papers_df.assign(month=papers_df["published"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "paper_count"})
        .sort_values("month")
    )

    if months_window and len(monthly_counts) > months_window:
        monthly_counts = monthly_counts.tail(months_window)

    return monthly_counts


def extract_trending_keywords(papers_df, top_n=15):
    """
    Extract trending keywords from paper titles and abstracts.

    Parameters:
    - papers_df (DataFrame): Dataset of papers.
    - top_n (int): Number of keywords to return.

    Returns:
    - DataFrame: Top keywords and frequencies.
    """

    if papers_df.empty:
        return pd.DataFrame(columns=["keyword", "count"])

    text_blob = " ".join((papers_df["title"].fillna("") + " " + papers_df["summary"].fillna("")).tolist())
    cleaned_text = re.sub(r"[^a-zA-Z\s]", " ", text_blob.lower())
    docx = nlp(cleaned_text)

    lemmas = [
        token.lemma_
        for token in docx
        if token.is_alpha and (not token.is_stop) and len(token.lemma_) > 2
    ]

    keyword_freq = Counter(lemmas)
    return pd.DataFrame(keyword_freq.most_common(top_n), columns=["keyword", "count"])


def compute_sentiment_trend(papers_df):
    """
    Compute sentiment score per paper and average sentiment.

    Parameters:
    - papers_df (DataFrame): Dataset of papers.

    Returns:
    - tuple: (DataFrame with paper sentiment, float average sentiment)
    """

    if papers_df.empty:
        return papers_df.copy(), 0.0

    sentiment_scores = papers_df["summary"].fillna("").apply(lambda x: TextBlob(x).sentiment.polarity)
    sentiment_df = papers_df.copy()
    sentiment_df["sentiment"] = sentiment_scores

    average_sentiment = float(sentiment_scores.mean()) if len(sentiment_scores) else 0.0
    return sentiment_df, average_sentiment


def compute_research_momentum(monthly_counts_df):
    """
    Compute growth momentum from recent and previous 3-month windows.

    Parameters:
    - monthly_counts_df (DataFrame): Monthly publication counts.

    Returns:
    - float: Percent growth from previous 3 months to recent 3 months.
    """

    if monthly_counts_df.empty or len(monthly_counts_df) < 6:
        return 0.0

    recent_window = monthly_counts_df.tail(3)["paper_count"].sum()
    previous_window = monthly_counts_df.tail(6).head(3)["paper_count"].sum()

    if previous_window == 0:
        return 0.0

    return ((recent_window - previous_window) / previous_window) * 100


def compute_christ_domain_relevance(papers_df):
    """
    Map current research trends to CHRIST university domain themes.

    Parameters:
    - papers_df (DataFrame): Dataset of papers.

    Returns:
    - DataFrame: Department-wise relevance score table.
    """

    domain_keywords = {
        "School of Sciences": ["biology", "chemistry", "physics", "mathematics", "biotech", "ecology"],
        "School of Engineering": ["engineering", "iot", "robotics", "sensor", "network", "embedded"],
        "Computer Science & AI": ["artificial", "intelligence", "machine", "learning", "nlp", "data", "vision"],
        "Commerce & Management": ["finance", "market", "business", "economics", "management", "accounting"],
        "Social Sciences": ["education", "society", "policy", "psychology", "health", "behavior"],
    }

    if papers_df.empty:
        return pd.DataFrame(columns=["domain", "match_count", "relevance_score"])

    corpus = (papers_df["title"].fillna("") + " " + papers_df["summary"].fillna(""))
    merged_text = " ".join(corpus.tolist()).lower()

    rows = []
    for domain, keywords in domain_keywords.items():
        matches = sum(merged_text.count(keyword) for keyword in keywords)
        rows.append({"domain": domain, "match_count": matches})

    domain_df = pd.DataFrame(rows)
    total_matches = domain_df["match_count"].sum()
    if total_matches == 0:
        domain_df["relevance_score"] = 0.0
    else:
        domain_df["relevance_score"] = (domain_df["match_count"] / total_matches * 100).round(2)

    return domain_df.sort_values("relevance_score", ascending=False).reset_index(drop=True)
