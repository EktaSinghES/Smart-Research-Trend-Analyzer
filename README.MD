# Smart Research Trend Analyzer using NLP & Time-Series Analytics

A single-purpose Streamlit application that extends the original Text Analysis NLP app to provide real-time research trend intelligence for CHRIST University.

## What This App Does

Enter any research topic and get:
- **Live papers** fetched from arXiv API
- **Monthly publication trend** time-series chart
- **3-month growth momentum** percentage
- **Trending keywords** extracted using spaCy NLP
- **CHRIST domain relevance** mapping for academic planning
- **Sentiment analysis** of research abstracts (TextBlob)
- **Word cloud** visualization
- **Token analysis & Named Entity Recognition** (spaCy)
- **Downloadable results** as CSV

## Built On Top Of

This app reuses all NLP methods from the original [Text-Analysis-Using-NLP-Streamlit](https://github.com/mohammadreza-mohammadi94/Text-Analysis-Using-NLP-Streamlit):

| Method | Purpose |
|--------|---------|
| `text_analyzer()` | Token analysis with spaCy |
| `get_entities()` | Named entity recognition |
| `get_sentiment()` | TextBlob sentiment scoring |
| `plot_wordcloud()` | Word cloud generation |
| `download()` | CSV export |

## New Features Added

- Real-time arXiv API integration
- Monthly time-series trend analytics
- Research momentum calculation
- CHRIST-specific domain relevance scoring
- Interactive Plotly visualizations

## Run Locally

```bash
# Clone and enter directory
git clone <your-repo-url>
cd Text_Analysis_Using_NLP_Streamlit

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run app
streamlit run app.py
```

## Tech Stack

- Python, Streamlit
- spaCy (NLP from original app)
- TextBlob (Sentiment from original app)
- WordCloud (Visualization from original app)
- Plotly (New charts)
- arXiv API (Real-time data)
- pandas (Data processing)

## VIVA Points

1. **Problem**: Students need data-driven guidance for research topic selection
2. **Solution**: Real-time trend analyzer built on existing NLP tools
3. **Innovation**: Live API + time-series + domain mapping
4. **Reuse**: All core NLP from original app (spaCy, TextBlob, WordCloud)
5. **CHRIST relevance**: Domain mapping for academic planning

## License

MIT License
