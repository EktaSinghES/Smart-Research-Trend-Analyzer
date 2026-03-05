import streamlit as st
import plotly.express as px
from app_methods import *

# Page config
st.set_page_config(
    page_title="Smart Research Trend Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #667eea; font-weight: 800 !important; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 15px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    [data-testid="metric-container"] label { color: rgba(255,255,255,0.8) !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: white !important; font-weight: 700; }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 600;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
</style>
""", unsafe_allow_html=True)


def main():
    """
    Smart Research Trend Analyzer - Single-purpose application
    Reuses NLP methods from the original Text Analysis app (spaCy, TextBlob, WordCloud)
    """

    st.title("🔬 Smart Research Trend Analyzer")
    st.caption("NLP & Time-Series Analytics for CHRIST University | Powered by spaCy, TextBlob & arXiv API")

    # Sidebar controls
    st.sidebar.header("📊 Analysis Settings")
    
    topic = st.sidebar.text_input(
        "Research Topic",
        value="natural language processing in education",
        help="Enter a research theme to analyze trends"
    )
    
    max_results = st.sidebar.slider("Papers to Fetch", min_value=30, max_value=300, value=100, step=10)
    months_window = st.sidebar.selectbox("Time Window (Months)", options=[12, 24, 36], index=1)
    num_keywords = st.sidebar.slider("Top Keywords to Extract", min_value=5, max_value=25, value=15)
    
    st.sidebar.markdown("---")
    st.sidebar.info(" This app uses the NLP tools (spaCy, TextBlob, WordCloud) from the original Text Analysis application.")

    # Main action button
    if st.button("Analyze Research Trends", type="primary", use_container_width=True):
        
        with st.spinner("Fetching live research data from arXiv..."):
            try:
                papers_df = fetch_arxiv_research_data(topic, max_results)
            except Exception as error:
                st.error(f"Unable to fetch data: {error}")
                return

        if papers_df is None or papers_df.empty:
            st.warning(" No papers found. Try a different topic.")
            return

        st.success(f"Retrieved {len(papers_df)} papers for analysis")
        # Build analytics
        monthly_counts_df = build_publication_timeseries(papers_df, months_window)
        keywords_df = extract_trending_keywords(papers_df, top_n=num_keywords)
        sentiment_df, avg_sentiment = compute_sentiment_trend(papers_df)
        momentum = compute_research_momentum(monthly_counts_df)
        domain_df = compute_christ_domain_relevance(papers_df)

        # Combine all paper text for NLP analysis (reusing original methods)
        combined_text = " ".join(
            (papers_df["title"].fillna("") + " " + papers_df["summary"].fillna("")).tolist()
        )

        # ========== KPI METRICS ==========
        st.markdown("### Key Metrics")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Papers Analyzed", f"{len(papers_df)}")
        kpi2.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        kpi3.metric("3-Month Momentum", f"{momentum:.1f}%")
        kpi4.metric("Unique Keywords", f"{len(keywords_df)}")

        st.markdown("---")

        # ========== TIME SERIES TREND ==========
        st.markdown("### Publication Trend Over Time")
        if not monthly_counts_df.empty:
            trend_fig = px.line(
                monthly_counts_df, x="month", y="paper_count",
                markers=True, title=""
            )
            trend_fig.update_layout(
                xaxis_title="Month", yaxis_title="Papers Published",
                template="plotly_white", hovermode="x unified"
            )
            trend_fig.update_traces(line=dict(color="#667eea", width=3), marker=dict(size=8, color="#764ba2"))
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("Insufficient data for time-series chart.")

        # ========== TWO-COLUMN CHARTS ==========
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Trending Keywords")
            if not keywords_df.empty:
                kw_fig = px.bar(
                    keywords_df.sort_values("count", ascending=True),
                    x="count", y="keyword", orientation="h",
                    color_discrete_sequence=["#667eea"]
                )
                kw_fig.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(kw_fig, use_container_width=True)

        with col2:
            st.markdown("### CHRIST Domain Relevance")
            if not domain_df.empty:
                dom_fig = px.bar(
                    domain_df.sort_values("relevance_score", ascending=True),
                    x="relevance_score", y="domain", orientation="h",
                    color_discrete_sequence=["#764ba2"]
                )
                dom_fig.update_layout(template="plotly_white", showlegend=False, xaxis_title="Relevance %")
                st.plotly_chart(dom_fig, use_container_width=True)

        st.markdown("---")

        # ========== NLP ANALYSIS (Reusing original app methods) ==========
        st.markdown("###  NLP Deep Dive (Using Original App Methods)")

        nlp_col1, nlp_col2 = st.columns(2)

        with nlp_col1:
            # Word Cloud (original method)
            with st.expander("☁️ Word Cloud from Abstracts", expanded=True):
                plot_wordcloud(combined_text)

            # Sentiment breakdown (original method)
            with st.expander(" Sentiment Analysis", expanded=True):
                sentiment_result = get_sentiment(combined_text)
                st.write(f"**Polarity:** {sentiment_result.polarity:.3f} (-1 negative, +1 positive)")
                st.write(f"**Subjectivity:** {sentiment_result.subjectivity:.3f} (0 factual, 1 opinionated)")
                st.progress(min(1.0, (sentiment_result.polarity + 1) / 2))

        with nlp_col2:
            # Token analysis (original method - on sample)
            with st.expander(" Token Analysis (Sample)", expanded=True):
                sample_text = combined_text[:3000]  # First 3000 chars for speed
                token_df = text_analyzer(sample_text)
                st.dataframe(token_df.head(50), use_container_width=True, hide_index=True)

            # Named entities (original method)
            with st.expander("Named Entities Detected", expanded=True):
                entities = get_entities(combined_text[:5000])
                if entities:
                    entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])
                    entity_counts = entity_df["Type"].value_counts().reset_index()
                    entity_counts.columns = ["Entity Type", "Count"]
                    st.dataframe(entity_counts, use_container_width=True, hide_index=True)
                else:
                    st.info("No entities detected.")

        st.markdown("---")

        # ========== PAPERS TABLE ==========
        st.markdown("### 📄 Recent Papers")
        display_df = sentiment_df[["published", "title", "authors", "sentiment", "url"]].copy()
        display_df["published"] = display_df["published"].dt.strftime("%Y-%m-%d")
        display_df.columns = ["Date", "Title", "Authors", "Sentiment", "URL"]
        st.dataframe(display_df.head(30), use_container_width=True, hide_index=True)

        # ========== INSIGHTS ==========
        st.markdown("###  Actionable Insights for CHRIST Projects")
        top_domains = domain_df.head(2)["domain"].tolist()
        st.success(f" Research momentum: **{momentum:.1f}%** — {'Growing' if momentum > 0 else 'Declining'} trend")
        st.info(f"🏛️ Best-fit CHRIST domains: **{', '.join(top_domains)}**")
        st.warning(" Use the extracted keywords to formulate mini-project titles, thesis proposals, or hackathon themes.")

        # ========== DOWNLOAD ==========
        with st.expander("⬇️ Download Analysis Results"):
            download(token_df)


if __name__ == "__main__":
    main()
