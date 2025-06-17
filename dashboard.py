import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MultiLabelBinarizer
from lightfm import LightFM
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import requests
from io import StringIO
from datetime import datetime
from fpdf import FPDF
import base64
import cohere

# --- Preparation ---
df = pd.read_csv("clean_data_model.csv")
df['related_list'] = df['related_products'].apply(eval)
issue_cols = [
    "Infrastructure", "Data", "AI", "Security", "Collaboration",
    "Sustainability", "Customer Experience", "Supply Chain", "Manufacturing"
]
df["issue_tags"] = (
    df[issue_cols]
    .apply(lambda r: [c for c in issue_cols if r[c] == 1], axis=1)
)

# --- Get API keys ---
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# --- Processing ---
cat_feats = ['business_need', 'industry', 'region']
num_feats = ['weighted_tone', 'weighted_article_count']
ord_feats = ['employees']
ord_cats  = [['1-49 employees','50-999 employees','1,000-9,999 employees','10,000+ employees']]

preprocessor_full = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_feats),
    ('num', StandardScaler(), num_feats),
    ('ord', OrdinalEncoder(categories=ord_cats), ord_feats)
], remainder='drop')

X_full = preprocessor_full.fit_transform(df)

mlb_full = MultiLabelBinarizer(sparse_output=True)
R_full = mlb_full.fit_transform(df['related_list'])
product_names = mlb_full.classes_

interactions_full = csr_matrix(R_full)
user_feat_full = csr_matrix(X_full)
item_features = csr_matrix(np.eye(len(product_names)))

model_full = LightFM(
    no_components=20,
    loss='warp',
    learning_rate=0.05,
    user_alpha=1e-6,
    item_alpha=1e-6
)
model_full.fit(
    interactions_full,
    user_features=user_feat_full,
    item_features=item_features,
    epochs=30,
    num_threads=4
)

# --- Helpers ---
def predict_from_inputs(
    business_need, industry, region,
    weighted_tone, weighted_article_count, employees,
    infrastructure, data, ai, security, collaboration,
    sustainability, customer_experience, supply_chain, manufacturing,
    model, preprocessor, item_features, product_names, top_n=5):

    input_dict = {
        'business_need': [business_need],
        'industry': [industry],
        'region': [region],
        'weighted_tone': [weighted_tone],
        'weighted_article_count': [weighted_article_count],
        'employees': [employees],
        'Infrastructure':[infrastructure],
        'Data':[data],
        'AI':[ai],
        'Security':[security],
        'Collaboration':[collaboration],
        'Sustainability':[sustainability],
        'Customer Experience':[customer_experience],
        'Supply Chain':[supply_chain],
        'Manufacturing':[manufacturing]
    }
    df_input = pd.DataFrame(input_dict)
    X_user = preprocessor.transform(df_input)
    user_features = csr_matrix(X_user)
    n_items = len(product_names)
    scores = model.predict(0, np.arange(n_items), user_features=user_features, item_features=item_features)
    top_indices = np.argsort(-scores)[:top_n]
    return [(product_names[i], scores[i]) for i in top_indices]

def find_similar_cases_full(new_row_df, top_n_products=5, top_k_cases=5):
    x_new = preprocessor_full.transform(new_row_df)
    u_new = csr_matrix(x_new)
    scores_new = model_full.predict(0, np.arange(len(product_names)), user_features=u_new, item_features=item_features)
    top_products = np.argsort(-scores_new)[:top_n_products]
    prod_mask = np.zeros(len(product_names), dtype=int)
    prod_mask[top_products] = 1
    f_new = np.hstack([x_new.ravel(), prod_mask])
    F_hist = np.hstack([X_full, R_full.toarray()])
    nn = NearestNeighbors(n_neighbors=min(top_k_cases, F_hist.shape[0]), metric='cosine').fit(F_hist)
    dist, idxs = nn.kneighbors(f_new.reshape(1, -1))
    sims, neighbors = 1 - dist[0], idxs[0]

    results = []
    for i, sim in zip(neighbors, sims):
        row = df.iloc[i]
        results.append({
            "similarity": float(sim),
            "index": int(i),
            "business_need": row['business_need'],
            "industry": row['industry'],
            "region": row['region'],
            "employees": row['employees'],
            "related_list": row['related_list'],
            "issue_tags": row['issue_tags'],
            "full_row": row.to_dict()
        })
    return results

def get_industry_news(industry, max_articles=5):
    query = industry.replace("&", "and")  # sanitize query
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize={max_articles}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        headlines = [f"- {a['title']} ({a['source']['name']})" for a in articles]
        full_text = "\n".join([a['description'] or a['title'] for a in articles if a.get('description')])
        return headlines, full_text
    except Exception as e:
        return [f"Error fetching news: {e}"], ""
    
def get_company_news(company_name, df):
    modes = ["timelinetone", "timelinevolraw"]
    today = datetime.today().strftime('%Y%m%d')
    end_date = (pd.to_datetime(today, format='%Y%m%d') - pd.DateOffset(days=3)).strftime('%Y%m%d')
    start_date = (pd.to_datetime(end_date, format='%Y%m%d') - pd.DateOffset(days=90)).strftime('%Y%m%d')
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }

    merged_df = pd.DataFrame()
    weighted_tone = df['weighted_tone'].mean()
    weighted_article_count = df['weighted_article_count'].mean()

    for mode in modes:
        params = {
            "query":         f"{company_name}",
            "mode":          mode,
            "format":        "csv",
            "startdatetime": f"{start_date}000000",
            "enddatetime":   f"{end_date}000000",
            "maxrecords":    250,
            "sort":          "datedesc"
        }

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=3)
        except requests.RequestException as e:
            continue
        if resp.status_code != 200:
            continue
        if len(resp.text) <= 1:
            continue

        resp_df = pd.read_csv(StringIO(resp.text))
        if mode in {"timelinetone", "timelinevolraw", "timelinelang", "timelinesourcecountry"}:
            if {'Date', 'Series', 'Value'}.issubset(resp_df.columns):
                resp_df = resp_df.pivot(index="Date", columns="Series", values="Value").reset_index()
            else:
                continue
        
        if merged_df.empty:
            merged_df = resp_df
        else:
            merged_df = pd.merge(merged_df, resp_df, on="Date", how="outer")
            merged_df = merged_df.dropna(subset=['Average Tone', 'Article Count', 'Total Monitored Articles'])

            weighted_tone = (
                (merged_df['Average Tone'] * merged_df['Article Count']).sum()
                / merged_df['Article Count'].sum()
            )

            weighted_article_count = (
                (merged_df['Article Count'] / merged_df['Total Monitored Articles']).mean()
            )
    
    return weighted_tone, weighted_article_count

def generate_email(sim_cases, recommendations, business_need, industry, region):
    example_case = sim_cases[0]
    case_industry = example_case['industry']
    case_region = example_case['region']
    case_need = example_case['business_need']
    case_products = example_case['related_list']

    business_impact_map = {
        "Customer Engagement": "increase conversion by 15–30%",
        "Supply Chain Optimization": "reduce operational costs by up to 20%",
        "Workforce Productivity": "boost output per employee by 25%",
        "Sustainability": "lower energy costs by 12–18%",
        "Data Analytics": "accelerate insights generation by 40%"
    }
    impact_sentence = business_impact_map.get(business_need, "achieve measurable business impact")

    summary_prompt = f"""
    You're a strategic Microsoft sales advisor writing a consultative pitch.

    1. Start with a short sentence explaining a **typical challenge** for a company in the {industry} sector with a need for {business_need.lower()} in {region}.
    2. Present Microsoft's **recommended solutions**: {', '.join(p for p, _ in recommendations)}.
    3. Summarize how these tools solve the problem and lead to outcomes like {impact_sentence}.
    4. Include a brief **story from a similar case**: A company in the {case_industry} sector (also focused on {case_need}) benefited from: {', '.join(case_products)}.
    5. Keep the tone persuasive but professional.
    6. Wrap up with a positive outlook for digital transformation.

    Please write this as a short outreach email. No headings, just full text.
    """

    try:
        co = cohere.Client(COHERE_API_KEY)
        response = co.generate(model='command-r-plus', prompt=summary_prompt, max_tokens=300)
        summary = response.generations[0].text.strip()
    except Exception as e:
        summary = f"⚠️ Cohere API error: {str(e)}"

    email_txt = f"""{summary}\n\nBest regards,\nYour Microsoft Sales Team"""

    return email_txt

def generate_trends(news_headlines, news_text, industry):
    if news_text:
        try:
            news_summary_prompt = f"""
            Please extract exactly three current trends from the following news descriptions related to the {industry} industry.

            Format your response as a numbered list with short, clear sentences. Each point should be no longer than 2 lines:

            {news_text}
            """.strip()

            try:
                co = cohere.Client(COHERE_API_KEY)
                news_response = co.generate(
                    model='command-r-plus',
                    prompt=news_summary_prompt,
                    max_tokens=90,
                    temperature=0.4
                )
                trends = news_response.generations[0].text.strip()
            except Exception as e:
                trends = f"⚠️ Cohere API error: {str(e)}""
        except Exception as e:
            trends = f"⚠️ Cohere summarization error: {str(e)}"
    else:
        trends = "No recent news available for this industry."

    return news_headlines, trends

# --- UI ---
st.set_page_config(page_title="Microsoft Product Recommender", layout="wide")
st.title("🔍 Microsoft Product Recommender")
st.markdown("This tool helps Microsoft sales teams recommend products based on company features, historical sales, and news sentiment. Get started by filling out the form in the sidebar!")

# --- Sidebar ---
with st.sidebar:
    region_order = ['North America', 'South America', 'Nordics', 'Western Europe', 'Central Europe','Eastern Europe',
                    'Southern Europe', 'Africa', 'Middle East', 'East Asia', 'Southeast Asia',  'South Asia', 'Oceania']
    employees_order = ['1-49 employees', '50-999 employees', '1,000-9,999 employees', '10,000+ employees']
    company_name = st.text_input("Company Name")
    business_need = st.selectbox("Business Need",sorted(df["business_need"].dropna().unique()))
    industry = st.selectbox("Industry", sorted(df["industry"].dropna().unique()))
    region = st.selectbox("Region", region_order)
    employees = st.selectbox("Employees", employees_order)
    st.markdown("---")
    st.markdown("Select issue tags")
    tags = ['Infrastructure', 'Data', 'AI', 'Security', 'Collaboration', 'Sustainability', 'Customer Experience', 'Supply Chain', 'Manufacturing']
    tag_inputs = {tag: int(st.checkbox(tag)) for tag in tags}
    st.markdown("---")
    tone = st.selectbox("Email Style", ["Formal", "Playful", "Concise"])
    st.markdown("---")
    trigger = st.button("🚀 Generate Insights")

# --- Generation ---
if trigger:
    with st.status("Generating insights…", expanded=False) as status:

# Company News        
        status.write("🔍 Fetching company news")
    
        weighted_tone, weighted_article_count = get_company_news(company_name, df)

# Recommendations
        status.write("♟️ Computing recommendations")

        recommendations = predict_from_inputs(business_need, industry, region, weighted_tone, weighted_article_count, employees,
            tag_inputs['Infrastructure'], tag_inputs['Data'], tag_inputs['AI'], tag_inputs['Security'], tag_inputs['Collaboration'],
            tag_inputs['Sustainability'], tag_inputs['Customer Experience'], tag_inputs['Supply Chain'], tag_inputs['Manufacturing'],
            model_full, preprocessor_full, item_features, product_names)
        
        inputs_summary = {
            "Company":               company_name or "—",
            "Business need":         business_need,
            "Industry":              industry,
            "Region":                region,
            "Employees":             employees,
            "Weighted tone":         round(weighted_tone, 2),
            "Weighted art. count":   round(weighted_article_count, 3),
            "Issue tags selected":   ", ".join([t for t, v in tag_inputs.items() if v]) or "None"
        }
        
        inputs_df = (
            pd.DataFrame.from_dict(inputs_summary, orient="index", columns=["Value"])
            .rename_axis("")
        )

# Similar Cases
        status.write("🗂️ Looking up similar cases")

        sim_cases = find_similar_cases_full(pd.DataFrame({
            'business_need': [business_need],
            'industry': [industry],
            'region': [region],
            'weighted_tone': [weighted_tone],
            'weighted_article_count': [weighted_article_count],
            'employees': [employees],
            **{tag: [tag_inputs[tag]] for tag in tags}
        }))

# Email Generation
        status.write("✍️ Drafting outreach email")

        email_txt = generate_email(sim_cases, recommendations, business_need, industry, region)

# Industry News and Trends
        status.write("📰 Generating industry trends")

        news_headlines, news_text = get_industry_news(industry)
        news_headlines, trends = generate_trends(news_headlines, news_text, industry)

        status.update(label="All done!", state="complete")

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["📌 Recommendations", "📂 Similar Cases", "🎯 Sales Pitch"])

with tab1:
    st.subheader("Top 5 Recommended Products")
    st.dataframe(pd.DataFrame(recommendations, columns=["Product", "Score"]))
    with st.expander("View Input Parameters"):
        st.markdown("These parameters were used to generate the recommendations:")
        st.table(inputs_df)

with tab2:
    col_order=["similarity", "company_name_cleaned", "industry", "region", "employees", "business_need", "related_list", "url", "weighted_tone", "weighted_article_count",
                "Infrastructure", "Data", "AI", "Security", "Collaboration", "Sustainability", "Customer Experience", "Supply Chain", "Manufacturing", "related_products"]
    st.subheader("Most Similar Use Cases")
    sim_df = pd.DataFrame([{**c, **c['full_row']} for c in sim_cases]).drop(columns=['full_row'])
    sim_df = sim_df[col_order]
    st.dataframe(sim_df)
    
with tab3:
    st.subheader("🎯 Sales Story Generator")

    st.markdown("### ✉️ Suggested Outreach Email")
    st.text_area("Generated Email", email_txt, height=250)

    st.download_button("Download Email (.txt)", email_txt, file_name="sales_email.txt")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in email_txt.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output("sales_email.pdf")
    with open("sales_email.pdf", "rb") as f:
        pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="sales_email.pdf">Download Email (.pdf)</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Industry Trends You Should Know")

    st.markdown("**📰 Top Headlines:**")
    for hl in news_headlines:
        st.markdown(hl)

    st.markdown("**🧠 Key Industry Trends:**")
    st.markdown(f"<div style='line-height: 1.6'>{trends.replace(chr(10), '<br><br>')}</div>", unsafe_allow_html=True)