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
import cohere

# --- Load data ---
df = pd.read_csv("clean_data_model.csv")
product_descriptions = pd.read_csv("clean_product_descriptions.csv")

# --- Preparation ---
df['related_list'] = df['related_products'].apply(eval)
issue_cols = [
    "Infrastructure", "Data", "AI", "Security", "Collaboration",
    "Sustainability", "Customer Experience", "Supply Chain", "Manufacturing"
]
df["issue_tags"] = (
    df[issue_cols]
    .apply(lambda r: [c for c in issue_cols if r[c] == 1], axis=1)
)
region_order = ['North America', 'South America', 'Nordics', 'Western Europe', 'Central Europe','Eastern Europe',
                    'Southern Europe', 'Africa', 'Middle East', 'East Asia', 'Southeast Asia',  'South Asia', 'Oceania']
employees_order = ['1-49 employees', '50-999 employees', '1,000-9,999 employees', '10,000+ employees']
DESC_LOOKUP = dict(
    zip(product_descriptions["related_products"],
        product_descriptions["descriptions"])
)

# --- API keys ---
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# --- Model Training ---
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

# --- Helper Functions ---
def predict_from_inputs(
    new_row_df, model,
    preprocessor, item_features,
    product_names, top_n=5
    ):
    
    X_user = preprocessor.transform(new_row_df)
    user_features = csr_matrix(X_user)
    
    n_items = len(product_names)
    scores = model.predict(
        user_ids=0,
        item_ids=np.arange(n_items),
        user_features=user_features,
        item_features=item_features
    )

    top_indices = np.argsort(-scores)[:top_n]
    return [(product_names[i], scores[i]) for i in top_indices]

def find_similar_cases_full(
    new_row_df, predicted_products,
    X_full, R_full,
    df, product_names,
    preprocessor_full, top_k_cases=5
    ):

    x_new = preprocessor_full.transform(new_row_df)
    u_new = csr_matrix(x_new)

    n_items = len(product_names)
    prod_mask = np.zeros(n_items, dtype=int)
    name_to_idx = {name: idx for idx, name in enumerate(product_names)}
    for prod in predicted_products:
        idx = name_to_idx.get(prod)
        if idx is not None:
            prod_mask[idx] = 1

    f_new = np.hstack([x_new.ravel(), prod_mask])
    F_hist = np.hstack([X_full, R_full.toarray()])

    nn = NearestNeighbors(
        n_neighbors=min(top_k_cases, F_hist.shape[0]),
        metric='cosine'
    ).fit(F_hist)
    dist, idxs = nn.kneighbors(f_new.reshape(1, -1))
    sims = 1 - dist[0]
    neighbors = idxs[0]

    results = []
    for sim, i in zip(sims, neighbors):
        row = df.iloc[i]
        results.append({
            "similarity":          float(sim),
            "index":               int(i),
            "business_need":       row['business_need'],
            "industry":            row['industry'],
            "region":              row['region'],
            "employees":           row['employees'],
            "related_list":        row['related_list'],
            "issue_tags":          row['issue_tags'],
            "company_name_cleaned": row.get("company_name_cleaned", row.get("company_name", "—")),
            "full_row":            row.to_dict()
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

def to_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text.splitlines():
        pdf.multi_cell(0, 10, line)

    return pdf.output(dest="S").encode("latin-1")

def ask_llm(prompt, model='command-r-plus', max_tokens=300):
    try:
        co = cohere.Client(COHERE_API_KEY)
        response = co.generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens
            )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"⚠️ Cohere API error: {str(e)}"
    
def generate_project_plan(new_row_df, recommendations_df):
    company_name = new_row_df['company_name'].values[0]
    business_need = new_row_df['business_need'].values[0]
    industry = new_row_df['industry'].values[0]
    region = new_row_df['region'].values[0]
    employees = new_row_df['employees'].values[0]
    issues = new_row_df['issue_tags'].values[0]

    issues_str = ", ".join(issues) if isinstance(issues, list) else issues
    product_descriptions_str = "\n".join([f"- {p}: {d}" for p, d in zip(recommendations_df['Product'], recommendations_df['Description'])])

    prompt = f"""
    You're a Microsoft sales advisor writing a project plan for a potential client.
    
    1. For each of the recommended Microsoft products below, use the descriptions to show how they can help {company_name} with their {business_need} needs in the {industry} industry in {region} and tackle their issues.
    
    Issues:
    {issues_str}
    
    Product Descriptions:
    {product_descriptions_str}

    2. Use the information you have to create a project plan that outlines the architecture of the solution, the expected outcomes, and how it will help {company_name} achieve their business goals.

    3. Create a roadmap for the implementation of the solution, including key milestones and deliverables.
    """.strip()

    project_plan_content = ask_llm(prompt, max_tokens=1500)
    
    return project_plan_content, prompt

def generate_email(new_row_df, predicted_products, sim_df, tone):
    company_name = new_row_df['company_name'].values[0]
    business_need = new_row_df['business_need'].values[0]
    industry = new_row_df['industry'].values[0]
    region = new_row_df['region'].values[0]
    employees = new_row_df['employees'].values[0]
    issues = new_row_df['issue_tags'].values[0]

    issues_str = ", ".join(issues) if isinstance(issues, list) else issues
    products_str = ", ".join(predicted_products)
    sim_cases_str = "\n".join([f"- {c['company_name_cleaned']} ({c['industry']}, {c['region']}, {c['employees']}, {c['business_need']}, {c['related_list']}, {c['issue_tags']})" for c in sim_df])

    prompt = f"""
    You're a Microsoft sales advisor writing a {tone} and engaging sales pitch email to a potential client.
    Convince {company_name} how Microsoft products can help with their {business_need} needs in the {industry} industry.

    Tailor your arguments to the company's specific needs and challenges.
    Highlight how the recommended Microsoft products can address the issues.
    Base your arguments on how similar companies have successfully used these products without naming the companies.

    The email should be structured as follows:
    1. Start with a short sentence explaining a typical challenge for a company in the {industry} sector with issues in {issues_str} and a need for {business_need} in {region}.
    2. Present Microsoft's recommended products: {products_str}.
    3. Summarize how these products address the issues {issues_str} and lead to qualitative and/or quantitative improvements of {business_need}.
    4. Include a brief story of the most relevant similar case and the results Microsoft achieved for that company, but don't name it: {sim_cases_str}.
    5. Wrap up with a positive outlook for digital transformation when partnering with Microsoft.

    Write the email in paragraphs without headings
    Please return only the email content without any additional text or explanations.
    """.strip()

    email_content = ask_llm(prompt, max_tokens=500)
    
    return email_content, prompt

def generate_linkedin(new_row_df, tone):
    company_name = new_row_df['company_name'].values[0]
    business_need = new_row_df['business_need'].values[0]
    industry = new_row_df['industry'].values[0]
    region = new_row_df['region'].values[0]
    employees = new_row_df['employees'].values[0]
    issues = new_row_df['issue_tags'].values[0]

    prompt = f"""
    You're a Microsoft sales advisor writing a brief, {tone} and engaging sales pitch via LinkedIn to a potential client.
    Convince {company_name} how Microsoft products can help with their {business_need} needs in the {industry} industry in {region} tackling their {issues} issues.

    Tailor your arguments to the company's specific needs and challenges.
    Highlight how the recommended Microsoft products can address the issues.
    Base your arguments on how similar companies have successfully used these products without naming the companies.

    Make sure to stay below 200 characters at all time by prioritizing what you think would have the biggest effect on convincing the client.
    Please return only the message content without any additional text or explanations.
    """.strip()

    linkedin_content = ask_llm(prompt, max_tokens=90)
    
    return linkedin_content, prompt

def generate_trends(news_headlines, news_text, industry):
    if news_text and news_headlines:
        
        prompt = f"""
        Please extract exactly three current trends from the following news descriptions related to the {industry} industry.
        Format your response as a numbered list with short, clear sentences. Each point should be no longer than 2 lines:
        {news_text}
        """.strip()

        trends = ask_llm(prompt, max_tokens=90)

        return news_headlines, trends, prompt
        
    else:
        return "No recent news available for this industry."

# --- UI ---
st.set_page_config(page_title="Microsoft Product Recommender", layout="wide")
st.title("🔍 Microsoft Product Recommender")
st.markdown("This tool helps Microsoft sales teams recommend products based on company features, historical sales, and news sentiment. Get started by filling out the form in the sidebar!")

# --- Sidebar ---
with st.sidebar:
    st.header("Input Parameters")
    st.markdown("---")
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
    n_recs = st.slider("No. of recommended products", min_value=1,  max_value=10,  value=5,  step=1)
    st.markdown("---")
    tone = st.selectbox("Email Style", ["Formal", "Playful", "Concise"])
    st.markdown("---")
    trigger = st.button("🚀 Generate Insights", use_container_width=True)

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["📌 Recommendations", "📂 Similar Cases", "🎯 Sales Pitch"])

if trigger:
    new_row_df = pd.DataFrame([{
        "company_name":            company_name,
        "business_need":           business_need,
        "industry":                industry,
        "region":                  region,
        "employees":               employees,
        "weighted_tone":           0.0,  # Placeholder
        "weighted_article_count":  0.0,  # Placeholder
        **tag_inputs
    }])

    for col in issue_cols:
        if col not in new_row_df.columns:
            new_row_df[col] = 0

    new_row_df["issue_tags"] = (
        new_row_df[issue_cols]
        .apply(lambda r: [c for c in issue_cols if r[c] == 1], axis=1)
    )

    file_company_name = (company_name or "company").strip().lower().replace(" ", "_")

# Recommendations
    with tab1:
        with st.status("Generating recommendations...", expanded=False) as status:

            status.write("🔍 Fetching company news")
            weighted_tone, weighted_article_count = get_company_news(company_name, df)
            new_row_df['weighted_tone'] = weighted_tone
            new_row_df['weighted_article_count'] = weighted_article_count

            status.write("♟️ Computing recommendations")
            recommendations = predict_from_inputs(
                new_row_df=new_row_df,
                model=model_full,
                preprocessor=preprocessor_full,
                item_features=item_features,
                product_names=product_names,
                top_n=n_recs
            )
            predicted_products = [p for p, _ in recommendations]

            recommendations = [
                (p, DESC_LOOKUP.get(p, "No description available"), score)
                for p, score in recommendations
            ]
            recommendations_df = pd.DataFrame(recommendations, columns=["Product", "Description", "Score"])

            inputs_summary = {
                "Company":               company_name or "—",
                "Business need":         business_need,
                "Industry":              industry,
                "Region":                region,
                "Employees":             employees,
                "Weighted tone":         weighted_tone,
                "Weighted art. count":   weighted_article_count,
                "Issue tags selected":   ", ".join([t for t, v in tag_inputs.items() if v]) or "None"
            }
            inputs_df = (
                pd.DataFrame.from_dict(inputs_summary, orient="index", columns=["Value"])
                .rename_axis("")
            )

            status.update(label="All done!", state="complete")

        st.subheader(f"Top {n_recs} Recommended Products")
        st.dataframe(recommendations_df)
        with st.expander("View Input Parameters"):
            st.table(inputs_df)

# Similar Cases
    with tab2:
        with st.status("Finding similar cases...", expanded=False) as status:
            
            status.write("🗂️ Looking up similar cases")
            sim_cases = find_similar_cases_full(
                new_row_df=new_row_df,
                predicted_products=predicted_products,
                X_full=X_full,
                R_full=interactions_full,
                df=df,
                product_names=product_names,
                preprocessor_full=preprocessor_full,
                top_k_cases=5
            )

            status.write("📊 Formatting results")
            col_order = ["similarity", "company_name_cleaned", "industry", "region", "employees", "business_need", "related_list", "issue_tags", "url", "weighted_tone", "weighted_article_count"]
            sim_df = pd.DataFrame([{**c, **c['full_row']} for c in sim_cases]).drop(columns=['full_row'])
            sim_df = sim_df[col_order]

            status.update(label="All done!", state="complete")
        
        st.subheader("Most Similar Use Cases")
        st.dataframe(sim_df)
        with st.expander("View Input Parameters"):
            st.table(inputs_df)

# Sales Pitch
    with tab3:
        with st.status("Generating sales pitch...", expanded=False) as status:

            status.write("📑 Creating project plan")
            project_plan_content, project_plan_prompt = generate_project_plan(
                new_row_df=new_row_df,
                recommendations_df=recommendations_df
            )

            status.write("💬 Generating LinkedIn message")
            linkedin_content, linkedin_prompt = generate_linkedin(
                new_row_df=new_row_df,
                tone=tone.lower()
            )

            status.write("✉️ Generating outreach email")
            email_content, email_prompt = generate_email(
                new_row_df=new_row_df,
                predicted_products=predicted_products,
                sim_df=sim_cases,
                tone=tone.lower()
            )
            email_txt = f"""Dear {company_name} Team,\n
                        {email_content}\n
                        Best regards,\n
                        Your Microsoft Sales Team"""

            status.write("📰 Fetching industry news")
            news_headlines, news_text = get_industry_news(industry)

            status.write("📈 Analyzing trends")
            news_headlines, trends, trends_prompt = generate_trends(news_headlines, news_text, industry)

            status.update(label="All done!", state="complete")

        st.subheader("📑 Project Plan")
    
        st.markdown("Suggested Project Plan")
        st.text_area("Project Plan", project_plan_content, height=250)
        with st.expander("Prompt Used", expanded=False):
            st.code(project_plan_prompt, language=None)

        col_txt, col_pdf = st.columns(2)
        with col_txt:
            st.download_button(
                "📄 Download Project Plan (.txt)",
                data=project_plan_content,
                file_name=f"{file_company_name}_project_plan.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_pdf:
            st.download_button(
                "📑 Download Project Plan (.pdf)",
                data=to_pdf_bytes(project_plan_content),
                file_name=f"{file_company_name}_project_plan.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("📈 Industry Trends")

        st.markdown("Key Industry Trends:")
        st.markdown(f"<div style='line-height: 1.6'>{trends.replace(chr(10), '<br><br>')}</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")
        st.markdown("Top Headlines:")
        for hl in news_headlines:
            st.markdown(hl)
        with st.expander("Prompt Used", expanded=False):
            st.code(trends_prompt, language=None)

        st.markdown("---")
        st.subheader("✉️ Outreach Proposal")

        st.markdown("Suggested LinkedIn Message")
        st.text_area("Generated LinkedIn Message", linkedin_content, height=100)
        with st.expander("Prompt Used", expanded=False):
            st.code(linkedin_prompt, language=None)
        
        col_txt, col_pdf = st.columns(2)
        with col_txt:
            st.download_button(
                "📄 Download LinkedIn Message (.txt)",
                data=linkedin_content,
                file_name=f"{file_company_name}_linkedin_message.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_pdf:
            st.download_button(
                "📑 Download LinkedIn Message (.pdf)",
                data=to_pdf_bytes(linkedin_content),
                file_name=f"{file_company_name}_linkedin_message.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown("Suggested Outreach Email")
        st.text_area("Generated Email", email_txt, height=250)
        with st.expander("Prompt Used", expanded=False):
            st.code(email_prompt, language=None)

        col_txt, col_pdf = st.columns(2)
        with col_txt:
            st.download_button(
                "📄 Download Email (.txt)",
                data=email_txt,
                file_name=f"{file_company_name}_sales_email.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_pdf:
            st.download_button(
                "📑 Download Email (.pdf)",
                data=to_pdf_bytes(email_txt),
                file_name=f"{file_company_name}_sales_email.pdf",
                mime="application/pdf",
                use_container_width=True
            )