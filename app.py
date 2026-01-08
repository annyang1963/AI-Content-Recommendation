import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load data
df = pd.read_parquet("lesson_embeddings_openai.parquet")

# Init OpenAI client
key = 'voc-1729574681126677149492966d214ffce5460.25792363'
client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key = key
)

def search_lessons(query, df, client, top_k=10):

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding

    lesson_embeddings = np.array(df['openai_embedding'].tolist())
    similarities = cosine_similarity([query_embedding], lesson_embeddings)[0]

    df_temp = df.copy()
    df_temp['similarity'] = similarities

    df_temp = df_temp.sort_values(by='similarity', ascending=False)

    # Drop duplicates by lesson title (keep the most relevant version)
    df_temp = df_temp.drop_duplicates(subset='lesson_title', keep='first')

    # Return top results
    return df_temp.head(top_k)



# intent classification
# intent classification
def classify_query_scope_gpt(query, client):
    prompt = f"""
You are an expert in learning content classification.

Given a user query, classify their learning intent into one of the following scopes:

- `lesson`: A very specific skill or technique, usually taught in a single lesson.  
  *Examples: "sql join", "python dictionaries", "design phase", "data cleaning", "for loops in C++"*

- `course`: A broader topic that includes multiple lessons, such as a course or unit.  
  *Examples: "practical statistics", "intro to python", "advanced data analysis", "sql fundamentals", "web development with Flask"*

- `nanodegree`: A full learning path or specialization made up of multiple courses, often covering an entire role or domain.  
  *Examples: "intermediate java", "generative AI", "data scientist", "android app developer", "machine learning engineer"*

---

### Example Queries:

| Query                          | Scope     |
|-------------------------------|-----------|
| "sql join"                    | lesson    |
| "polishing dashboards"        | lesson    |
| "python dictionaries"         | lesson    |
| "intro to python"             | course    |
| "web development with Flask"  | course   |
| "generative AI"               | nanodegree   |
| "data scientist"              | nanodegree  |
| "machine learning engineer"     | nanodegree  |

---

### Now classify this user query:

Query: "{query}"

Respond with one word only:  
`lesson`, `course`, or `nanodegree`

"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower()


import streamlit as st

def smart_recommend(query, df, client, top_k=50):
    # Step 1: Detect scope using GPT
    scope = classify_query_scope_gpt(query, client)
    st.markdown(f"### 🔍 Detected scope: `{scope.upper()}` for query: _{query}_")

    # Step 2: Semantic search
    results = search_lessons(query, df, client, top_k=top_k)

    # Step 3: Precompute top results
    top_lessons = results.drop_duplicates(subset='lesson_title')
    top_parts = results.drop_duplicates(subset='part_title')
    top_programs = results.drop_duplicates(subset='program_title')

    # Step 4: Display based on scope
    if scope == "nanodegree":
        st.markdown("## 🏫 Recommended Programs:")
        for _, row in top_programs.head(2).iterrows():
            st.write("—" * 50)
            st.subheader(row['program_title'])
            summary = row['part_summary'] if pd.notna(row['part_summary']) else "No summary available."
            st.markdown(f"📝 _Sample Course Summary_: {summary[:300]}")

        st.markdown("## 📚 Recommended Courses:")
        for _, row in top_parts.head(3).iterrows():
            st.write("—" * 50)
            st.subheader(row['part_title'])
            summary = row['part_summary'] if pd.notna(row['part_summary']) else "No summary available."
            st.markdown(f"📝 _Course Summary_: {summary[:300]}")
            st.markdown(f"🏫 _Program_: {row['program_title']}")

    elif scope == "course":
        st.markdown("## 📚 Recommended Courses:")
        for _, row in top_parts.head(3).iterrows():
            st.write("—" * 50)
            st.subheader(row['part_title'])
            summary = row['part_summary'] if pd.notna(row['part_summary']) else "No summary available."
            st.markdown(f"📝 _Course Summary_: {summary[:300]}")
            st.markdown(f"🏫 _Program_: {row['program_title']}")

        st.markdown("## 🎯 Recommended Lessons:")
        for _, row in top_lessons.head(3).iterrows():
            st.write("—" * 50)
            st.subheader(row['lesson_title'])
            summary = row['lesson_summary'] if pd.notna(row['lesson_summary']) else "No summary available."
            st.markdown(f"📝 _Lesson Summary_: {summary[:300]}")
            st.markdown(f"📚 _Course_: {row['part_title']}")

    elif scope == "lesson":
        st.markdown("## 🎯 Recommended Lessons:")
        for _, row in top_lessons.head(5).iterrows():
            st.write("—" * 50)
            st.subheader(row['lesson_title'])
            summary = row['lesson_summary'] if pd.notna(row['lesson_summary']) else "No summary available."
            st.markdown(f"📝 _Lesson Summary_: {summary[:300]}")
            st.markdown(f"📚 _Course_: {row['part_title']}")

    else:
        st.warning(f"⚠️ Unrecognized scope `{scope}`. Showing lesson-level results.")
        for _, row in top_lessons.head(5).iterrows():
            st.write("—" * 50)
            st.subheader(row['lesson_title'])
            summary = row['lesson_summary'] if pd.notna(row['lesson_summary']) else "No summary available."
            st.markdown(f"📝 _Lesson Summary_: {summary[:300]}")
            st.markdown(f"📚 _Course_: {row['part_title']}")



st.title("AI Content Recommender")
st.markdown("Search for learning content using natural language.")

# Input fields
text_search = st.text_input("What do you want to learn?", placeholder="e.g. sql joins, learn python, generative AI")

# Optional difficulty filter (can integrate later)
# difficulty_level = st.selectbox("Choose difficulty level (optional)", ["All", "Beginner", "Intermediate", "Advanced"])

# Run search on button press
if st.button("Search"):
    if not text_search.strip():
        st.warning("Please enter a query.")
    else:
        smart_recommend(text_search, df, client, top_k=50)

