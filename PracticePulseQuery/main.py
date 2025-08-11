import os
import re
import requests
import mysql.connector
import pandas as pd
import plotly.express as px
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent
from typing import Tuple

# ----------------- CONFIG -----------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Zeal@94269",
    "database": "appointments_db"
}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

SAFE_MODE = True  # Change to False if you want to allow writes
# -------------------------------------------

def ollama_generate(prompt: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    r.raise_for_status()
    return r.json().get("text", "").strip()

def get_schema_info():
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = [t[0] for t in cur.fetchall()]
    schema = ""
    for t in tables:
        cur.execute(f"SHOW CREATE TABLE {t}")
        schema += f"\n\n{cur.fetchone()[1]}"
    cur.close()
    conn.close()
    return schema

# ----------------- SAFETY CHECK -----------------
def is_query_safe(sql: str) -> Tuple[bool, str]:
    sql_lower = sql.lower().strip()

    # Block certain statements completely
    forbidden = ["drop ", "truncate ", "alter "]
    if any(word in sql_lower for word in forbidden):
        return False, "Query contains dangerous statement (DROP/TRUNCATE/ALTER)."

    # In safe mode, only allow SELECT
    if SAFE_MODE and not sql_lower.startswith("select"):
        return False, "SAFE_MODE is ON — only SELECT queries are allowed."

    # Block DELETE without WHERE
    if sql_lower.startswith("delete") and "where" not in sql_lower:
        return False, "DELETE without WHERE is not allowed."

    return True, ""

def execute_query(sql: str):
    safe, msg = is_query_safe(sql)
    if not safe:
        return f"BLOCKED: {msg}"

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        if sql.strip().lower().startswith("select"):
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            return df
        else:
            conn.commit()
            return f"{cur.rowcount} row(s) affected."
    finally:
        cur.close()
        conn.close()

# ----------------- AUTOGEN SETUP -----------------
schema_info = get_schema_info()

assistant = AssistantAgent(
    name="MySQLAgent",
    system_message=f"""
You are a MySQL expert.
Given a user request and DB schema, write a correct and SAFE MySQL query.
- If SAFE_MODE is ON, generate only SELECT statements.
- Never use DROP, TRUNCATE, ALTER.
- Always include WHERE for DELETE.
Schema:
{schema_info}
""",
    llm_config = {
    "cache_seed": 42,
    "config_list": [{
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"  # dummy key, Ollama ignores it
    }]
}

)

user_proxy = UserProxyAgent(
    name="UserExecutor",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config=False,
)

@user_proxy.register_for_execution()
def run_sql_query(query: str):
    return execute_query(query)

# ----------------- STREAMLIT UI -----------------
st.title("AutoGen MySQL LLM Agent — SAFE MODE")
st.caption("SAFE_MODE prevents dangerous queries. Set SAFE_MODE=False to allow writes.")

user_request = st.text_input("Ask your database", "Show me total sales by month for 2024")

if st.button("Run"):
    with st.spinner("Agent is thinking..."):
        assistant.initiate_chat(
            user_proxy,
            message=f"User request: {user_request}. Please provide SQL to answer this."
        )

    sql_query = assistant.last_message()["content"].strip("` \n")
    st.subheader("Generated SQL")
    st.code(sql_query, language="sql")

    result = execute_query(sql_query)
    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
        if result.shape[0] > 0 and result.shape[1] >= 2:
            fig = px.bar(result, x=result.columns[0], y=result.columns[1], title="Query Result")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(result)



# streamlit run PracticePulseQuery\main.py
