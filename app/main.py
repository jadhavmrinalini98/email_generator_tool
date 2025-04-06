import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://jobs.apple.com/en-us/details/200576351/junior-server-engineer-health-software")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            st.info("🔄 Scraping job posting and generating email...")

            # Load job description
            loader = WebBaseLoader([url_input])
            raw_content = loader.load().pop().page_content
            data = clean_text(raw_content)

            # Load portfolio data before querying links
            portfolio.load_portfolio()

            # Extract job details
            jobs = llm.extract_jobs(data)
            if not jobs:
                st.warning("⚠️ No jobs found in the scraped content.")
                return

            for job in jobs:
                skills = job.get('skills', [])

                # Ensure `skills` is a list of strings, not a dictionary
                if isinstance(skills, dict):
                    skills = list(skills.values())  # Extract dictionary values into a list

                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')


        except Exception as e:
            st.error(f"❌ An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)

