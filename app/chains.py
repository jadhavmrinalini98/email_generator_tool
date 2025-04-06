import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.0,
                            groq_api_key ='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',
                            model="llama-3.1-8b-instant"
                            )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")

        # Ensure the output is always a list of dictionaries
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        # Convert job dictionary to a readable string format
        job_description = "\n".join([f"{key.capitalize()}: {value}" for key, value in job.items()])

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mrinalini, a highly motivated graduate student with a Master's degree in Information Systems from Northeastern University. 
            Your goal is to write a **persuasive, professional, and engaging cold email** to the hiring manager at the company regarding the job mentioned above.

            The email should:
            - Express genuine enthusiasm for the role and the company.
            - Highlight **Mrinalini’s technical expertise, problem-solving skills, and relevant experiences** that align with the job requirements.
            - Incorporate **specific skills or achievements** that demonstrate Mrinalini’s ability to contribute effectively.
            - Integrate the most relevant portfolio links from the following list to showcase Mrinalini's past work: {link_list}.
            - End with a **clear call-to-action**, such as requesting a conversation or expressing interest in discussing the role further.

            Keep the tone **confident, concise, and professional**. Personalize the email to make it compelling and avoid generic statements.

            **Do not provide a preamble.**

            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": job_description, "link_list": links})
        return res.content


if __name__ == "__main__":
    os.getenv("GROQ_API_KEY")

