import os
from os.path import join, dirname
from dotenv import load_dotenv
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Sidebar contents
st.sidebar.title('TopBoss PDF ChatBot')
st.sidebar.markdown('''
    ## About
    Try uploading your PDF file and ask any questions you need to know
''')

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
# load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf_files = st.file_uploader("Upload your PDF files", type='pdf', accept_multiple_files=True)

    if pdf_files is None or len(pdf_files) == 0:
        st.error("Please upload a PDF file.")
        return

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        
        if pdf_reader is not None and len(pdf_reader.pages) > 0:
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            store_name = pdf.name[:-4]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=2000,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

            query = st.text_input("Ask questions about your PDF file:", key="query_input")

            # Get suggested queries based on PDF content or commonly asked questions
            suggested_queries = get_suggested_queries(VectorStore, query)

            # Display suggested queries as a dropdown list
            selected_query = st.selectbox("Suggested queries:", suggested_queries, key="suggested_queries")

            if selected_query:
                query = selected_query

            if query:
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                
                st.write(response)
                
                # Allow user to rate the answer
                col1, col2 = st.columns(2)
                if col1.button("Like"):
                    st.success("Thank you for your feedback. You liked the answer.")
                if col2.button("Dislike"):
                    send_dislike_email(response)
                    st.success("Thank you for your feedback. An email has been sent with your dislike.")

def get_suggested_queries(VectorStore, query):
    # Implement your suggestion or autocomplete logic here
    commonly_asked_questions = [
       "What is the purpose of this document?",
        "Give a brief explanation of the content",
    ]    
    # Filter suggested queries based on the input query
    suggested_queries = [q for q in commonly_asked_questions if query.lower() in q.lower()]
    
    return suggested_queries

def send_dislike_email(response):

    # Get email details from environment variables
    email_address = str(os.environ.get("email_address"))
    email_password = str(os.environ.get("email_password"))
    recipient_email = str(os.environ.get("recipient_email"))


    # Create email message
    msg = MIMEMultipart()
    msg["From"] = email_address
    msg["To"] = recipient_email
    msg["Subject"] = "Disliked Answer"

    # Compose email content
    email_content = f"The user disliked the following answer:\n\n{response}"
    msg.attach(MIMEText(email_content, "plain"))

    # Create SMTP session
    with smtplib.SMTP("smtp-mail.outlook.com", 587) as server:
        server.ehlo()
        server.starttls()

        server.login(email_address, email_password)
        server.sendmail(email_address, recipient_email, msg.as_string())
        server.quit()
if __name__ == '__main__':
    main()
