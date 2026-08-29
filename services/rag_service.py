import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from config import GROQ_API_KEY
from datetime import datetime
import re

VECTORSTORE_DIR = "vectorstore"

_embeddings = None
_llm = None
_vectorstore = None
_cached_available_years = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
    return _llm

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=get_embeddings()
        )
    return _vectorstore

def invalidate_years_cache():
    global _cached_available_years
    _cached_available_years = None

def index_notification(file_path: str, title: str, year: int):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["title"] = title
        chunk.metadata["year"] = year
        chunk.metadata["file_path"] = file_path
        chunk.metadata["uploaded_at"] = datetime.now().strftime("%Y-%m-%d")
        chunk.metadata["source_label"] = f"{title} ({year})"

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    invalidate_years_cache()
    print(f"✅ Indexed {len(chunks)} chunks for: {title} (year: {year})")


async def query_notifications(question: str, year: int = None):
    from services.result_extraction_service import get_announcement_vectorstore

    notif_vs = get_vectorstore()
    ann_vs = get_announcement_vectorstore()

    # Dynamic available years discovery from notification vectorstore
    all_docs = notif_vs.get()
    available_years = []
    if all_docs and all_docs.get('metadatas'):
        available_years = sorted(set(
            m.get('year') for m in all_docs['metadatas'] if m and m.get('year')
        ))

    current_year = datetime.now().year
    latest_year = max(available_years) if available_years else current_year

    # Year detection in question
    question_lower = question.lower()
    year_match = re.search(r'\b(20\d{2})\b', question_lower)
    requested_year = int(year_match.group(1)) if year_match else year

    docs = []

    # 1. If explicit year is requested, try filtered search in notifications first
    if requested_year and requested_year in available_years:
        try:
            r_notif = notif_vs.as_retriever(search_kwargs={"k": 4, "filter": {"year": requested_year}})
            docs.extend(r_notif.invoke(question))
        except Exception as e:
            print(f"Filtered vectorstore search error: {e}")

    # 2. Retrieve top chunks from BOTH notification and announcement vectorstores
    try:
        r_notif_all = notif_vs.as_retriever(search_kwargs={"k": 3})
        docs.extend(r_notif_all.invoke(question))
    except Exception as e:
        print(f"Notification vectorstore retrieval error: {e}")

    try:
        r_ann_all = ann_vs.as_retriever(search_kwargs={"k": 3})
        docs.extend(r_ann_all.invoke(question))
    except Exception as e:
        print(f"Announcement vectorstore retrieval error: {e}")

    # 3. Deduplicate retrieved documents by page_content
    seen_content = set()
    unique_docs = []
    for d in docs:
        content_key = d.page_content.strip()
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_docs.append(d)

    context = "\n\n".join([doc.page_content for doc in unique_docs])
    if len(context) > 1500:
        context = context[:1500]

    prompt = f"""You are an official university AI assistant for NIT Kurukshetra.

User Question: {question}
Current System Year: {current_year}

CRITICAL GROUNDING RULES (MUST FOLLOW STRICTLY):
1. STRICTLY FORBIDDEN FROM GENERATING FAKE, GENERIC, OR TEMPLATE NOTICES. NEVER use placeholders like "[Insert Date]", "[Name]", or generate fictional holiday schedules.
2. Rely ONLY on official documents provided in the DOCUMENT CONTEXT below.
3. Check if the DOCUMENT CONTEXT contains any actual notice or circular about the requested topic ("{question}"):
   - CASE A (No Notice Exists): If NO notice or document regarding this specific topic is present in the context below, answer EXACTLY:
     "No official notice regarding **{question}** has been uploaded to the system."
   - CASE B (Previous Year Notice Only): If an official notice for this topic exists in context from a PREVIOUS year (e.g., 2024 or earlier), but NO notice is present for the current year ({current_year}):
     Clearly state at the top:
     "An official notice regarding **{question}** from a previous year is available in the database, but no notice has been uploaded yet for the current year ({current_year})."
     Then present the exact details, title, and year from that previous notice.
   - CASE C (Current Year Notice Exists): If an official notice for the current year ({current_year}) exists in context:
     Present the exact dates and official details directly from the current notice.

DOCUMENT CONTEXT:
{context or "No document context available."}

Answer:"""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content