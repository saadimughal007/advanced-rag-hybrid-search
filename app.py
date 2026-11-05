import os,re,tempfile
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer,CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types

load_dotenv()
st.set_page_config(page_title="Advanced RAG Assistant",page_icon="🔎",layout="wide")
st.title("🔎 Advanced RAG Document Assistant")
st.caption("Semantic search + BM25 + Reciprocal Rank Fusion + Cross-encoder reranking + Gemini")
EMBED="sentence-transformers/all-MiniLM-L6-v2"
RERANK="cross-encoder/ms-marco-MiniLM-L-6-v2"

@st.cache_resource
def embedding_model(): return SentenceTransformer(EMBED)
@st.cache_resource
def rerank_model(): return CrossEncoder(RERANK)

def extract(file):
    name=file.name.lower(); data=file.getvalue()
    if name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as f:
            f.write(data); path=f.name
        try: return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        finally: os.unlink(path)
    if name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(suffix=".docx",delete=False) as f:
            f.write(data); path=f.name
        try: return "\n".join(p.text for p in Document(path).paragraphs)
        finally: os.unlink(path)
    return data.decode("utf-8",errors="ignore")

def chunks(text,size=900,overlap=150):
    words=re.sub(r"\s+"," ",text).strip().split(); out=[]; start=0
    while start<len(words):
        end=min(start+size,len(words)); out.append(" ".join(words[start:end]))
        if end==len(words): break
        start=max(end-overlap,start+1)
    return out

def tokens(s): return re.findall(r"\b[a-zA-Z0-9_'-]+\b",s.lower())

def make_index(docs,model):
    em=model.encode([d["text"] for d in docs],normalize_embeddings=True)
    return {"docs":docs,"emb":em,"bm25":BM25Okapi([tokens(d["text"]) for d in docs])}

def hybrid(query,index,model,k=8):
    sem=cosine_similarity(model.encode([query],normalize_embeddings=True),index["emb"])[0]
    kw=index["bm25"].get_scores(tokens(query))
    ranks={}
    for order in (sem.argsort()[::-1],kw.argsort()[::-1]):
        for rank,i in enumerate(order[:k*3],1):
            ranks[int(i)]=ranks.get(int(i),0)+1/(60+rank)
    result=[]
    for i,score in sorted(ranks.items(),key=lambda x:x[1],reverse=True)[:k]:
        result.append({**index["docs"][i],"idx":i,"semantic":float(sem[i]),"bm25":float(kw[i]),"rrf":score})
    return result

def rerank(query,candidates,model,k=4):
    scores=model.predict([(query,x["text"]) for x in candidates])
    ranked=[{**x,"rerank":float(s)} for x,s in zip(candidates,scores)]
    return sorted(ranked,key=lambda x:x["rerank"],reverse=True)[:k]

def gemini_answer(q,selected,key,model_name):
    client=genai.Client(api_key=key)
    ctx="\\n\\n".join(f"[Source: {x['source']} | Chunk: {x['chunk']} ]\\n{x['text']}" for x in selected)
    prompt=f"""Answer using ONLY the document excerpts. If they do not contain enough evidence, respond exactly:
"I couldn't find that information in the uploaded documents."
Do not use outside knowledge. Cite sources as [Source: filename, Chunk: number].
DOCUMENT EXCERPTS:
{ctx}
QUESTION: {q}"""
    response=client.models.generate_content(model=model_name,contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1))
    return response.text or "Empty model response."

with st.sidebar:
    st.header("Configuration")
    key=st.text_input("Gemini API key",value=os.getenv("GEMINI_API_KEY",""),type="password")
    model_name=st.text_input("Gemini model","gemini-2.5-flash")
    chunk_size=st.slider("Chunk size (words)",300,1400,900,100)
    overlap=st.slider("Chunk overlap",0,300,150,25)
    k=st.slider("Hybrid candidates",3,20,8)
    final_k=st.slider("Final context chunks",1,8,4)

t1,t2,t3=st.tabs(["📚 Knowledge Base","💬 Ask Documents","🧪 Evaluation"])
with t1:
    st.subheader("Upload or paste documents")
    uploads=st.file_uploader("PDF, DOCX, TXT, Markdown",type=["pdf","docx","txt","md"],accept_multiple_files=True)
    pasted=st.text_area("Or paste document text",height=130)
    if st.button("Process documents",type="primary"):
        docs=[]
        for f in uploads or []:
            text=extract(f)
            for i,c in enumerate(chunks(text,chunk_size,overlap)):
                docs.append({"source":f.name,"chunk":i,"text":c})
        for i,c in enumerate(chunks(pasted,chunk_size,overlap)):
            docs.append({"source":"Pasted text","chunk":i,"text":c})
        if docs:
            with st.spinner("Creating embeddings and BM25 index..."):
                st.session_state.idx=make_index(docs,embedding_model())
            st.success(f"Knowledge base indexed: {len(docs)} chunks.")
        else: st.warning("Upload a document or paste text first.")
    if "idx" in st.session_state:
        st.metric("Indexed chunks",len(st.session_state.idx["docs"]))
        st.dataframe([{"Source":d["source"],"Chunk":d["chunk"],"Words":len(d["text"].split()),"Preview":d["text"][:160]} for d in st.session_state.idx["docs"]],use_container_width=True)
with t2:
    q=st.text_input("Ask a question about your documents")
    if st.button("Run Advanced RAG",type="primary"):
        if "idx" not in st.session_state: st.warning("Process documents first.")
        elif not q.strip(): st.warning("Enter a question.")
        elif not key: st.warning("Enter Gemini API key in the sidebar or .env.")
        else:
            try:
                with st.spinner("Semantic + BM25 retrieval..."): candidates=hybrid(q,st.session_state.idx,embedding_model(),k)
                with st.spinner("Cross-encoder reranking..."): selected=rerank(q,candidates,rerank_model(),final_k)
                with st.spinner("Generating grounded answer..."): answer=gemini_answer(q,selected,key,model_name)
                st.session_state.last={"q":q,"candidates":candidates,"selected":selected,"answer":answer}
            except Exception as e: st.error(f"Pipeline error: {e}")
    if "last" in st.session_state:
        x=st.session_state.last
        st.subheader("Answer"); st.write(x["answer"])
        a,b=st.columns(2)
        with a:
            st.markdown("**Hybrid results before reranking**")
            st.dataframe([{"Source":d["source"],"Chunk":d["chunk"],"Semantic":round(d["semantic"],4),"BM25":round(d["bm25"],4),"RRF":round(d["rrf"],5)} for d in x["candidates"]],use_container_width=True)
        with b:
            st.markdown("**Selected results after reranking**")
            st.dataframe([{"Source":d["source"],"Chunk":d["chunk"],"Cross-encoder score":round(d["rerank"],4)} for d in x["selected"]],use_container_width=True)
        with st.expander("Inspect context sent to Gemini"):
            for d in x["selected"]:
                st.markdown(f"**{d['source']} — chunk {d['chunk']}**"); st.write(d["text"])
with t3:
    st.subheader("Basic RAG evaluation")
    st.markdown("""Run two tests:
- **Supported question:** ask for a fact explicitly present in your document.
- **Unsupported question:** ask for a fact absent from all uploaded documents.

Check whether the relevant chunk appears in retrieval, whether reranking promotes it, whether the answer cites its source, and whether the system abstains for missing information.""")
    if "last" in st.session_state: st.json({"question":st.session_state.last["q"],"hybrid_candidates":len(st.session_state.last["candidates"]),"final_context":len(st.session_state.last["selected"]),"answer":st.session_state.last["answer"]})
