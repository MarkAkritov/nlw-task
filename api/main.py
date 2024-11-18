from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import schemas, settings
from .utils import prompt_query, upload_pdf_doc

app = FastAPI(
    title="PDF-Assistant",
    description="AI Assistant for answering questions related to uploaded document.",
    version="0.0.1",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-pdf")
async def post_pdf_doc(file: UploadFile = File(...)):
    """**`POST`** request for uploading PDF document to Vector DB.

    **Args:**
     * *file*: File to be uploaded `.pdf`

    **Raises:**
     * *`HTTPException`*: if the file extension is not supported (Ex.: `.csv`)

    **Returns:**
     * *`JSON`*: messege about file access and upload in the background.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=404, detail="File extension not supported.")

    file_path = settings.DOCS_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    upload_pdf_doc(file_path)

    return {"filename": file.filename}


@app.post("/prompt", response_model=schemas.Answer)
async def prompt(query: schemas.InputPrompt):
    """**`POST`** request answers for provided queries.

    **Args:**
     * *query*: Input prompt as {input: question}.

    **Returns:**
     * *`JSON`*: Answer with context and sources.
    """
    return prompt_query(query=query.model_dump())
