from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"message": "C# Code Generator API"}

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    return {"generated_text": f"// Generated code for: {request.prompt}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

