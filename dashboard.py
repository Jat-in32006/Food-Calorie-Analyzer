import sys
from pathlib import Path

# Add local myenv to path for package resolution
myenv_path = str((Path(__file__).resolve().parent / "myenv").resolve())
if myenv_path not in sys.path and Path(myenv_path).exists():
    sys.path.append(myenv_path)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from dashboard_app.api.endpoints import router as api_router

app = FastAPI(title="NutriScan AI - Food Calorie Analyzer")

# Mount API routes
app.include_router(api_router, prefix="/api")

# Mount Static Files (CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("dashboard_app/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
