"""
Run with:
    conda activate vibe-coders
    cd vibe-coders
    PYTHONPATH=. python scripts/serve_ui.py

UI:  http://localhost:8000/ui
API: http://localhost:8000/docs
"""
import pathlib
import uvicorn
from fastapi.responses import FileResponse
from backend.main import app

UI_PATH = pathlib.Path(__file__).parent.parent / "ui" / "index.html"


@app.get("/ui", include_in_schema=False)
def serve_ui():
    return FileResponse(UI_PATH, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
