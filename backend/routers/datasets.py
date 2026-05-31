from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.main import get_db
from src.dataset_processing.dataset_load_utils import process_csv_upload
from src.models import DataPoint
from src.schemas import DatasetUploadResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    rows = (
        db.query(
            DataPoint.dataset_name,
            func.count(DataPoint.id).label("n_points"),
            func.count(DataPoint.embedding).label("has_embeddings"),
        )
        .group_by(DataPoint.dataset_name)
        .order_by(DataPoint.dataset_name)
        .all()
    )
    return [
        {
            "dataset_name": row.dataset_name,
            "n_points": int(row.n_points),
            "has_embeddings": int(row.has_embeddings),
        }
        for row in rows
    ]


@router.post("/upload", response_model=DatasetUploadResponse)
def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    generate_embeddings: bool = Form(True),
    db: Session = Depends(get_db),
):
    if not dataset_name.strip():
        raise HTTPException(status_code=400, detail="dataset_name is required")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file")
    try:
        result = process_csv_upload(
            file.file,
            dataset_name=dataset_name,
            db=db,
            generate_embeddings=generate_embeddings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return DatasetUploadResponse(dataset_name=dataset_name, **result)


@router.delete("/{dataset_name}")
def delete_dataset(dataset_name: str, db: Session = Depends(get_db)):
    deleted = (
        db.query(DataPoint)
        .filter(DataPoint.dataset_name == dataset_name)
        .delete(synchronize_session=False)
    )
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    db.commit()
    return {"dataset_name": dataset_name, "deleted": deleted}
