from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.main import get_db
from src.dataset_processing.dataset_load_utils import process_csv_upload
from src.models import DataPoint, Dataset
from src.schemas import DatasetUploadResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    counts = dict(
        db.query(
            DataPoint.dataset_id,
            func.count(DataPoint.id).label("n_points"),
        )
        .group_by(DataPoint.dataset_id)
        .all()
    )
    embedded = dict(
        db.query(
            DataPoint.dataset_id,
            func.count(DataPoint.embedding).label("has_embeddings"),
        )
        .group_by(DataPoint.dataset_id)
        .all()
    )
    datasets = db.query(Dataset).order_by(Dataset.name).all()
    return [
        {
            "dataset_id": d.id,
            "dataset_name": d.name,
            "n_points": int(counts.get(d.id, 0)),
            "has_embeddings": int(embedded.get(d.id, 0)),
            "description": d.description or "",
        }
        for d in datasets
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


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    name = dataset.name
    db.delete(dataset)
    db.commit()
    return {"dataset_id": dataset_id, "dataset_name": name}
