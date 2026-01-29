from fastapi import APIRouter, HTTPException
from database.adapters import MongoDBAdapter
from model_trainer import run_full_pipeline

