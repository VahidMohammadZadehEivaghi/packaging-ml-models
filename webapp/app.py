import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

import torch
import numpy as np
from transformers import RobertaTokenizer

import onnxruntime as rt


from pydantic import BaseModel
from typing import Union


class Request(BaseModel):
    text: Union[str, None] = "MLOps is critical for robustness"


artifacts = {}


@asynccontextmanager
async def lifespan(app_: FastAPI):
    artifacts["tokenizer"] = RobertaTokenizer.from_pretrained("roberta-base")
    artifacts["model"] = rt.InferenceSession(
        "../model/roberta-sequence-classification-9.onnx"
    )
    yield artifacts
    artifacts.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def predict(request: Request):
    input_ids = torch.tensor(artifacts["tokenizer"].encode(request.text,
                                                           add_special_tokens=True)).unsqueeze(0)
    if input_ids.requires_grad:
        numpy_func = input_ids.detach().cpu().numpy()
    else:
        numpy_func = input_ids.cpu().numpy()

    inputs = {artifacts["model"].get_inputs()[0].name: numpy_func}
    out = artifacts["model"].run(None, inputs)

    result = np.argmax(out)
    return {"positive": bool(result)}


if __name__ == "__main__":
    uvicorn.run("app:app")




