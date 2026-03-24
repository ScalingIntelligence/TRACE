# SPDX-License-Identifier: Apache-2.0
# Weighted LoRA merge endpoint for combining multiple adapters

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.api_router import models
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.lora.protocol import CreateWeightedLoRARequest
from vllm.logger import init_logger

logger = init_logger(__name__)


def attach_weighted_lora_router(app: FastAPI):
    """Register the /v1/create_weighted_lora endpoint directly on the app."""

    @app.post("/v1/create_weighted_lora",
              dependencies=[Depends(validate_json_request)])
    async def create_weighted_lora(
        request: CreateWeightedLoRARequest, raw_request: Request
    ):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.create_weighted_lora(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(),
                status_code=response.error.code,
            )
        return Response(status_code=200, content=response)

    logger.info("Weighted LoRA merge endpoint registered: "
                "POST /v1/create_weighted_lora")
