# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.models.api_router import models
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


def attach_router(app: FastAPI):
    if not envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        """If LoRA dynamic loading & unloading is not enabled, do nothing."""
        return
    logger.warning(
        "LoRA dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!"
    )

    @sagemaker_standards.register_load_adapter_handler(
        request_shape={
            "lora_name": "body.name",
            "lora_path": "body.src",
            "load_inplace": "body.load_inplace || `false`",
        },
    )
    @router.post("/v1/load_lora_adapter", dependencies=[Depends(validate_json_request)])
    async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )

        return Response(status_code=200, content=response)

    @sagemaker_standards.register_unload_adapter_handler(
        request_shape={
            "lora_name": "path_params.adapter_name",
        }
    )
    @router.post(
        "/v1/unload_lora_adapter", dependencies=[Depends(validate_json_request)]
    )
    async def unload_lora_adapter(
        request: UnloadLoRAAdapterRequest, raw_request: Request
    ):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )

        return Response(status_code=200, content=response)

    @router.post("/v1/merge_lora_into_base", dependencies=[Depends(validate_json_request)])
    async def merge_lora_into_base(raw_request: Request):
        body = await raw_request.json()
        # Support two formats:
        # 1. Single adapter: {"adapter_path": "/path/to/adapter"}
        # 2. Multi-adapter:  {"adapters": [{"adapter_path": "...", "weight": 0.7}, ...]}
        adapters = body.get("adapters")
        adapter_path = body.get("adapter_path")
        if adapters:
            adapter_spec = adapters  # list of dicts
        elif adapter_path:
            adapter_spec = adapter_path  # single string
        else:
            return JSONResponse(
                content={"error": "adapter_path or adapters is required"}, status_code=400
            )
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.merge_lora_into_base(adapter_spec)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )
        return Response(status_code=200, content=response)

    @router.post("/v1/restore_base_weights")
    async def restore_base_weights(raw_request: Request):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.restore_base_weights()
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )
        return Response(status_code=200, content=response)

    # register the router
    app.include_router(router)
