# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, Field


class LoadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str
    load_inplace: bool = False


class UnloadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_int_id: int | None = Field(default=None)


class WeightedAdapter(BaseModel):
    """A single adapter with its soft routing weight."""
    name: str       # must match a registered adapter name
    weight: float   # softmax probability from classifier


class CreateWeightedLoRARequest(BaseModel):
    """Request to create a merged LoRA adapter from weighted combination."""
    adapters: list[WeightedAdapter]
    output_name: str | None = None  # auto-generated if omitted
