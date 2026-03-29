# Copied from evals/benchmarks/ToolSandbox/tool_sandbox/common/tool_conversion.py
# with langchain_core.pydantic_v1 compatibility fix for langchain-core >= 0.3.
#
# Original: Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Taken from https://github.com/langchain-ai/langchain/blob/
# 90184255f880a26cbdffd7b764deae9be3242ece/libs/core/langchain_core/utils/function_calling.py#L276
# Modified to allow for tool augmentations

import inspect
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

try:
    from langchain_core.pydantic_v1 import BaseModel
except (ImportError, ModuleNotFoundError):
    from pydantic import BaseModel

from tool_sandbox.common.execution_context import (
    TOOL_AUGMENTATION_TYPE,
    ScenarioCategories,
    get_current_context,
)
from tool_sandbox.common.utils import NotGiven


def default_parameter_processing(
    params: list[inspect.Parameter],
) -> list[inspect.Parameter]:
    """Modify the function arg spec of tool to a format more friendly for convert_to_openai_tool.
    In specific

    1. Remove Optional
    2. Remove NotGiven from Union types

    Args:
        params:   params to be processed

    Returns:
        Modified tool params
    """
    processed_params = []
    for param in params:
        parameter_type = param.annotation
        type_origin = get_origin(parameter_type)
        type_args = get_args(parameter_type)

        if type_origin != Union:
            processed_params.append(param)
            continue

        real_types = [t for t in type_args if t not in (type(None), NotGiven)]
        if len(real_types) == 1:
            param = param.replace(annotation=type_args[0])
        elif all(type in (int, float) for type in type_args):
            param = param.replace(annotation=float)

        processed_params.append(param)

    return processed_params


def maybe_scramble_arg_type(
    params: list[inspect.Parameter],
    tool_augmentation_list: list[TOOL_AUGMENTATION_TYPE],
) -> list[inspect.Parameter]:
    scramble = ScenarioCategories.ARG_TYPE_SCRAMBLED in tool_augmentation_list
    return [
        parameter.replace(annotation=type(None)) if scramble else parameter
        for parameter in params
    ]


def maybe_scramble_arg_type_from_context(
    params: list[inspect.Parameter],
) -> list[inspect.Parameter]:
    return maybe_scramble_arg_type(
        params=params,
        tool_augmentation_list=get_current_context().tool_augmentation_list,
    )


def maybe_scramble_tool_description(
    docstring: str, tool_augmentation_list: list[TOOL_AUGMENTATION_TYPE]
) -> str:
    if (
        ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED in tool_augmentation_list
        and docstring
    ):
        docstring = re.sub(r"^.*(?=Args:.*$)", "", docstring, flags=re.DOTALL)
    return docstring


def maybe_scramble_tool_description_from_context(docstring: str) -> str:
    return maybe_scramble_tool_description(
        docstring=docstring,
        tool_augmentation_list=get_current_context().tool_augmentation_list,
    )


def maybe_scramble_arg_description(
    docstring: str, tool_augmentation_list: list[TOOL_AUGMENTATION_TYPE]
) -> str:
    if ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED in tool_augmentation_list:
        if docstring:
            docstring = re.sub(
                r"(?<=Args:\n)(.*?)(?=\nReturns)", "", docstring, flags=re.DOTALL
            )
    return docstring


def maybe_scramble_arg_description_from_context(docstring: str) -> str:
    return maybe_scramble_arg_description(
        docstring=docstring,
        tool_augmentation_list=get_current_context().tool_augmentation_list,
    )


def augmented_getdoc(tool: Callable[..., Any]) -> Optional[str]:
    docstring: Optional[str] = inspect.getdoc(tool)
    if docstring is None:
        return docstring
    docstring = maybe_scramble_tool_description_from_context(docstring)
    docstring = maybe_scramble_arg_description_from_context(docstring)
    return docstring


def augmented_parameters(tool: Callable[..., Any]) -> list[inspect.Parameter]:
    params = list(inspect.signature(tool).parameters.values())
    params = default_parameter_processing(params)
    params = maybe_scramble_arg_type_from_context(params)
    return params


PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _get_python_function_name(function: Callable[..., Any]) -> str:
    return function.__name__


def _parse_python_function_docstring(
    function: Callable[..., Any],
) -> Tuple[str, dict[str, str]]:
    docstring = augmented_getdoc(function)
    if docstring:
        docstring_blocks = docstring.split("\n\n")
        descriptors = []
        args_block = None
        past_descriptors = False
        for block in docstring_blocks:
            if block.startswith("Args:"):
                args_block = block
                break
            elif block.startswith("Returns:") or block.startswith("Example:"):
                past_descriptors = True
            elif not past_descriptors:
                descriptors.append(block)
            else:
                continue
        description = " ".join(descriptors)
    else:
        description = ""
        args_block = None
    arg_descriptions = {}
    if args_block:
        arg = None
        for line in args_block.split("\n")[1:]:
            if ":" in line:
                arg, desc = line.split(":", maxsplit=1)
                arg_descriptions[arg.strip()] = desc.strip()
            elif arg:
                arg_descriptions[arg.strip()] += " " + line.strip()
    return description, arg_descriptions


def _get_python_function_arguments(
    function: Callable[..., Any], arg_descriptions: dict[str, str]
) -> dict[str, Any]:
    properties = {}
    parameters = augmented_parameters(function)
    for param in parameters:
        arg = param.name
        arg_type = param.annotation
        if arg == "return":
            continue
        if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
            properties[arg] = arg_type.schema()
        elif (
            hasattr(arg_type, "__name__")
            and getattr(arg_type, "__name__") in PYTHON_TO_JSON_TYPES
        ):
            properties[arg] = {"type": PYTHON_TO_JSON_TYPES[arg_type.__name__]}
        elif (
            hasattr(arg_type, "__dict__")
            and getattr(arg_type, "__dict__").get("__origin__", None) == Literal
        ):
            properties[arg] = {
                "enum": list(arg_type.__args__),
                "type": PYTHON_TO_JSON_TYPES[arg_type.__args__[0].__class__.__name__],
            }
        if arg in arg_descriptions:
            if arg not in properties:
                properties[arg] = {}
            properties[arg]["description"] = arg_descriptions[arg]
    return properties


def _get_python_function_required_args(function: Callable[..., Any]) -> List[str]:
    params = augmented_parameters(function)
    required = [p.name for p in params if p.default == p.empty]
    is_class = type(function) is type
    if is_class and required[0] == "self":
        required = required[1:]
    return required


def convert_python_function_to_openai_function(
    name: str,
    function: Callable[..., Any],
) -> Dict[str, Any]:
    description, arg_descriptions = _parse_python_function_docstring(function)
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": _get_python_function_arguments(function, arg_descriptions),
            "required": _get_python_function_required_args(function),
        },
    }


def convert_to_openai_tool(
    tool: Callable[..., Any],
    name: Optional[str] = None,
) -> dict[str, Any]:
    name = _get_python_function_name(tool) if name is None else name
    function = convert_python_function_to_openai_function(name, tool)
    return {"type": "function", "function": function}


def convert_to_openai_tools(
    name_to_tool: dict[str, Callable[..., Any]],
) -> list[dict[str, Any]]:
    return [convert_to_openai_tool(tool, name) for name, tool in name_to_tool.items()]
