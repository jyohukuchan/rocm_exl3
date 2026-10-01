import json
import os
from pathlib import Path

import pytest

from rocm_tools.exl3_server.protocol import messages_for_template, reasoning_kwargs
from rocm_tools.exl3_server.schema import prepare_constraints


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("fixture", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_real_client_schemas_compile_and_render_with_actual_model_tokenizer(fixture):
    model_tokenizer = Path(os.environ.get("EXL3_SCHEMA_TEST_TOKENIZER", ""))
    if not model_tokenizer.is_file():
        pytest.skip("Actual model tokenizer is not available")
    from exllamav3 import Config, Tokenizer
    tokenizer = Tokenizer(Config.from_directory(str(model_tokenizer.parent)))
    body = json.loads(fixture.read_text())
    kwargs = reasoning_kwargs(body.get("chat_template_kwargs", {}) | {
        "reasoning_effort": body.get("reasoning_effort", "xhigh")})
    messages = messages_for_template(body["messages"], body.get("tools"))
    prompt = tokenizer.hf_render_chat_template(messages, tools=body.get("tools"), **kwargs)
    assert isinstance(prompt, str) and prompt
    plan = prepare_constraints(tokenizer, body.get("tools"), body.get("tool_choice", "auto"),
                               thinking=kwargs.get("enable_thinking", True))
    assert len(plan.filters) == 1
    assert plan.kind == "tools"
