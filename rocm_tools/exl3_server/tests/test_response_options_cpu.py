import pytest
from test_server_http import http_state, request_json, chat_payloads
from rocm_tools.exl3_server import server


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('name,override,visible', [
    ('fake-model', None, True), ('fake-model-plain', None, False),
    ('fake-model-plain', True, True), ('fake-model', False, False)])
async def test_plain_alias_avoids_footer_for_machine_consumed_output(http_state, monkeypatch, stream, name, override, visible):
    monkeypatch.setattr(server.state, 'plain_model_name', 'fake-model-plain')
    server.state.args.include_timings = True
    body = {'model': name, 'messages': [{'role': 'user', 'content': 'title'}], 'stream': stream}
    if override is not None:
        body['include_timings'] = override
    response = await request_json('POST', '/v1/chat/completions', body)
    assert response.status_code == 200
    text, measured = chat_payloads(response, stream)
    assert ('exl3-timings' in text) is visible
    assert 'exl3_metrics' in measured
    models = await request_json('GET', '/v1/models', {})
    assert [m['id'] for m in models.json()['data']] == ['fake-model', 'fake-model-plain']
