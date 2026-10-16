import base64
import json

import httpx
import pytest
from google import genai

import matharena.api_client as api_module
from matharena.api_client import APIClient
from matharena.request_logger import RequestLogger


@pytest.mark.parametrize('resume_from_checkpoint', [False, True])
@pytest.mark.parametrize('empty_hop', [False, True])
@pytest.mark.parametrize('final_text,finish_reason', [(r'\boxed{13}', 'STOP'), ('', 'MAX_TOKENS')])
def test_native_google_continues_empty_thinking_hop_and_counts_usage(
    monkeypatch, tmp_path, final_text, finish_reason, empty_hop, resume_from_checkpoint
):
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-google-key')
    monkeypatch.setattr(api_module.time, 'sleep', lambda _: None)
    log = RequestLogger()
    log.log_dir = str(tmp_path)
    monkeypatch.setattr(api_module, 'request_logger', log)
    token = base64.b64encode(b'opaque-model-state').decode()
    requests = []
    responses = [
        {
            'candidates': [{'content': {'role': 'model', 'parts': []},
                            'finishReason': 'CONTINUATION', 'continuationToken': token}],
            'usageMetadata': {'promptTokenCount': 100, 'cachedContentTokenCount': 20,
                              'candidatesTokenCount': 0, 'thoughtsTokenCount': 32768,
                              'totalTokenCount': 32868},
        },
        {
            'candidates': [{'content': {'role': 'model', 'parts': [
                {'thought': True, 'text': 'Checked the calculation.'}, {'text': final_text},
            ]}, 'finishReason': finish_reason}],
            'usageMetadata': {'promptTokenCount': 33000, 'cachedContentTokenCount': 32000,
                              'candidatesTokenCount': 7, 'thoughtsTokenCount': 90,
                              'totalTokenCount': 33097},
        },
    ]

    if empty_hop:
        responses[1:1] = [
            {'candidates': [{'content': {'role': 'model', 'parts': []},
                             'finishReason': 'CONTINUATION', 'continuationToken': token}],
             'usageMetadata': {'promptTokenCount': 33000, 'cachedContentTokenCount': 32000,
                               'candidatesTokenCount': 0, 'thoughtsTokenCount': 32768,
                               'totalTokenCount': 65768}},
            {},  # SDK 2.30.1 returns the previous CONTINUATION merge after an empty hop.
        ]

    def handler(request):
        assert request.url.path == '/v1beta/models/gemini-4-argon:generateContent'
        assert request.headers['x-goog-api-key'] == 'test-google-key'
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(requests) - 1])

    real_client = genai.Client

    def factory(**kwargs):
        assert kwargs['api_key'] == 'test-google-key'
        assert kwargs['http_options']['timeout'] == 1_200_000
        kwargs['http_options']['httpx_client'] = httpx.Client(transport=httpx.MockTransport(handler))
        return real_client(**kwargs)

    monkeypatch.setattr(genai, 'Client', factory)
    client = APIClient(model='gemini-4-argon', api='google', use_google_genai=True,
                       max_tokens=524288, reasoning_effort='high', timeout=1200,
                       continuation_token=b'checkpoint' if resume_from_checkpoint else None,
                       read_cost=2, cache_read_cost=.1, write_cost=10,
                       max_retries=1, throw_error_on_failure=True)
    query = [{'role': 'system', 'content': 'Report a boxed answer.'},
             {'role': 'user', 'content': 'What is 6+7?'},
             {'role': 'assistant', 'content': 'A draft answer.'},
             {'role': 'user', 'content': 'Check and report the final answer.'}]
    [(index, conversation, cost)] = client.run_queries([query], no_tqdm=True)
    assert index == 0
    assert len(requests) == (4 if empty_hop else 2)
    assert requests[1]['contents'] == requests[0]['contents']
    assert [item['role'] for item in requests[0]['contents']] == ['user', 'model', 'user']
    assert requests[0]['systemInstruction']['parts'] == [{'text': 'Report a boxed answer.'}]
    assert requests[0].get('continuationToken') == (base64.b64encode(b'checkpoint').decode()
                                                        if resume_from_checkpoint else None)
    assert requests[1]['continuationToken'] == token
    for request in requests:
        assert request['generationConfig']['maxOutputTokens'] == 524288
        thinking = request['generationConfig']['thinkingConfig']
        assert thinking.get('thinkingLevel', thinking.get('thinking_level')).lower() == 'high'
        assert thinking.get('includeThoughts', thinking.get('include_thoughts')) is True
    assert conversation[-1] == {'role': 'assistant', 'type': 'response', 'content': final_text}
    assert conversation[-2]['type'] == 'cot'
    assert cost['input_tokens'] == 33100 + (33000 if empty_hop else 0)
    assert cost['output_tokens'] == 32865 + (32768 if empty_hop else 0)
    assert cost['cached_input_tokens'] == 32020 + (32000 if empty_hop else 0)
    assert cost['cost'] == pytest.approx(((cost['input_tokens'] - cost['cached_input_tokens']) * 2
                                         + cost['cached_input_tokens'] * .1 + cost['output_tokens'] * 10) / 1e6)
    recorded = json.loads(max(tmp_path.rglob('*.json'), key=lambda p: p.stat().st_mtime).read_text())
    assert recorded['response']['candidates'][0]['finish_reason'] == finish_reason
    assert 'test-google-key' not in json.dumps(recorded)


def test_existing_google_configs_keep_openai_transport(monkeypatch):
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-google-key')
    client = APIClient(model='gemini-3.8-flash', api='google', reasoning_effort='high')
    assert client.api == 'openai'
    assert client.base_url == 'https://generativelanguage.googleapis.com/v1beta/openai/'
