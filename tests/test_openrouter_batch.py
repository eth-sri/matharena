"""Offline coverage of OpenRouter native Messages batching and synchronous isolation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import yaml

import matharena.api_client as api_module
from matharena.api_client import APIClient
from matharena.runner import Runner
from matharena.solvers import PureModelSolver
from matharena.solvers.harness_solver import HarnessSolver


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-openrouter-key')
    monkeypatch.setattr(api_module.request_logger, 'enabled', False)
    monkeypatch.setattr(api_module.time, 'sleep', lambda _: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Live network access is forbidden in these tests')
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)


def install_http(monkeypatch, handler):
    def request(self, method, url, **kwargs):
        prepared = requests.Request(method, url, headers=self.headers, json=kwargs.get('json')).prepare()
        outcome = handler(prepared)
        if isinstance(outcome, Exception):
            raise outcome
        code, body = outcome
        response = requests.Response()
        response.status_code = code
        response._content = json.dumps(body).encode()
        response.request = prepared
        response.url = url
        response.headers['Content-Type'] = 'application/json'
        return response
    monkeypatch.setattr(requests.sessions.Session, 'request', request)


def client(**kwargs):
    args = dict(model='anthropic/claude-opus-5.5', api='openrouter', batch_processing=True,
                max_tokens=128000, batch_max_tokens=300000, reasoning_effort='max',
                anthropic_betas=['output-300k-2026-03-24'], max_retries=2, max_retries_inner=2,
                read_cost=4, write_cost=20, cache_read_cost=.2, cache_write_cost=5,
                temperature=None, top_p=None, sleep_on_error=0,
                extra_body={'provider': {'only': ['anthropic'], 'ignore': ['anthropic/fast'],
                                         'allow_fallbacks': False}, 'cache_control': {'type': 'ephemeral'}})
    args.update(kwargs)
    return APIClient(**args)


def batch(status='completed', results=None, batch_id='batch-test', **kwargs):
    return {'id': batch_id, 'status': status, 'results': results,
            'request_counts': {'total': 2, 'completed': len(results or []), 'failed': 0}, **kwargs}


def success(idx, text='42', stop_reason='end_turn', content=None):
    return {'custom_id': f'apiquery-{idx}', 'error': None,
            'response': {'status_code': 200, 'body': {
                'id': 'msg-test', 'type': 'message', 'role': 'assistant',
                'model': 'anthropic/claude-opus-5.5', 'stop_reason': stop_reason,
                'stop_sequence': None,
                'content': content if content is not None else [
                    {'type': 'thinking', 'thinking': 'Reasoning preserved', 'signature': 'sig'},
                    {'type': 'text', 'text': text}],
                'usage': {'input_tokens': 10, 'output_tokens': 100,
                          'cache_read_input_tokens': 20, 'cache_creation_input_tokens': 30}}}}


def failure(idx):
    return {'custom_id': f'apiquery-{idx}', 'response': None,
            'error': {'code': 'server_error', 'message': 'Provider temporarily unavailable'}}


def query(text):
    return [{'role': 'developer', 'content': 'Solve carefully'}, {'role': 'user', 'content': text}]


def test_wire_payload_out_of_order_results_cache_costs_and_beta_header(monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        if request.method == 'POST':
            body = json.loads(request.body)
            assert request.url == 'https://openrouter.ai/api/v1/batches'
            assert request.headers['Authorization'] == 'Bearer test-openrouter-key'
            assert request.headers['anthropic-beta'] == 'output-300k-2026-03-24'
            assert list(body)[-1] == 'requests'
            assert body['endpoint'] == '/v1/messages'
            assert body['model'] == 'anthropic/claude-opus-5.5'
            assert body['provider'] == {'only': ['anthropic']}
            for item in body['requests']:
                params = item['body']
                assert params['max_tokens'] == 300000
                assert params['thinking'] == {'type': 'adaptive'}
                assert params['output_config'] == {'effort': 'max'}
                assert params['cache_control'] == {'type': 'ephemeral'}
                assert params['system']
                assert not {'reasoning_effort','provider','extra_body','temperature','top_p','batch_max_tokens'} & params.keys()
            return 202, batch('validating')
        return 200, batch(results=[success(71, 'second'), success(12, 'first')])
    install_http(monkeypatch, handler)
    results = list(client().run_queries([query('first'), query('second')], custom_indices=[12,71], no_tqdm=True))
    assert [r[0] for r in results] == [0,1]
    assert [r[1][-1]['content'] for r in results] == ['first','second']
    for _, conversation, usage in results:
        assert conversation[-2]['type'] == 'cot' and conversation[-2]['content'] == 'Reasoning preserved'
        assert usage['input_tokens'] == 30 and usage['cached_input_tokens'] == 20
        assert usage['cached_write_tokens'] == 30 and usage['output_tokens'] == 100
        assert usage['cost'] == pytest.approx((10*4+20*.2+30*5+100*20)/2/1e6)
    assert [c.method for c in calls] == ['POST','GET']


def test_retry_only_failed_or_missing_items_with_original_custom_ids(monkeypatch):
    submissions=[]
    def handler(request):
        body=json.loads(request.body)
        submissions.append([x['custom_id'] for x in body['requests']])
        if len(submissions)==1:
            return 202, batch(results=[success(91), failure(42)]) # 77 deliberately omitted
        return 202, batch(results=[success(77),success(42)])
    install_http(monkeypatch,handler)
    rr=list(client().run_queries([query(str(i)) for i in range(3)],custom_indices=[42,91,77],no_tqdm=True))
    assert submissions==[['apiquery-42','apiquery-91','apiquery-77'],['apiquery-42','apiquery-77']]
    assert [r[2]['n_retries'] for r in rr]==[1,0,1]
    assert all(not r[2].get('request_failed') for r in rr)


@pytest.mark.parametrize('terminal', ['failed','expired'])
def test_terminal_failure_has_bounded_retries_and_stays_pending(monkeypatch,terminal):
    calls=[]
    def handler(request):
        calls.append(request.method)
        return 202,batch(terminal,error={'message':'Provider failed'})
    install_http(monkeypatch,handler)
    [(_,_,usage)]=client().run_queries([query('a')],no_tqdm=True)
    assert calls==['POST','POST']
    assert usage['request_failed'] and usage['output_tokens']==0


def test_cancelled_batch_is_not_resubmitted(monkeypatch):
    calls=[]
    def handler(request):
        calls.append(request.method)
        return 202,batch('cancelled')
    install_http(monkeypatch,handler)
    [(_,_,usage)]=client().run_queries([query('a')],no_tqdm=True)
    assert calls==['POST'] and usage['request_failed']


def test_initial_404_and_transient_poll_errors_do_not_resubmit(monkeypatch):
    calls=[]
    responses=[(202,batch('validating')),(404,{'error':'not yet visible'}),
               (200,batch('in_progress')),(503,{'error':'unavailable'}),
               (200,batch('finalizing')),(200,batch(results=[success(0)]))]
    def handler(request):
        calls.append(request.method)
        return responses.pop(0)
    install_http(monkeypatch,handler)
    [(_,_,usage)]=client().run_queries([query('a')],no_tqdm=True)
    assert calls==['POST','GET','GET','GET','GET','GET']
    assert not usage.get('request_failed')


def test_poll_exhaustion_preserves_batch_id_and_does_not_resubmit(monkeypatch):
    calls=[]; logs=[]
    monkeypatch.setattr(api_module.request_logger,'log_response',lambda **kw:logs.append(kw))
    def handler(request):
        calls.append(request.method)
        return (202,batch('validating')) if request.method=='POST' else (503,{'error':'Unavailable'})
    install_http(monkeypatch,handler)
    [(_,_,usage)]=client().run_queries([query('a')],no_tqdm=True)
    assert calls==['POST','GET','GET'] and usage['request_failed']
    assert any(x.get('batch_id')=='batch-test' and x.get('polling_error') for x in logs)


@pytest.mark.parametrize('result', [requests.Timeout('unknown submission outcome'), (400,{'error':'invalid beta'})])
def test_ambiguous_or_permanent_submit_failure_is_not_resubmitted(monkeypatch,result):
    calls=[]
    def handler(request):
        calls.append(request.method)
        return result
    install_http(monkeypatch,handler)
    [(_,_,usage)]=client().run_queries([query('a')],no_tqdm=True)
    assert calls==['POST'] and usage['request_failed']


def test_rate_limited_submission_retries(monkeypatch):
    replies=[(429,{'error':'rate limited'}),(202,batch(results=[success(0)]))]
    install_http(monkeypatch,lambda _:replies.pop(0))
    [(_,_,usage)]=client().run_queries([query('a')],no_tqdm=True)
    assert not replies and usage['n_retries']==1


def test_capped_reasoning_only_reply_keeps_usage_and_is_not_provider_failure(monkeypatch):
    item=success(0,stop_reason='max_tokens',content=[{'type':'thinking','thinking':'Long reasoning','signature':'sig'}])
    item['response']['body']['usage']['output_tokens']=300000
    install_http(monkeypatch,lambda _:(202,batch(results=[item])))
    [(_,conversation,usage)]=client().run_queries([query('a')],no_tqdm=True)
    assert usage['output_tokens']==300000 and not usage.get('request_failed')
    assert conversation[-2]['content']=='Long reasoning' and conversation[-1]['content']==''


def test_invalid_success_body_is_retried_without_losing_other_success(monkeypatch):
    invalid=success(1); invalid['response']['body']={'type':'error','error':{'message':'provider failed'}}
    replies=[(202,batch(results=[success(0),invalid])),(202,batch(results=[success(1)]))]
    install_http(monkeypatch,lambda _ :replies.pop(0))
    rr=list(client().run_queries([query('a'),query('b')],no_tqdm=True))
    assert [r[2]['n_retries'] for r in rr]==[0,1]


def test_config_uses_300k_batches_but_128k_last_chance_and_august_harness(monkeypatch,tmp_path):
    config=yaml.safe_load(Path('configs/models/anthropic/opus_55_max.yaml').read_text())
    solver_config={'type':'pure_model','model_config':config,'scaffold_config':None}
    for comp in ['arxiv/may','arxiv/june','arxiv_false/june','arxiv/august','arxiv_false/august']:
        competition=yaml.safe_load(Path(f'configs/competitions/{comp}.yaml').read_text())
        runner=SimpleNamespace(comp_name=comp,competition_config=competition)
        assert Runner._resolve_harness(runner,solver_config)==('claude' if comp.endswith('august') else None)
    args=copy.deepcopy(config)
    for key in ['date','human_readable_id','other_params','harness','harness_version','harness_config']:
        args.pop(key,None)
    solver=PureModelSolver(solver_config,'{problem}',args,'Give the final answer')
    assert solver.client.batch_processing and solver.client.batch_max_tokens==300000
    assert not solver.last_chance_client.batch_processing
    assert solver.last_chance_client.kwargs['max_tokens']==128000
    assert not {'batch_max_tokens','anthropic_betas'} & solver.last_chance_client.kwargs.keys()
    args=copy.deepcopy(config)
    args['harness_config']['workspace_root']=str(tmp_path/'p{problem_idx}_r{run_idx}')
    real_client=APIClient
    constructed=[]
    def capture(**kwargs):
        constructed.append(kwargs)
        return real_client(**kwargs)
    monkeypatch.setattr('matharena.solvers.harness_solver.APIClient',capture)
    harness=HarnessSolver(solver_config,'{problem}',args,'')
    model=harness._build_model()
    assert constructed[-1]['batch_processing'] is False
    assert not {'batch_max_tokens','anthropic_betas'} & constructed[-1].keys()
    assert model.endpoint_for('anthropic').url=='https://openrouter.ai/api'
    assert model.reasoning=='max'
    assert model.request_overrides()==config['extra_body']
    assert args['harness_config']['environment']['CLAUDE_CODE_MAX_OUTPUT_TOKENS']=='128000'


def test_unsupported_openrouter_batch_model_fails_explicitly():
    with pytest.raises(ValueError,match='Anthropic models only'):
        client(model='x-ai/grok-4.7')


def test_malformed_acknowledgement_does_not_submit_duplicate_batch(monkeypatch):
    calls=[]
    def handler(request):
        calls.append(request.method)
        return 202, []
    install_http(monkeypatch,handler)
    [(_,_,usage)]=client().run_queries([query('a')],no_tqdm=True)
    assert calls==['POST'] and usage['request_failed']


def test_malformed_item_does_not_discard_successful_items(monkeypatch):
    invalid=success(1); invalid['response']['body']=None
    replies=[(202,batch(results=[None,success(0),invalid])),(202,batch(results=[success(1)]))]
    install_http(monkeypatch,lambda _:replies.pop(0))
    rr=list(client().run_queries([query('a'),query('b')],no_tqdm=True))
    assert [r[2]['n_retries'] for r in rr]==[0,1]
