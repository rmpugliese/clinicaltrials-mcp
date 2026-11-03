from unittest.mock import MagicMock, patch

import pytest

import clinicaltrials_mcp as mcp_server
import llm

LLM_VARS = ('LLM_BASE_URL', 'LLM_API_KEY', 'LLM_MODEL', 'LLM_REASONING_EFFORT')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start from no LLM_* settings, whatever the developer's .env says."""
    for name in LLM_VARS:
        monkeypatch.delenv(name, raising=False)


class TestSettings:

    def test_default_is_openai_o3_mini(self):
        assert llm.base_url() is None
        assert llm.request_params() == {
            'model': 'o3-mini',
            'response_format': {'type': 'json_object'},
            'reasoning_effort': 'medium',
        }

    def test_custom_endpoint(self, monkeypatch):
        monkeypatch.setenv('LLM_BASE_URL', 'https://llm.example.org/v1')
        monkeypatch.setenv('LLM_MODEL', 'mistral-small')
        assert llm.request_params() == {'model': 'mistral-small', 'response_format': {'type': 'json_object'}}

    def test_reasoning_effort_when_set(self, monkeypatch):
        monkeypatch.setenv('LLM_BASE_URL', 'https://llm.example.org/v1')
        monkeypatch.setenv('LLM_REASONING_EFFORT', 'high')
        assert llm.request_params()['reasoning_effort'] == 'high'

    def test_client_uses_base_url_and_key(self, monkeypatch):
        monkeypatch.setenv('LLM_BASE_URL', 'https://llm.example.org/v1')
        monkeypatch.setenv('LLM_API_KEY', 'dev-key')
        client = llm.make_client()
        assert str(client.base_url).startswith('https://llm.example.org/v1')
        assert client.api_key == 'dev-key'

    def test_api_key_falls_back_to_openai_key(self, monkeypatch):
        monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
        assert llm.make_client().api_key == 'openai-key'

    def test_local_server_without_key(self, monkeypatch):
        monkeypatch.setenv('LLM_BASE_URL', 'http://localhost:11434/v1')
        monkeypatch.setenv('OPENAI_API_KEY', '')
        assert llm.make_client().api_key == 'not-needed'

    def test_openai_key_never_sent_to_other_provider(self, monkeypatch):
        monkeypatch.setenv('LLM_BASE_URL', 'https://llm.example.org/v1')
        monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
        assert llm.make_client().api_key == 'not-needed'


class TestParseJsonAnswer:

    def test_plain(self):
        assert llm.parse_json_answer('{"result": "yes"}') == {'result': 'yes'}

    def test_fenced(self):
        assert llm.parse_json_answer('```json\n{"result": "no"}\n```') == {'result': 'no'}

    def test_think_block_and_surrounding_text(self):
        text = '<think>criterion 1 {is} met</think>\nEcco la valutazione: {"result": "unknown"} Fine.'
        assert llm.parse_json_answer(text) == {'result': 'unknown'}

    def test_no_json(self):
        with pytest.raises(ValueError):
            llm.parse_json_answer('Non posso valutare.')


def test_check_eligibility_uses_configured_model(monkeypatch, ctgov_study):
    monkeypatch.setenv('LLM_BASE_URL', 'https://llm.example.org/v1')
    monkeypatch.setenv('LLM_MODEL', 'mistral-small')
    answer = MagicMock(choices=[MagicMock(message=MagicMock(
        content='```json\n{"result": "unknown", "explanation": "dati mancanti"}\n```'))])
    with patch.object(mcp_server, '_fetch_trials', return_value={'studies': [ctgov_study]}), \
         patch.object(mcp_server.openai_client.chat.completions, 'create', return_value=answer) as create:
        result = mcp_server.check_eligibility('NCT11111111', 'glioblastoma', 'Age: 45')

    assert result['eligibility']['result'] == 'unknown'
    kwargs = create.call_args.kwargs
    assert kwargs['model'] == 'mistral-small'
    assert 'reasoning_effort' not in kwargs
