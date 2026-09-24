from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import clinicaltrials_mcp as mcp_server


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No network and no cache files: every test starts from empty in-memory caches."""
    monkeypatch.setattr(mcp_server, 'api_cache', {})
    monkeypatch.setattr(mcp_server, 'response_cache', {})
    monkeypatch.setattr(mcp_server, 'ctis_cache', {})
    monkeypatch.setattr(mcp_server, 'trial_cache', {})
    monkeypatch.setattr(mcp_server, '_save_json', lambda path, data: None)
    monkeypatch.setattr(mcp_server.requests, 'get', MagicMock(side_effect=AssertionError('unexpected GET')))
    monkeypatch.setattr(mcp_server.requests, 'post', MagicMock(side_effect=AssertionError('unexpected POST')))


def _response(status_code, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


class TestGetAllTrials:

    def test_records_carry_status_and_registry(self, ctgov_study, ctis_raw, ctis_overview):
        ctis_parsed = [mcp_server._ctis_parse_trial(ctis_raw, ctis_overview)]
        with patch.object(mcp_server, '_fetch_trials', return_value={'studies': [ctgov_study]}), \
             patch.object(mcp_server, '_ctis_fetch_parsed', return_value=ctis_parsed):
            trials = mcp_server.get_all_trials('glioblastoma')['trials']

        by_id = {t['NCTId']: t for t in trials}
        assert by_id['NCT11111111']['OverallStatus'] == 'RECRUITING'
        assert by_id['NCT11111111']['Registry'] == 'clinicaltrials.gov'
        assert by_id['2025-500001-11-00']['OverallStatus'] == 'AUTHORISED'
        assert by_id['2025-500001-11-00']['Registry'] == 'ctis'
        assert by_id['2025-500001-11-00']['Recruiting'] is True

    def test_ctis_trials_pass_the_country_filter(self, ctgov_study, ctis_raw):
        ctis_parsed = [mcp_server._ctis_parse_trial(ctis_raw)]
        with patch.object(mcp_server, '_fetch_trials', return_value={'studies': [ctgov_study]}), \
             patch.object(mcp_server, '_ctis_fetch_parsed', return_value=ctis_parsed):
            italy = mcp_server.get_all_trials('glioblastoma', 'Italy')['trials']
            mcp_server.response_cache.clear()
            france = mcp_server.get_all_trials('glioblastoma', 'France')['trials']

        assert '2025-500001-11-00' in {t['NCTId'] for t in italy}
        assert '2025-500001-11-00' not in {t['NCTId'] for t in france}

    def test_response_cache_key_is_versioned(self):
        assert mcp_server._response_key('all_trials', 'glioblastoma', 'None') != \
            mcp_server.hashlib.md5(b'all_trials_glioblastoma_None').hexdigest()


class TestGetCurrentTrials:

    def test_ctis_trials_recruiting_in_the_country(self, ctgov_study, ctis_raw):
        ctis_parsed = [mcp_server._ctis_parse_trial(ctis_raw)]
        results = {}
        with patch.object(mcp_server, '_fetch_trials', return_value={'studies': [ctgov_study]}), \
             patch.object(mcp_server, '_ctis_fetch_parsed', return_value=ctis_parsed) as fetch:
            for country in ('Italy', 'Denmark', 'United States'):
                results[country] = {t['NCTId'] for t in mcp_server.get_current_trials('glioblastoma', country)['trials']}

        assert fetch.call_args[0][1] == 'ongoing'
        assert results['Italy'] == {'NCT11111111', '2025-500001-11-00'}
        assert results['Denmark'] == set()          # authorised there, recruitment not started
        assert results['United States'] == set()    # outside CTIS: previously got every CTIS trial

    def test_ongoing_searches_codes_that_may_be_recruiting(self):
        assert mcp_server._CTIS_STATUS_ALIASES['ongoing'] == [3, 4, 5]


class TestGetTrialNct:

    def test_found_in_disease_cache(self, ctgov_study):
        mcp_server.api_cache['k'] = {'timestamp': datetime.now().timestamp(), 'data': {'studies': [ctgov_study]}}
        trial = mcp_server.get_trial('NCT11111111')['trial']
        assert trial['NCTId'] == 'NCT11111111'
        assert trial['OverallStatus'] == 'RECRUITING'
        assert trial['Details']['protocolSection']['outcomesModule']
        mcp_server.requests.get.assert_not_called()

    def test_downloaded_when_not_cached(self, ctgov_study):
        mcp_server.requests.get = MagicMock(return_value=_response(200, ctgov_study))
        trial = mcp_server.get_trial(' nct11111111 ')['trial']
        assert trial['NCTId'] == 'NCT11111111'
        assert mcp_server.requests.get.call_args[0][0] == 'https://clinicaltrials.gov/api/v2/studies/NCT11111111'
        assert 'NCT11111111' in mcp_server.trial_cache

        mcp_server.requests.get.reset_mock()
        assert mcp_server.get_trial('NCT11111111')['trial']['NCTId'] == 'NCT11111111'
        mcp_server.requests.get.assert_not_called()

    def test_expired_disease_cache_is_ignored(self, ctgov_study):
        mcp_server.api_cache['k'] = {'timestamp': 0, 'data': {'studies': [ctgov_study]}}
        mcp_server.requests.get = MagicMock(return_value=_response(200, ctgov_study))
        mcp_server.get_trial('NCT11111111')
        mcp_server.requests.get.assert_called_once()

    def test_not_found(self):
        mcp_server.requests.get = MagicMock(return_value=_response(404))
        assert mcp_server.get_trial('NCT99999999') == {'error': 'Trial NCT99999999 not found'}

    def test_registry_error(self):
        mcp_server.requests.get = MagicMock(return_value=_response(503))
        assert 'HTTP 503' in mcp_server.get_trial('NCT99999999')['error']


class TestGetTrialEuct:

    def test_found(self, ctis_raw, ctis_overview):
        with patch.object(mcp_server, '_ctis_get_detail', return_value=ctis_raw), \
             patch.object(mcp_server, '_ctis_get_overview', return_value=ctis_overview):
            trial = mcp_server.get_trial('2025-500001-11-00')['trial']
        assert trial['NCTId'] == '2025-500001-11-00'
        assert trial['Registry'] == 'ctis'
        assert trial['OverallStatus'] == 'AUTHORISED'
        assert trial['Details']['inclusion_criteria'].startswith('1. Newly diagnosed')
        assert trial['Details']['nct_number'] == 'NCT22222222'
        assert trial['Details']['country_status'][0]['country'] == 'Italy'

    def test_works_without_overview(self, ctis_raw):
        with patch.object(mcp_server, '_ctis_get_detail', return_value=ctis_raw), \
             patch.object(mcp_server, '_ctis_get_overview', return_value=None):
            trial = mcp_server.get_trial('2025-500001-11-00')['trial']
        assert trial['BriefTitle'] == 'A study of examplinib in newly diagnosed glioblastoma'

    def test_not_found(self):
        with patch.object(mcp_server, '_ctis_get_detail', return_value=None):
            assert mcp_server.get_trial('2023-505701-14-00') == {'error': 'CTIS trial 2023-505701-14-00 not found'}

    def test_empty_retrieve_is_not_cached(self):
        mcp_server.requests.get = MagicMock(return_value=MagicMock(json=MagicMock(return_value={})))
        assert mcp_server._ctis_get_detail('2023-505701-14-00') is None
        assert mcp_server.ctis_cache == {}

    def test_overview_lookup_by_number(self, ctis_overview):
        mcp_server.requests.post = MagicMock(return_value=MagicMock(json=MagicMock(
            return_value={'data': [ctis_overview]})))
        assert mcp_server._ctis_get_overview('2025-500001-11-00') == ctis_overview
        payload = mcp_server.requests.post.call_args.kwargs['json']
        assert payload['searchCriteria']['number'] == '2025-500001-11-00'
        assert payload['searchCriteria']['containAll'] is None


def test_check_eligibility_sends_trial_criteria(ctgov_study):
    ctgov_study['protocolSection']['eligibilityModule']['eligibilityCriteria'] = (
        'Inclusion Criteria:\n* Adult\n\nExclusion Criteria:\n* Prior radiotherapy')
    answer = MagicMock(choices=[MagicMock(message=MagicMock(content='{"result": "unknown", "explanation": "x"}'))])
    with patch.object(mcp_server, '_fetch_trials', return_value={'studies': [ctgov_study]}), \
         patch.object(mcp_server.openai_client.chat.completions, 'create', return_value=answer) as mock_create:
        mcp_server.check_eligibility('NCT11111111', 'glioblastoma', 'Age: 45')

    prompt = mock_create.call_args.kwargs['messages'][1]['content']
    assert 'INCLUSION CRITERIA:\n* Adult' in prompt
    assert 'EXCLUSION CRITERIA:\n* Prior radiotherapy' in prompt


def test_get_trial_rejects_unknown_ids():
    assert 'Unrecognised trial ID' in mcp_server.get_trial('12345')['error']
