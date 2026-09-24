import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import tempfile
from clinicaltrialservice import app, api_cache, response_cache, get_api_cache_key, get_response_cache_key, is_cache_valid


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Keep tests off CTIS and off the cache files: no CTIS trials, no cached responses."""
    monkeypatch.setattr('clinicaltrialservice._ctis_fetch_parsed', lambda *args, **kwargs: [])
    monkeypatch.setattr('clinicaltrialservice.get_cached_response', lambda *args: None)
    monkeypatch.setattr('clinicaltrialservice.set_response_cache_response', lambda *args: None)


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_api_key():
    """Return a valid API key from environment."""
    return os.getenv('ALLOWED_API_KEYS', '').split(',')[0]


@pytest.fixture
def mock_trial_data():
    """Mock trial data structure from ClinicalTrials.gov."""
    return {
        'studies': [
            {
                'protocolSection': {
                    'identificationModule': {
                        'nctId': 'NCT12345678',
                        'briefTitle': 'Test Trial for Cancer Treatment'
                    },
                    'statusModule': {
                        'overallStatus': 'RECRUITING',
                        'completionDateStruct': {'date': '2025-12-31'}
                    },
                    'descriptionModule': {
                        'briefSummary': 'This is a test trial for cancer treatment.'
                    },
                    'designModule': {
                        'phases': ['PHASE2', 'PHASE3'],
                        'studyType': 'INTERVENTIONAL'
                    },
                    'armsInterventionsModule': {
                        'interventions': [
                            {'type': 'DRUG', 'name': 'Test Drug A'},
                            {'type': 'DRUG', 'name': 'Test Drug B'}
                        ]
                    },
                    'contactsLocationsModule': {
                        'locations': [
                            {
                                'facility': 'Test Hospital',
                                'city': 'New York',
                                'state': 'NY',
                                'country': 'United States'
                            },
                            {
                                'facility': 'Test Clinic',
                                'city': 'Toronto',
                                'state': 'ON',
                                'country': 'Canada'
                            }
                        ]
                    },
                    'eligibilityModule': {
                        'eligibilityCriteria': 'Inclusion Criteria:\n* Age 18-65, diagnosed with cancer\n\nExclusion Criteria:\n* Pregnant',
                        'minimumAge': '18 Years',
                        'maximumAge': '65 Years',
                        'gender': 'ALL'
                    }
                }
            },
            {
                'protocolSection': {
                    'identificationModule': {
                        'nctId': 'NCT87654321',
                        'briefTitle': 'Another Test Trial'
                    },
                    'statusModule': {
                        'overallStatus': 'COMPLETED',
                        'completionDateStruct': {'date': '2024-06-30'}
                    },
                    'descriptionModule': {
                        'briefSummary': 'Another test trial summary.'
                    },
                    'designModule': {
                        'phases': ['PHASE1'],
                        'studyType': 'INTERVENTIONAL'
                    },
                    'armsInterventionsModule': {
                        'interventions': [
                            {'type': 'BIOLOGICAL', 'name': 'Test Biologic'}
                        ]
                    },
                    'contactsLocationsModule': {
                        'locations': [
                            {
                                'facility': 'Research Center',
                                'city': 'Boston',
                                'state': 'MA',
                                'country': 'United States'
                            }
                        ]
                    },
                    'eligibilityModule': {
                        'eligibilityCriteria': 'Inclusion Criteria:\n* Age 21+',
                        'minimumAge': '21 Years',
                        'maximumAge': 'N/A',
                        'gender': 'ALL'
                    }
                }
            }
        ]
    }


class TestAuthentication:
    """Test API authentication."""

    def test_missing_api_key(self, client):
        """Test that requests without API key are rejected."""
        response = client.get('/current_trials?disease=cancer&country=United States')
        assert response.status_code == 401
        assert response.json['error'] == 'Unauthorized'

    def test_invalid_api_key(self, client):
        """Test that requests with invalid API key are rejected."""
        response = client.get(
            '/current_trials?disease=cancer&country=United States',
            headers={'x-api-key': 'invalid_key_12345'}
        )
        assert response.status_code == 401
        assert response.json['error'] == 'Unauthorized'

    def test_valid_api_key(self, client, valid_api_key, mock_trial_data):
        """Test that requests with valid API key are accepted."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data):
            response = client.get(
                '/current_trials?disease=cancer&country=United States',
                headers={'x-api-key': valid_api_key}
            )
            assert response.status_code == 200


class TestCurrentTrialsEndpoint:
    """Test /current_trials endpoint."""

    def test_current_trials_success(self, client, valid_api_key, mock_trial_data):
        """Test successful retrieval of current trials."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/current_trials?disease=cancer&country=United States',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'trials' in data
            assert len(data['trials']) == 1  # Only RECRUITING trial in United States
            assert data['trials'][0]['NCTId'] == 'NCT12345678'
            assert data['trials'][0]['BriefTitle'] == 'Test Trial for Cancer Treatment'

    def test_current_trials_no_matching_country(self, client, valid_api_key, mock_trial_data):
        """Test current trials with no matching country."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/current_trials?disease=cancer&country=Germany',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'trials' in data
            assert len(data['trials']) == 0

    def test_current_trials_no_data(self, client, valid_api_key):
        """Test current trials when no data is found."""
        with patch('clinicaltrialservice.get_trial_data', return_value=None):
            response = client.get(
                '/current_trials?disease=rare_disease&country=United States',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 404
            assert response.json['error'] == 'No data found'

    def test_current_trials_cached_response(self, client, valid_api_key):
        """Test that cached responses are returned."""
        cached_data = [{'NCTId': 'NCT99999999', 'BriefTitle': 'Cached Trial'}]

        with patch('clinicaltrialservice.get_cached_response', return_value=cached_data):
            response = client.get(
                '/current_trials?disease=cancer&country=United States',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert data['trials'] == cached_data


class TestAllTrialsEndpoint:
    """Test /all_trials endpoint."""

    def test_all_trials_success(self, client, valid_api_key, mock_trial_data):
        """Test successful retrieval of all trials."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/all_trials?disease=cancer',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'trials' in data
            assert len(data['trials']) == 2  # Both trials returned

    def test_all_trials_with_country_filter(self, client, valid_api_key, mock_trial_data):
        """Test all trials with country filter."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/all_trials?disease=cancer&country=Canada',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'trials' in data
            assert len(data['trials']) == 1  # Only one trial has Canada location

    def test_all_trials_missing_disease(self, client, valid_api_key):
        """Test all trials without disease parameter."""
        response = client.get(
            '/all_trials',
            headers={'x-api-key': valid_api_key}
        )

        assert response.status_code == 400
        assert response.json['error'] == 'Disease parameter is required'


class TestSpecializedCentersEndpoint:
    """Test /specialized_centers endpoint."""

    def test_specialized_centers_success(self, client, valid_api_key, mock_trial_data):
        """Test successful retrieval of specialized centers."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/specialized_centers?disease=cancer&country=United States',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'centers' in data
            # Centers with count >= 2 are returned

    def test_specialized_centers_fuzzy_matching(self, client, valid_api_key):
        """Test that fuzzy matching works for similar facility names."""
        # Five trials at the same hospital, spelled in slightly different ways
        # (only centres with more than 4 trials are returned)
        names = ['General Hospital', 'General Hospital', 'General Hospital NYC',
                 'The General Hospital', 'General Hospital']
        similar_facilities_data = {
            'studies': [
                {
                    'protocolSection': {
                        'identificationModule': {'nctId': f'NCT{i}'},
                        'contactsLocationsModule': {
                            'locations': [{'facility': name, 'city': 'NYC', 'country': 'United States'}]
                        },
                        'armsInterventionsModule': {'interventions': [{'name': f'Drug {i}'}]}
                    }
                }
                for i, name in enumerate(names)
            ]
        }

        with patch('clinicaltrialservice.get_trial_data', return_value=similar_facilities_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/specialized_centers?disease=cancer&country=United States',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'centers' in data
            assert len(data['centers']) == 1
            assert data['centers'][0]['facility'] == 'General Hospital'
            assert data['centers'][0]['trialCount'] == 5


class TestAvailableTreatmentsEndpoint:
    """Test /available_treatments endpoint."""

    def test_available_treatments_success(self, client, valid_api_key, mock_trial_data):
        """Test successful retrieval of available treatments."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/available_treatments?disease=cancer',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'treatments' in data

    def test_available_treatments_interventional_only(self, client, valid_api_key):
        """Test that only interventional studies are included."""
        observational_data = {
            'studies': [
                {
                    'protocolSection': {
                        'designModule': {'studyType': 'OBSERVATIONAL'},
                        'armsInterventionsModule': {'interventions': [{'name': 'Observation', 'type': 'OTHER'}]}
                    }
                }
            ]
        }

        with patch('clinicaltrialservice.get_trial_data', return_value=observational_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/available_treatments?disease=cancer',
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'treatments' in data
            assert len(data['treatments']) == 0  # No interventional studies


class TestCheckEligibilityEndpoint:
    """Test /check_eligibility endpoint."""

    def test_check_eligibility_success(self, client, valid_api_key, mock_trial_data):
        """Test successful eligibility check."""
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [
            MagicMock(message=MagicMock(content='{"result": "yes", "explanation": "Patient meets all criteria."}'))
        ]

        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.client.chat.completions.create', return_value=mock_openai_response):

            payload = {
                'nctId': 'NCT12345678',
                'disease': 'cancer',
                'patient_info': 'Age: 45, Gender: Male, Diagnosed with cancer'
            }

            response = client.post(
                '/check_eligibility',
                json=payload,
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 200
            data = response.json
            assert 'nctId' in data
            assert 'eligibility' in data
            assert data['eligibility']['result'] == 'yes'
            assert 'explanation' in data['eligibility']

    def test_check_eligibility_sends_trial_criteria(self, client, valid_api_key, mock_trial_data):
        """The prompt carries the trial's inclusion and exclusion criteria."""
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [
            MagicMock(message=MagicMock(content='{"result": "unknown", "explanation": "Missing data."}'))
        ]

        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.client.chat.completions.create', return_value=mock_openai_response) as mock_create:
            client.post(
                '/check_eligibility',
                json={'nctId': 'NCT12345678', 'disease': 'cancer', 'patient_info': 'Age: 45'},
                headers={'x-api-key': valid_api_key}
            )

        prompt = mock_create.call_args.kwargs['messages'][1]['content']
        assert 'INCLUSION CRITERIA:\n* Age 18-65, diagnosed with cancer' in prompt
        assert 'EXCLUSION CRITERIA:\n* Pregnant' in prompt
        assert 'No criteria provided' not in prompt

    def test_check_eligibility_missing_parameters(self, client, valid_api_key):
        """Test eligibility check with missing parameters."""
        payload = {
            'nctId': 'NCT12345678'
            # Missing disease and patient_info
        }

        response = client.post(
            '/check_eligibility',
            json=payload,
            headers={'x-api-key': valid_api_key}
        )

        assert response.status_code == 400
        assert 'required parameters' in response.json['error']

    def test_check_eligibility_trial_not_found(self, client, valid_api_key, mock_trial_data):
        """Test eligibility check for non-existent trial."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data):
            payload = {
                'nctId': 'NCT99999999',  # Non-existent trial
                'disease': 'cancer',
                'patient_info': 'Age: 45'
            }

            response = client.post(
                '/check_eligibility',
                json=payload,
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 404
            assert 'not found' in response.json['error']

    def test_check_eligibility_openai_error(self, client, valid_api_key, mock_trial_data):
        """Test eligibility check when OpenAI API fails."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.client.chat.completions.create', side_effect=Exception('API Error')):

            payload = {
                'nctId': 'NCT12345678',
                'disease': 'cancer',
                'patient_info': 'Age: 45'
            }

            response = client.post(
                '/check_eligibility',
                json=payload,
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 500
            assert 'OpenAI API' in response.json['error']

    def test_check_eligibility_invalid_json_response(self, client, valid_api_key, mock_trial_data):
        """Test eligibility check with invalid JSON from OpenAI."""
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [
            MagicMock(message=MagicMock(content='This is not valid JSON'))
        ]

        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.client.chat.completions.create', return_value=mock_openai_response):

            payload = {
                'nctId': 'NCT12345678',
                'disease': 'cancer',
                'patient_info': 'Age: 45'
            }

            response = client.post(
                '/check_eligibility',
                json=payload,
                headers={'x-api-key': valid_api_key}
            )

            assert response.status_code == 500
            assert 'Failed to parse' in response.json['error']


class TestCachingFunctions:
    """Test caching functionality."""

    def test_get_api_cache_key(self):
        """Test API cache key generation."""
        key1 = get_api_cache_key('cancer')
        key2 = get_api_cache_key('cancer')
        key3 = get_api_cache_key('diabetes')

        assert key1 == key2  # Same disease should generate same key
        assert key1 != key3  # Different diseases should generate different keys

    def test_get_response_cache_key(self):
        """Test response cache key generation."""
        key1 = get_response_cache_key('current_trials', 'cancer', 'United States')
        key2 = get_response_cache_key('current_trials', 'cancer', 'United States')
        key3 = get_response_cache_key('current_trials', 'cancer', 'Canada')

        assert key1 == key2
        assert key1 != key3

    def test_is_cache_valid_expired(self):
        """Test that expired cache is detected."""
        old_entry = {
            'timestamp': datetime.now().timestamp() - 90000,  # More than 24 hours ago
            'data': {}
        }

        assert not is_cache_valid(old_entry)

    def test_is_cache_valid_fresh(self):
        """Test that fresh cache is detected."""
        fresh_entry = {
            'timestamp': datetime.now().timestamp(),
            'data': {}
        }

        assert is_cache_valid(fresh_entry)


class TestExternalAPIIntegration:
    """Test interaction with external ClinicalTrials.gov API."""

    @patch('clinicaltrialservice.requests.get')
    def test_get_trial_data_api_call(self, mock_get, mock_trial_data):
        """Test that get_trial_data makes correct API calls."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_trial_data
        mock_get.return_value = mock_response

        # Clear cache to force API call
        with patch('clinicaltrialservice.get_cached_api_data', return_value=None), \
             patch('clinicaltrialservice.set_api_cache_data'):

            from clinicaltrialservice import get_trial_data
            result = get_trial_data('cancer')

            assert result is not None
            assert 'studies' in result
            mock_get.assert_called_once()

    @patch('clinicaltrialservice.requests.get')
    def test_get_trial_data_api_error(self, mock_get):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with patch('clinicaltrialservice.get_cached_api_data', return_value=None):
            from clinicaltrialservice import get_trial_data
            result = get_trial_data('cancer')

            assert result is None

    @patch('clinicaltrialservice.requests.get')
    def test_get_trial_data_pagination(self, mock_get, mock_trial_data):
        """Test that pagination is handled correctly."""
        # First page with next token
        first_response = MagicMock()
        first_response.status_code = 200
        first_response.json.return_value = {
            'studies': [mock_trial_data['studies'][0]],
            'nextPageToken': 'token123'
        }

        # Second page without next token
        second_response = MagicMock()
        second_response.status_code = 200
        second_response.json.return_value = {
            'studies': [mock_trial_data['studies'][1]],
            'nextPageToken': None
        }

        mock_get.side_effect = [first_response, second_response]

        with patch('clinicaltrialservice.get_cached_api_data', return_value=None), \
             patch('clinicaltrialservice.set_api_cache_data'):

            from clinicaltrialservice import get_trial_data
            result = get_trial_data('cancer')

            assert result is not None
            assert len(result['studies']) == 2
            assert mock_get.call_count == 2


class TestDataExtraction:
    """Test data extraction from complex nested structures."""

    def test_extract_trial_fields(self, client, valid_api_key, mock_trial_data):
        """Test that all required fields are extracted correctly."""
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice.get_cached_response', return_value=None), \
             patch('clinicaltrialservice.set_response_cache_response'):

            response = client.get(
                '/all_trials?disease=cancer',
                headers={'x-api-key': valid_api_key}
            )

            trial = response.json['trials'][0]

            # Check all expected fields
            assert 'NCTId' in trial
            assert 'BriefTitle' in trial
            assert 'StudyUrl' in trial
            assert 'BriefSummary' in trial
            assert 'InterventionType' in trial
            assert 'InterventionName' in trial
            assert 'CompletionDate' in trial
            assert 'Locations' in trial
            assert 'Phases' in trial
            assert 'StudyType' in trial
            assert 'EligibilityModule' in trial

            # Fields added in v2
            assert trial['OverallStatus'] == 'RECRUITING'
            assert trial['Registry'] == 'clinicaltrials.gov'
            assert 'StartDate' in trial
            assert 'LeadSponsor' in trial
            assert 'EnrollmentCount' in trial

            # Check URL format
            assert trial['StudyUrl'].startswith('https://clinicaltrials.gov/study/')

    def test_ctis_trials_in_all_trials(self, client, valid_api_key, mock_trial_data, ctis_raw, ctis_overview):
        """CTIS trials carry their registry, status and sites."""
        from clinicaltrialservice import _ctis_parse_trial
        parsed = [_ctis_parse_trial(ctis_raw, ctis_overview)]
        with patch('clinicaltrialservice.get_trial_data', return_value=mock_trial_data), \
             patch('clinicaltrialservice._ctis_fetch_parsed', return_value=parsed):
            response = client.get('/all_trials?disease=cancer&country=Italy', headers={'x-api-key': valid_api_key})

        trials = response.json['trials']
        assert [t['NCTId'] for t in trials] == ['2025-500001-11-00']
        assert trials[0]['Registry'] == 'ctis'
        assert trials[0]['OverallStatus'] == 'AUTHORISED'


class TestTrialEndpoint:
    """Test /trial/<trial_id> endpoint."""

    def test_requires_api_key(self, client):
        assert client.get('/trial/NCT12345678').status_code == 401

    def test_nct_from_disease_cache(self, client, valid_api_key, mock_trial_data):
        cache = {'k': {'timestamp': datetime.now().timestamp(), 'data': mock_trial_data}}
        with patch.dict('clinicaltrialservice.api_cache', cache, clear=True), \
             patch('clinicaltrialservice.requests.get') as mock_get:
            response = client.get('/trial/NCT87654321', headers={'x-api-key': valid_api_key})

        assert response.status_code == 200
        trial = response.json['trial']
        assert trial['NCTId'] == 'NCT87654321'
        assert trial['OverallStatus'] == 'COMPLETED'
        assert trial['Details']['protocolSection']['identificationModule']['nctId'] == 'NCT87654321'
        mock_get.assert_not_called()

    def test_nct_downloaded(self, client, valid_api_key, mock_trial_data):
        study = mock_trial_data['studies'][0]
        with patch.dict('clinicaltrialservice.api_cache', {}, clear=True), \
             patch.dict('clinicaltrialservice.trial_cache', {}, clear=True), \
             patch('clinicaltrialservice.open', mock_open(), create=True), \
             patch('clinicaltrialservice.requests.get', return_value=MagicMock(status_code=200, json=lambda: study)) as mock_get:
            response = client.get('/trial/nct12345678', headers={'x-api-key': valid_api_key})

        assert response.status_code == 200
        assert response.json['trial']['NCTId'] == 'NCT12345678'
        assert mock_get.call_args[0][0] == 'https://clinicaltrials.gov/api/v2/studies/NCT12345678'

    def test_nct_not_found(self, client, valid_api_key):
        with patch.dict('clinicaltrialservice.api_cache', {}, clear=True), \
             patch.dict('clinicaltrialservice.trial_cache', {}, clear=True), \
             patch('clinicaltrialservice.requests.get', return_value=MagicMock(status_code=404)):
            response = client.get('/trial/NCT99999999', headers={'x-api-key': valid_api_key})

        assert response.status_code == 404
        assert 'not found' in response.json['error']

    def test_euct(self, client, valid_api_key, ctis_raw, ctis_overview):
        with patch('clinicaltrialservice._ctis_get_detail', return_value=ctis_raw), \
             patch('clinicaltrialservice._ctis_get_overview', return_value=ctis_overview):
            response = client.get('/trial/2025-500001-11-00', headers={'x-api-key': valid_api_key})

        assert response.status_code == 200
        trial = response.json['trial']
        assert trial['Registry'] == 'ctis'
        assert trial['Details']['exclusion_criteria'] == '1. Prior radiotherapy'

    def test_euct_not_found(self, client, valid_api_key):
        with patch('clinicaltrialservice._ctis_get_detail', return_value=None):
            response = client.get('/trial/2023-505701-14-00', headers={'x-api-key': valid_api_key})
        assert response.status_code == 404

    def test_unrecognised_id(self, client, valid_api_key):
        response = client.get('/trial/abc', headers={'x-api-key': valid_api_key})
        assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
