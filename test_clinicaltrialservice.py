import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import tempfile
from clinicaltrialservice import app, api_cache, response_cache, get_api_cache_key, get_response_cache_key, is_cache_valid


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
                        'criteria': 'Inclusion: Age 18-65, diagnosed with cancer. Exclusion: Pregnant.',
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
                        'criteria': 'Inclusion: Age 21+',
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
        # Create data with similar facility names
        similar_facilities_data = {
            'studies': [
                {
                    'protocolSection': {
                        'identificationModule': {'nctId': 'NCT1'},
                        'contactsLocationsModule': {
                            'locations': [{'facility': 'General Hospital', 'city': 'NYC', 'country': 'United States'}]
                        },
                        'armsInterventionsModule': {'interventions': [{'name': 'Drug A'}]}
                    }
                },
                {
                    'protocolSection': {
                        'identificationModule': {'nctId': 'NCT2'},
                        'contactsLocationsModule': {
                            'locations': [{'facility': 'General Hospital', 'city': 'NYC', 'country': 'United States'}]
                        },
                        'armsInterventionsModule': {'interventions': [{'name': 'Drug B'}]}
                    }
                }
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
            assert len(data['centers']) >= 1
            # Both trials should be counted under the same facility


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

            # Check URL format
            assert trial['StudyUrl'].startswith('https://clinicaltrials.gov/study/')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
