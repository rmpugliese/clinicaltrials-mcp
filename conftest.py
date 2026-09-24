import os

import pytest

# Both servers build an OpenAI client at import time; the tests never call it.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


@pytest.fixture
def ctgov_study():
    """A raw ClinicalTrials.gov study, as returned by /api/v2/studies."""
    return {
        'protocolSection': {
            'identificationModule': {'nctId': 'NCT11111111', 'briefTitle': 'Drug X in Glioblastoma'},
            'statusModule': {
                'overallStatus': 'RECRUITING',
                'startDateStruct': {'date': '2025-01-15'},
                'completionDateStruct': {'date': '2028-12'},
            },
            'sponsorCollaboratorsModule': {'leadSponsor': {'name': 'Example University'}},
            'descriptionModule': {'briefSummary': 'Phase 2 study of Drug X.'},
            'designModule': {
                'phases': ['PHASE2'],
                'studyType': 'INTERVENTIONAL',
                'enrollmentInfo': {'count': 60, 'type': 'ESTIMATED'},
            },
            'armsInterventionsModule': {'interventions': [{'type': 'DRUG', 'name': 'Drug X'}]},
            'contactsLocationsModule': {'locations': [
                {'facility': 'Ospedale Esempio', 'city': 'Milano', 'country': 'Italy', 'status': 'RECRUITING'},
            ]},
            'eligibilityModule': {'eligibilityCriteria': 'Inclusion Criteria:\n* Adult', 'minimumAge': '18 Years'},
            'outcomesModule': {'primaryOutcomes': [{'measure': 'Overall survival'}]},
        },
        'hasResults': False,
    }


@pytest.fixture
def ctis_raw():
    """A raw CTIS trial from the retrieve endpoint, trimmed to the fields the parser reads."""
    return {
        'ctNumber': '2025-500001-11-00',
        'ctStatus': 'Authorised',
        'ctPublicStatusCode': 4,
        'startDateEU': '2026-03-01',
        'decisionDate': '2025-10-14T07:34:07.002',
        'authorizedApplication': {
            'authorizedPartI': {
                'rowSubjectCount': 0,
                'medicalConditions': [{'medicalCondition': 'Glioblastoma'}],
                'products': [{
                    'productDictionaryInfo': {'prodName': 'Examplinib 50 mg tablets',
                                              'activeSubstanceName': 'EXAMPLINIB'},
                    'pharmaceuticalFormDisplay': 'TABLET',
                    'routes': ['ORAL USE'],
                }],
                'sponsors': [{
                    'primary': True,
                    'organisation': {'name': 'Example Hospital Trust'},
                    'publicContacts': [{'functionalName': 'Trial Office',
                                        'functionalEmailAddress': 'trials@example.org',
                                        'telephone': '+390000000'}],
                }],
                'trialDetails': {
                    'clinicalTrialIdentifiers': {
                        'publicTitle': 'A study of examplinib in newly diagnosed glioblastoma',
                        'fullTitle': 'A phase I study of examplinib added to standard therapy',
                        'shortTitle': 'EXA-1',
                        'secondaryIdentifyingNumbers': {'nctNumber': {'number': 'NCT22222222'}},
                    },
                    'trialInformation': {
                        'trialObjective': {
                            'mainObjective': 'To assess the safety of examplinib.',
                            'secondaryObjectives': [{'secondaryObjective': 'To assess survival.'}],
                        },
                        'eligibilityCriteria': {
                            'principalInclusionCriteria': [
                                {'principalInclusionCriteria': '1. Newly diagnosed glioblastoma'},
                                {'principalInclusionCriteria': '2. Age 18 or older'},
                            ],
                            'principalExclusionCriteria': [
                                {'principalExclusionCriteria': '1. Prior radiotherapy'},
                            ],
                        },
                        'endPoint': {
                            'primaryEndPoints': [{'endPoint': 'Dose-limiting toxicities'}],
                            'secondaryEndPoints': [{'endPoint': 'Overall survival'}],
                        },
                        'trialDuration': {'estimatedEndDate': '2028-11-01',
                                          'estimatedRecruitmentStartDate': '2026-01-01'},
                        'populationOfTrialSubjects': {'isFemaleSubjects': True, 'isMaleSubjects': True},
                    },
                },
            },
            'authorizedPartsII': [
                {
                    'recruitmentSubjectCount': 20,
                    'mscInfo': {'countryName': 'Italy', 'trialStatus': 'Authorised',
                                'hasRecruitmentStarted': True,
                                'activeTrialRecruitmentPeriod': {'recruitmentStartDate': '2026-03-27'}},
                    'trialSites': [{
                        'organisationAddressInfo': {
                            'organisation': {'name': 'Ospedale Esempio'},
                            'address': {'city': 'Milano', 'countryName': 'Italy'},
                        },
                        'departmentName': 'Neurosurgery',
                        'personInfo': {'firstName': 'Mario', 'lastName': 'Rossi'},
                    }],
                },
                {
                    'recruitmentSubjectCount': 16,
                    'mscInfo': {'countryName': 'Denmark', 'trialStatus': 'Authorised',
                                'hasRecruitmentStarted': False},
                    'trialSites': [],
                },
            ],
        },
    }


@pytest.fixture
def ctis_overview():
    """The same trial's row from the CTIS search endpoint."""
    return {
        'ctNumber': '2025-500001-11-00',
        'ctStatus': 4,
        'ctTitle': 'A study of examplinib in newly diagnosed glioblastoma',
        'sponsor': 'Example Hospital Trust',
        'sponsorType': 'Hospital/Clinic/Other health care facility',
        'trialPhase': 'Human Pharmacology (Phase I)-  Other',
        'ageGroup': '18-64 years, 65+ years',
        'gender': 'Female, Male',
        'totalNumberEnrolled': '36',
    }
