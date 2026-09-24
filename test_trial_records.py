from trial_records import (
    REGISTRY_CTGOV,
    REGISTRY_CTIS,
    ctgov_details,
    ctis_normalize,
    ctis_parse_trial,
    ctis_status_label,
    is_euct_id,
    is_nct_id,
    simplify_ctgov,
    status_enum,
)

# Fields chatbotgbm reads today: they must stay, with the same names.
EXISTING_FIELDS = {
    'NCTId', 'BriefTitle', 'StudyUrl', 'BriefSummary', 'InterventionType', 'InterventionName',
    'CompletionDate', 'Locations', 'Phases', 'StudyType', 'EligibilityModule',
}
NEW_FIELDS = {'Registry', 'OverallStatus', 'StartDate', 'LeadSponsor', 'EnrollmentCount'}


class TestIds:

    def test_nct(self):
        assert is_nct_id('NCT04512345')
        assert is_nct_id('nct04512345')
        assert not is_nct_id('NCT0451234')
        assert not is_nct_id('2023-505701-14-00')

    def test_euct(self):
        assert is_euct_id('2023-505701-14-00')
        assert not is_euct_id('NCT04512345')


class TestCtgov:

    def test_simplify_keeps_existing_fields_and_adds_new_ones(self, ctgov_study):
        record = simplify_ctgov(ctgov_study)
        assert EXISTING_FIELDS | NEW_FIELDS <= set(record)
        assert record['Registry'] == REGISTRY_CTGOV
        assert record['OverallStatus'] == 'RECRUITING'
        assert record['StartDate'] == '2025-01-15'
        assert record['LeadSponsor'] == 'Example University'
        assert record['EnrollmentCount'] == 60
        assert record['StudyUrl'] == 'https://clinicaltrials.gov/study/NCT11111111'

    def test_simplify_tolerates_missing_modules(self):
        record = simplify_ctgov({'protocolSection': {'identificationModule': {'nctId': 'NCT1'}}})
        assert record['OverallStatus'] is None
        assert record['LeadSponsor'] is None
        assert record['Locations'] == []

    def test_details_carry_full_protocol(self, ctgov_study):
        details = ctgov_details(ctgov_study)
        assert details['protocolSection']['outcomesModule']['primaryOutcomes'][0]['measure'] == 'Overall survival'
        assert details['hasResults'] is False


class TestCtisParse:

    def test_reads_current_json_layout(self, ctis_raw, ctis_overview):
        parsed = ctis_parse_trial(ctis_raw, ctis_overview)
        assert parsed['euct_number'] == '2025-500001-11-00'
        assert parsed['title'] == 'A study of examplinib in newly diagnosed glioblastoma'
        assert parsed['official_title'].startswith('A phase I study')
        assert parsed['primary_objective'] == 'To assess the safety of examplinib.'
        assert parsed['inclusion_criteria'] == '1. Newly diagnosed glioblastoma\n2. Age 18 or older'
        assert parsed['exclusion_criteria'] == '1. Prior radiotherapy'
        assert parsed['primary_endpoints'] == ['Dose-limiting toxicities']
        assert parsed['sponsor'] == 'Example Hospital Trust'
        assert parsed['sponsor_contact']['email'] == 'trials@example.org'
        assert parsed['interventions'][0]['active_substance'] == 'EXAMPLINIB'
        assert parsed['nct_number'] == 'NCT22222222'
        assert parsed['start_date'] == '2026-03-01'
        assert parsed['estimated_end_date'] == '2028-11-01'

    def test_sites_use_iso_country_codes(self, ctis_raw):
        parsed = ctis_parse_trial(ctis_raw)
        assert parsed['n_sites'] == 1
        site = parsed['sites'][0]
        assert site['country'] == 'IT'
        assert site['name'] == 'Ospedale Esempio'
        assert site['city'] == 'Milano'
        assert site['department'] == 'Neurosurgery'

    def test_status_and_recruitment(self, ctis_raw):
        parsed = ctis_parse_trial(ctis_raw)
        assert parsed['ctis_status'] == 'Authorised'
        assert parsed['recruitment_started'] is True
        by_country = {c['country']: c for c in parsed['country_status']}
        assert by_country['Italy']['recruitment_started'] is True
        assert by_country['Denmark']['recruitment_started'] is False
        assert by_country['Italy']['recruitment_start_date'] == '2026-03-27'

    def test_overview_supplies_phase_age_and_enrollment(self, ctis_raw, ctis_overview):
        parsed = ctis_parse_trial(ctis_raw, ctis_overview)
        assert parsed['trial_phase'] == 'Human Pharmacology (Phase I)-  Other'
        assert parsed['age_groups'] == ['18-64 years', '65+ years']
        assert parsed['enrollment'] == 36

    def test_enrollment_falls_back_to_planned_subjects(self, ctis_raw):
        assert ctis_parse_trial(ctis_raw)['enrollment'] == 36

    def test_gender(self, ctis_raw):
        assert ctis_parse_trial(ctis_raw)['gender'] == 'ALL'
        population = ctis_raw['authorizedApplication']['authorizedPartI']['trialDetails'][
            'trialInformation']['populationOfTrialSubjects']
        population['isMaleSubjects'] = False
        assert ctis_parse_trial(ctis_raw)['gender'] == 'FEMALE'

    def test_sparse_trial_does_not_fail(self):
        parsed = ctis_parse_trial({'ctNumber': '2025-500002-22-00', 'ctPublicStatusCode': 8})
        assert parsed['ctis_status'] == 'Ended'
        assert 'sites' not in parsed
        assert 'recruitment_started' not in parsed

    def test_status_label_from_code(self):
        assert ctis_status_label({}, {'ctStatus': 6}) == 'Halted'
        assert ctis_status_label({'ctStatus': 'Ended'}) == 'Ended'
        assert ctis_status_label({}) == ''

    def test_status_enum(self):
        assert status_enum('Authorised') == 'AUTHORISED'
        assert status_enum('Not authorised') == 'NOT_AUTHORISED'
        assert status_enum('Under evaluation') == 'UNDER_EVALUATION'


class TestCtisNormalize:

    def test_same_fields_as_ctgov(self, ctis_raw, ctis_overview):
        record = ctis_normalize(ctis_parse_trial(ctis_raw, ctis_overview))
        assert EXISTING_FIELDS | NEW_FIELDS <= set(record)
        assert record['NCTId'] == '2025-500001-11-00'
        assert record['Registry'] == REGISTRY_CTIS
        assert record['_source'] == 'ctis'
        assert record['OverallStatus'] == 'AUTHORISED'
        assert record['RecruitmentStarted'] is True
        assert record['BriefTitle'] == 'A study of examplinib in newly diagnosed glioblastoma'
        assert record['LeadSponsor'] == 'Example Hospital Trust'
        assert record['EnrollmentCount'] == 36
        assert record['Phases'] == ['Human Pharmacology (Phase I)-  Other']
        assert record['InterventionName'] == ['Examplinib 50 mg tablets']
        assert record['Locations'] == [
            {'facility': 'Ospedale Esempio', 'city': 'Milano', 'state': None, 'country': 'Italy'},
        ]

    def test_eligibility_module(self, ctis_raw, ctis_overview):
        eligibility = ctis_normalize(ctis_parse_trial(ctis_raw, ctis_overview))['EligibilityModule']
        assert eligibility['criteria'].startswith('Inclusion Criteria:\n1. Newly diagnosed')
        assert eligibility['eligibilityCriteria'] == eligibility['criteria']
        assert 'Exclusion Criteria:\n1. Prior radiotherapy' in eligibility['criteria']
        assert eligibility['gender'] == 'ALL'
        assert eligibility['stdAges'] == ['18-64 years', '65+ years']
        assert eligibility['minimumAge'] == 'not specified'
