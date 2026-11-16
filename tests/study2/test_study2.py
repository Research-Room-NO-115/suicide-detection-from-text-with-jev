"""Explicit Study 2 suite; the historical notebook's 101-test suite is preserved.

All data and transports in this module are synthetic. No live model is called.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from jev_research.additional_experiment import manifest_hash
from jev_research.budgeted_experiments import digest
from jev_research.client import BASE_MODEL,QUESTION,CohortMismatch,JevClientError,BudgetExhausted,text_sha256
from jev_research.study2 import (STUDY,STRATA,build_sample,policy_arrays,analyze_probability_sample,
    contributions,design_totals,sensitivity_decision,fit_local_text_baseline,
    load_historical,historical_checks,repeat_audit)
from jev_research.study2_collection import (freeze_for_collection,Study2JevClient,
    collect,ledger_status,verify_post_mapping,write_frozen_package)

ROOT=Path(__file__).resolve().parents[2]


def fixture_frame():
    rows=[];rid=0
    for low,high,n in [(.02,.24,900),(.26,.299,30),(.31,.79,50),(.81,.99,1020)]:
        for i,p in enumerate(np.linspace(low,high,n)):
            text=f'Synthetic study2 example {rid}'
            rows.append({'row_id':rid,'sample_id':text_sha256('synthetic:'+str(rid)),
                         'text':text,'text_sha256':text_sha256(text),'p_xgb':float(p),
                         'dataset_sha256':'a'*64,'baseline_fingerprint':'b'*64,
                         'source_split':'test','y_true':i%2})
            rid+=1
    return pd.DataFrame(rows)


def candidate():
    return {'xgb_threshold':.30,'jev_threshold':.03,'routing_interval':[.25,.8],
            'model':BASE_MODEL,'question_sha256':digest(QUESTION)}


def draft():
    return {'status':'DRAFT','study_id':STUDY,'candidate_policy':candidate(),'gates':{}}


class Response:
    status_code=200
    def json(self):return {'answers':{'department':{'score':.8}}}


class Transport:
    def __init__(self,failure=False):self.calls=0;self.failure=failure
    def post(self,*args,**kwargs):
        self.calls+=1
        assert kwargs['allow_redirects'] is False
        if self.failure:raise requests.Timeout('synthetic timeout')
        return Response()


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.full=fixture_frame()
        self.manifest,self.design=build_sample(self.full,{},candidate(),requested_n=400)
        truth=self.full.set_index('text_sha256').loc[self.manifest.text_sha256,'y_true'].to_numpy()
        self.sample=self.manifest.assign(y_true=truth,jev_score=np.where(truth==1,.8,.01),jev_status='succeeded')

    def test_sampling_label_blind_disjoint_strata_and_exact_quotas(self):
        other=self.full.assign(y_true=1-self.full.y_true).sample(frac=1,random_state=2)
        changed,design=build_sample(other,{},candidate(),requested_n=400)
        pd.testing.assert_frame_equal(changed,self.manifest)
        self.assertNotIn('y_true',self.manifest)
        self.assertEqual(set(self.manifest.stratum),set(STRATA))
        self.assertEqual(len(self.manifest),400)
        self.assertEqual(self.manifest.text_sha256.nunique(),400)
        inside=self.manifest.stratum.isin(STRATA[1:3])
        self.assertTrue(self.manifest.loc[inside,'inclusion_probability'].eq(1).all())
        self.assertAlmostEqual(self.design['population_routing_fraction'],.04)

    def test_excludes_previous_failures_and_upstream_rows(self):
        excluded=self.full.iloc[:10].copy();excluded['jev_status']='failed'
        manifest,_=build_sample(self.full,{'past':excluded},candidate(),requested_n=400)
        self.assertFalse(set(manifest.text_sha256)&set(excluded.text_sha256))
        with self.assertRaises(ValueError):build_sample(self.full,{},candidate(),requested_n=10001)
        with self.assertRaises(ValueError):build_sample(self.full,{},candidate(),requested_n=50)

    def test_population_random_budget_matches_despite_oversampling(self):
        arrays=policy_arrays(self.sample,candidate(),.04)
        weights=1/self.sample.inclusion_probability.to_numpy()
        selective=arrays['Selective JEV'][1];random=arrays['Budget-matched random'][1]
        self.assertNotAlmostEqual(selective.sum(),random.sum())
        self.assertAlmostEqual(np.dot(weights,selective),np.dot(weights,random))
        self.assertAlmostEqual(np.dot(weights,selective)/weights.sum(),.04)

    def test_expected_random_decision_matches_enumerated_two_point_mixture(self):
        arrays=policy_arrays(self.sample,candidate(),.04)
        b=arrays['XGBoost tuned'][0];j=(self.sample.jev_score.to_numpy()>=.03).astype(float)
        np.testing.assert_allclose(arrays['Budget-matched random'][0],.96*b+.04*j)

    def test_failure_fallback_uses_frozen_tuned_comparator(self):
        sample=self.sample.copy();sample['jev_status']='failed';sample['jev_score']=np.nan
        a=policy_arrays(sample,candidate(),.04)
        for name in ['Selective JEV','Budget-matched random','One-sided OR']:
            np.testing.assert_array_equal(a[name][0],a['XGBoost tuned'][0])

    def test_one_sided_or_never_loses_comparator_positive(self):
        a=policy_arrays(self.sample,candidate(),.04)
        b=a['XGBoost tuned'][0]
        self.assertTrue(np.all(a['One-sided OR'][0]>=b))
        self.assertFalse(np.any(a['One-sided OR'][1][b==1]))

    def test_weighted_population_metrics_and_finite_frame_bootstrap(self):
        table,paired,transitions=analyze_probability_sample(self.sample,candidate(),self.design,replicates=150)
        t=table.set_index('policy')
        self.assertAlmostEqual(t.loc['Selective JEV','requests_per_1000'],40.)
        self.assertAlmostEqual(t.loc['Budget-matched random','requests_per_1000'],40.)
        self.assertGreater(t.loc['Selective JEV','sensitivity'],t.loc['XGBoost tuned','sensitivity'])
        self.assertTrue(np.allclose(table.prevalence,table.prevalence.iloc[0]))
        self.assertFalse(paired.filter(like='ci_').isna().any().any())
        self.assertIn('introduced_false_negatives_per_1000',transitions)

    def test_census_has_zero_finite_frame_sampling_uncertainty(self):
        data=self.sample.copy();data['population_n']=data.sample_n;data['inclusion_probability']=1.
        values=contributions(data.y_true,data.y_true,np.ones(len(data)))
        total,draws=design_totals(data,values,replicates=100)
        np.testing.assert_allclose(draws,np.broadcast_to(total,draws.shape))

    def test_incomplete_strata_bad_weights_and_author_clusters_refused(self):
        bad=self.sample.iloc[1:].copy()
        with self.assertRaises(ValueError):analyze_probability_sample(bad,candidate(),self.design,replicates=10)
        bad=self.sample.copy();bad.loc[0,'inclusion_probability']=.001
        with self.assertRaises(ValueError):analyze_probability_sample(bad,candidate(),self.design,replicates=10)
        bad=self.sample.assign(author_id='same-author')
        with self.assertRaises(ValueError):analyze_probability_sample(bad,candidate(),self.design,replicates=10)

    def test_sensitivity_requires_external_margin_and_strict_bound(self):
        self.assertFalse(sensitivity_decision(.01)['preservation_established'])
        with self.assertRaises(ValueError):sensitivity_decision(-.01,.02)
        self.assertTrue(sensitivity_decision(-.01,.02,'synthetic external justification')['preservation_established'])
        self.assertFalse(sensitivity_decision(-.02,.02,'synthetic external justification')['preservation_established'])

    def test_user_level_label_inheritance_not_claimed_as_post_expert_label(self):
        mapping=self.full[['row_id','text_sha256']].copy()
        mapping['source_record_id']=mapping.row_id.astype(str)
        mapping['source_label']=self.full.y_true
        post=verify_post_mapping(self.full,mapping,source_annotation_unit='post')
        self.assertTrue(post['post_expert_reference_compatible'])
        author=verify_post_mapping(self.full,mapping,source_annotation_unit='author')
        self.assertFalse(author['post_expert_reference_compatible'])
        mapping.loc[0,'source_label']=1-mapping.loc[0,'source_label']
        self.assertEqual(verify_post_mapping(self.full,mapping,source_annotation_unit='post')['matched_label_disagreements_n'],1)

    def test_local_baseline_fits_training_vocabulary_only(self):
        train=pd.DataFrame({'row_id':range(20),'text':['neutral friendly weather sunshine' if i%2==0 else 'first person distress help needed' for i in range(20)],
                            'y_true':[i%2 for i in range(20)],'text_sha256':[text_sha256('train'+str(i)) for i in range(20)]})
        dev=pd.DataFrame({'row_id':range(30,38),'text':['validationonlymarker friendly weather' if i%2==0 else 'validationonlymarker distress help needed' for i in range(8)],
                          'y_true':[i%2 for i in range(8)],'text_sha256':[text_sha256('dev'+str(i)) for i in range(8)]})
        model,metadata=fit_local_text_baseline(train,dev)
        word=dict(model.named_steps['features'].transformer_list)['word']
        self.assertNotIn('validationonlymarker',word.vocabulary_)
        self.assertEqual(metadata['evaluation_results'],'not computed by fitting helper')


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
        self.full=fixture_frame();self.manifest,self.design=build_sample(self.full,{},candidate(),requested_n=400)
        self.ref={'verified':True,'primary_source_sha256':'a'*64,'construct_definition':'synthetic fixture',
                  'rater_qualifications':'synthetic fixture','source_citation':'synthetic fixture','verified_by':'test fixture',
                  'record_mapping_report':{'post_expert_reference_compatible':True,'unmatched_records_n':0}}
        self.transfer={'permitted':True,'source_terms_sha256':'b'*64,'interpretation':'synthetic fixture'}
        self.frozen=freeze_for_collection(draft(),self.manifest,self.design,reference_evidence=self.ref,
            transfer_evidence=self.transfer,precision_justification='synthetic precision fixture',output=self.directory/'protocol.json')
        self.path=self.directory/'budget.sqlite3';self.transport=Transport()
        self.credentials=patch.dict('os.environ',{'METIS_API_KEY':'offline-study2-test-only'});self.credentials.start()

    def tearDown(self):self.credentials.stop();self.temp.cleanup()

    def row_text(self,row):return self.full.set_index('text_sha256').loc[row.text_sha256,'text']

    def client(self,**kwargs):
        options={'protocol':self.frozen,'run_new_inference':True,'session':self.transport}
        options.update(kwargs)
        return Study2JevClient(self.path,self.manifest,**options)

    def test_unresolved_reference_and_transfer_block_freezing(self):
        for reference,transfer in [(dict(self.ref,verified=False),self.transfer),(self.ref,dict(self.transfer,permitted=False))]:
            with self.assertRaises(ValueError):freeze_for_collection(draft(),self.manifest,self.design,reference_evidence=reference,
                transfer_evidence=transfer,precision_justification='test',output=self.directory/'bad.json')
        with self.assertRaises(CohortMismatch):self.client(protocol=draft())
        self.assertEqual(self.transport.calls,0)

    def test_manifest_and_frozen_prompt_changes_rejected(self):
        changed=copy.deepcopy(self.frozen);changed['candidate_policy']['question_sha256']='changed'
        with self.assertRaises(CohortMismatch):self.client(protocol=changed)
        changed=copy.deepcopy(self.frozen);changed['evaluation_manifest_sha256']='changed'
        with self.assertRaises(CohortMismatch):self.client(protocol=changed)

    def test_disabled_collector_and_status_make_zero_calls(self):
        status=collect(self.manifest,self.full,self.frozen,self.design,ledger_path=self.path)
        self.assertEqual(status['actual_new_attempts'],0)
        self.assertEqual(status['unused_allowance'],10000)
        self.assertFalse(self.path.exists())

    def test_atomic_cap_shared_across_workers_and_cache_read_after_restart(self):
        clients=[self.client(max_attempts=2) for _ in range(4)]
        def send(i):
            row=self.manifest.iloc[i]
            try:return clients[i].predict(row.sample_id,self.row_text(row))
            except BudgetExhausted:return 'capped'
        with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(send,range(4)))
        self.assertEqual(outcomes.count('capped'),2)
        self.assertEqual(ledger_status(self.path)['actual_new_attempts'],2)
        self.assertEqual(self.transport.calls,2)
        success=self.client(max_attempts=2).ledger_records()[0]['sample_id']
        row=self.manifest.set_index('sample_id').loc[success]
        self.assertEqual(self.client(max_attempts=2,run_new_inference=False).predict(success,self.row_text(row)),.8)
        self.assertEqual(self.transport.calls,2)
        with self.assertRaises(ValueError):self.client(max_attempts=10001)

    def test_failures_count_and_are_not_replayed(self):
        self.transport.failure=True
        row=self.manifest.iloc[0];client=self.client()
        with self.assertRaises(JevClientError):client.predict(row.sample_id,self.row_text(row))
        with self.assertRaises(JevClientError):self.client().predict(row.sample_id,self.row_text(row))
        self.assertEqual(self.transport.calls,1)
        self.assertEqual(ledger_status(self.path)['failed'],1)

    def test_no_text_or_outcomes_in_public_selection(self):
        bad=self.manifest.assign(y_true=0)
        with self.assertRaises(ValueError):freeze_for_collection(draft(),bad,self.design,reference_evidence=self.ref,
            transfer_evidence=self.transfer,precision_justification='test',output=self.directory/'bad.json')

    def test_complete_mock_collection_roundtrip_and_readonly_fresh_replay(self):
        from jev_research.study2_outputs import replay_fresh_if_complete,write_report
        from jev_research.budgeted_experiments import file_digest
        output=self.directory/'research/results'/STUDY
        private=self.directory/'artifacts'/STUDY;private.mkdir(parents=True)
        self.path=private/'budget.sqlite3'
        write_frozen_package(output/'frozen_v1',self.frozen,self.manifest,self.design)
        manifest=pd.read_json(output/'frozen_v1/manifest.json',precise_float=True)
        self.assertEqual(manifest_hash(manifest),self.frozen['evaluation_manifest_sha256'])
        self.full.to_csv(private/'evaluation_private.csv.gz',index=False,compression='gzip')
        collected=collect(manifest,self.full,self.frozen,self.design,ledger_path=self.path,
                          run_new_inference=True,workers=4,session_factory=lambda:self.transport)
        self.assertEqual(collected['attempts_spent'],400)
        before=file_digest(self.path)
        with patch('requests.sessions.Session.request',side_effect=AssertionError('Offline replay must never use HTTP')):
            status=replay_fresh_if_complete(self.directory)
        self.assertEqual(status['analysis_status'],'COLLECTED AND REPLAYED')
        self.assertEqual(status['actual_new_attempts'],400)
        self.assertEqual(before,file_digest(self.path))
        self.assertEqual(status['sensitivity_preservation']['status'],'not_adjudicated_no_defensible_margin')
        population=pd.read_json(output/'study2_main_population_results.json',precise_float=True).set_index('policy')
        self.assertAlmostEqual(population.loc['Selective JEV','requests_per_1000'],population.loc['Budget-matched random','requests_per_1000'])
        write_report({'results':output},status)
        text=(output/'manuscript_report.md').read_text()
        self.assertIn('400 actual live attempts',text)  # synthetic transport run only, in a temporary fixture
        self.assertNotIn('Study 2 results are pending',text)


class HistoricalEvidenceTests(unittest.TestCase):
    def test_reference_counts_and_repeat_decisions_are_recomputed(self):
        root=Path(__file__).resolve().parents[2]
        frame,prior=load_historical(root)
        checks=historical_checks(frame)
        x=checks.loc[checks.development_target_recall.eq(.9)].set_index('policy')
        self.assertEqual(x.loc['XGBoost','fn'],181);self.assertEqual(x.loc['JEV replacement','fn'],181)
        self.assertEqual(x.loc['XGBoost','fp'],1010);self.assertEqual(x.loc['JEV replacement','fp'],410)
        repeat=repeat_audit(frame,prior).set_index('threshold')
        self.assertEqual(repeat.loc[.03,'different_binary_decision_n'],1)
        self.assertEqual(repeat.loc[.08,'different_binary_decision_n'],0)
        self.assertEqual(repeat.loc[.5,'repeated_text_n'],21)


if __name__=='__main__':unittest.main()
