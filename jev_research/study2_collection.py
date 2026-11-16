"""Separate authorized live collector; notebook reruns never call this module.

Owner authorization: <=10,000 new attempts across this Study 2 ledger. It does
not establish source-label compatibility, data transfer permission or available
credentials. Those documented prerequisites are enforced before dispatch.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading

import pandas as pd

from .additional_experiment import AdditionalJevClient, manifest_hash
from .budgeted_experiments import digest, save_json
from .client import BASE_MODEL, QUESTION, CohortMismatch, JevClientError, text_sha256
from .study2 import ATTEMPT_CEILING, STUDY


def verify_post_mapping(posts, source_mapping, *, source_annotation_unit):
    """Check linkage, not the truth of a paper's annotation claims.

    Mapping must retain source IDs, exact text hashes and original numeric
    labels. An inherited author label is explicitly incompatible with a claim
    of an independently expert-rated individual post.
    """
    required={'row_id','text_sha256','source_record_id','source_label'}
    if not required.issubset(source_mapping):
        raise ValueError('Mapping requires dataset row, text hash, source ID and source label.')
    if source_mapping.row_id.duplicated().any() or posts.row_id.duplicated().any():
        raise ValueError('Ambiguous record linkage.')
    join=posts[['row_id','text_sha256','y_true']].merge(source_mapping,on='row_id',how='left',suffixes=('_dataset','_source'))
    matched=join.source_record_id.notna() & join.text_sha256_dataset.eq(join.text_sha256_source)
    same=matched & join.y_true.eq(join.source_label)
    return {'dataset_records_n':len(posts),'matched_records_n':int(matched.sum()),
            'unmatched_records_n':int((~matched).sum()),'matched_label_disagreements_n':int((matched&~same).sum()),
            'source_annotation_unit':source_annotation_unit,
            'post_expert_reference_compatible':bool(source_annotation_unit=='post' and matched.all() and same.all()),
            'unit_mismatch':source_annotation_unit!='post',
            'limitation':'Exact record linkage does not independently verify specialist qualifications or annotation procedures.'}


def freeze_for_collection(draft,manifest,design,*,reference_evidence,transfer_evidence,precision_justification,output):
    ref=reference_evidence
    required=['primary_source_sha256','construct_definition','rater_qualifications','source_citation','verified_by']
    if not ref.get('verified') or not all(ref.get(k) for k in required):
        raise ValueError('Primary annotation source and construct must be documented before the main budget is spent.')
    mapping=ref.get('record_mapping_report',{})
    if not mapping.get('post_expert_reference_compatible') or mapping.get('unmatched_records_n',1)!=0:
        raise ValueError('Post-level expert-reference linkage is unresolved or incompatible.')
    if not transfer_evidence.get('permitted') or not transfer_evidence.get('source_terms_sha256') or not transfer_evidence.get('interpretation'):
        raise ValueError('Applicable permission for external JEV text transfer must be documented.')
    if not precision_justification:
        raise ValueError('Sample size needs design-aware precision justification; 10000 is only a ceiling.')
    if len(manifest)>ATTEMPT_CEILING or len(manifest)!=sum(s['sample_n'] for s in design['strata']):
        raise ValueError('Manifest/sample design does not fit the shared authorized ceiling.')
    if any(c in manifest for c in ['text','cleaned_text','y_true','y_xgb','error_type']):
        raise ValueError('The public frozen selection manifest must not contain text or evaluation outcomes.')
    if not manifest.study_partition.eq('evaluation').all():
        raise ValueError('This design plans zero paid pilots/repeats. Version a separate role-aware design before adding them; never mix them into evaluation.')
    frozen=json.loads(json.dumps(draft))
    frozen.update({'status':'FROZEN_INTERNAL_PROTOCOL','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                   'evaluation_manifest_sha256':manifest_hash(manifest),'design_sha256':digest(design),
                   'reference_evidence':ref,'transfer_evidence':transfer_evidence,
                   'precision_justification':precision_justification})
    frozen['gates'].update({k:True for k in ['primary_source_and_construct_verified','expert_record_mapping_verified',
                                           'external_transfer_terms_permit','complete_sampling_frame_verified','frozen_manifest_and_protocol']})
    frozen['gates'].pop('live_credential_available',None)
    frozen['runtime_credentials']='Checked at dispatch; no secret values stored in protocol.'
    save_json(output,frozen)
    return frozen


def write_frozen_package(directory,protocol,manifest,design):
    """Preserve full float round-trip precision for digest-bound manifests."""
    if manifest_hash(manifest)!=protocol.get('evaluation_manifest_sha256') or digest(design)!=protocol.get('design_sha256'):
        raise ValueError('Package does not match its frozen protocol.')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    save_json(directory/'protocol.json',protocol)
    save_json(directory/'manifest.json',manifest.to_dict('records'))
    save_json(directory/'design.json',design)


class Study2JevClient(AdditionalJevClient):
    """Reuse verified atomic accounting, with a NEW, separately scoped cap."""
    ATTEMPT_CAP=ATTEMPT_CEILING
    SAMPLE_CAP=ATTEMPT_CEILING

    def __init__(self,ledger_path,cohort,*,protocol,run_new_inference=False,max_attempts=ATTEMPT_CEILING,session=None,**money):
        if protocol.get('status')!='FROZEN_INTERNAL_PROTOCOL':
            raise CohortMismatch('Study 2 protocol is a draft; no evaluation dispatch is permitted.')
        required=['primary_source_and_construct_verified','expert_record_mapping_verified','external_transfer_terms_permit',
                  'complete_sampling_frame_verified','frozen_manifest_and_protocol']
        if not all(protocol.get('gates',{}).get(k) is True for k in required):
            raise CohortMismatch('Scientific/data prerequisites remain unresolved; no paid request permitted.')
        if protocol.get('evaluation_manifest_sha256')!=manifest_hash(pd.DataFrame(cohort)):
            raise CohortMismatch('Manifest differs from the frozen Study 2 protocol.')
        candidate=protocol.get('candidate_policy',{})
        if candidate.get('model')!=BASE_MODEL or candidate.get('question_sha256')!=digest(QUESTION):
            raise CohortMismatch('Frozen prompt/model differs from the supported transport.')
        self.protocol=protocol;self.protocol_sha256=digest(protocol)
        super().__init__(ledger_path,cohort,run_new_inference=run_new_inference,
                         authorized_manifest_sha256=protocol['evaluation_manifest_sha256'],
                         max_attempts=max_attempts,session=session,**money)

    def _authorize_cohort(self,cohort):
        required={'sample_id','text_sha256','p_xgb','source_split','study_partition'}
        if not required.issubset(cohort) or not 1<=len(cohort)<=self.SAMPLE_CAP:
            raise CohortMismatch('Require a frozen Study 2 cohort of at most 10000 posts.')
        if cohort.sample_id.duplicated().any() or cohort.text_sha256.duplicated().any():
            raise CohortMismatch('No duplicate Study 2 dispatch identities/texts.')
        records={}
        for r in cohort.to_dict('records'):
            if r['study_partition']!='evaluation' or r['source_split']!='test':
                raise CohortMismatch('This frozen design permits evaluation posts only, not unplanned adaptation or repeat calls.')
            p=r['p_xgb'];h=r['text_sha256'];s=r['sample_id']
            if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:
                raise CohortMismatch('Invalid baseline score.')
            if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) or not isinstance(s,str) or not s:
                raise CohortMismatch('Invalid dispatch identity.')
            records[s]={k:r[k] for k in sorted(required)}
        return records

    def _cohort_identity(self):
        return {'study_id':STUDY,'protocol_sha256':self.protocol_sha256,'manifest_sha256':self.manifest_sha256,
                'money':self.money,'question':QUESTION,'model':BASE_MODEL,
                'cohort':sorted(self.cohort.values(),key=lambda r:r['sample_id'])}

    def _initialize(self):
        super()._initialize()
        with self._connection() as db:
            db.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES ('evaluation_decisions_sha256',?)",(self.protocol_sha256,))


def ledger_status(path):
    path=Path(path)
    if not path.exists():
        return {'actual_new_attempts':0,'successes':0,'failed':0,'pending_uncertain':0,
                'unused_allowance':ATTEMPT_CEILING,'ledger_exists':False,'study_id':STUDY}
    uri=path.resolve().as_uri()+'?mode=ro'
    with sqlite3.connect(uri,uri=True) as db:
        counts=dict(db.execute('SELECT status,COUNT(*) FROM attempts GROUP BY status'))
        n=sum(counts.values())
    if n>ATTEMPT_CEILING:raise ValueError('Ledger exceeds authorization; preserve and diagnose.')
    return {'actual_new_attempts':n,'successes':counts.get('succeeded',0),'failed':counts.get('failed',0),
            'pending_uncertain':counts.get('pending',0),'unused_allowance':ATTEMPT_CEILING-n,
            'ledger_exists':True,'study_id':STUDY}


def collect(manifest,source_frame,protocol,design,*,ledger_path,run_new_inference=False,workers=4,session_factory=None,**money):
    if run_new_inference is not True:
        return {**ledger_status(ledger_path),'dispatch':'disabled'}
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=8:
        raise ValueError('Use one to eight bounded workers.')
    if digest(design)!=protocol.get('design_sha256'):
        raise CohortMismatch('Sampling design differs from frozen protocol.')
    if source_frame.text_sha256.duplicated().any():raise CohortMismatch('Ambiguous source text identity.')
    source=source_frame.set_index('text_sha256')
    for r in manifest.itertuples(index=False):
        s=source.loc[r.text_sha256]
        if text_sha256(s.text)!=r.text_sha256 or not math.isclose(float(s.p_xgb),r.p_xgb,rel_tol=0,abs_tol=1e-14):
            raise CohortMismatch('Source text or baseline score differs before collection.')
        for field in ['row_id','dataset_sha256','baseline_fingerprint','source_split']:
            if s[field]!=getattr(r,field):raise CohortMismatch('Source provenance differs before collection.')
    def client():
        return Study2JevClient(ledger_path,manifest,protocol=protocol,run_new_inference=True,
                              session=session_factory() if session_factory else None,**money)
    coordinator=client();already={r['sample_id'] for r in coordinator.ledger_records()}
    todo=iter(r for r in manifest.itertuples(index=False) if r.sample_id not in already)
    local=threading.local()
    def initialize():local.client=client()
    def predict(row):
        try:
            local.client.predict(row.sample_id,source.loc[row.text_sha256,'text'])
            return True
        except JevClientError:
            return False
    stopped=False
    with ThreadPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        pending=set()
        for _ in range(workers):
            r=next(todo,None)
            if r is not None:pending.add(pool.submit(predict,r))
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                if not future.result():stopped=True
            if not stopped:
                for _ in done:
                    r=next(todo,None)
                    if r is not None:pending.add(pool.submit(predict,r))
    if stopped:raise JevClientError('Study 2 stopped scheduling after an error; preserve ledger and diagnose before resuming unattempted rows.')
    return coordinator.ledger_summary()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,default=Path('artifacts')/STUDY/'budget.sqlite3')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--collect',action='store_true',help='Activate the already owner-authorized live phase after all scientific prerequisites are verified.')
    parser.add_argument('--manifest',type=Path);parser.add_argument('--protocol',type=Path)
    parser.add_argument('--design',type=Path);parser.add_argument('--private-source',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not args.collect:
        print(json.dumps(ledger_status(args.ledger),indent=2));return
    if any(p is None for p in [args.manifest,args.protocol,args.design,args.private_source]):
        parser.error('Collection requires frozen manifest/protocol/design and private source text table.')
    result=collect(pd.read_json(args.manifest,precise_float=True),
                   pd.read_csv(args.private_source,keep_default_na=False,float_precision='round_trip'),
                   json.loads(args.protocol.read_text()),json.loads(args.design.read_text()),
                   ledger_path=args.ledger,run_new_inference=True,workers=args.workers)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
