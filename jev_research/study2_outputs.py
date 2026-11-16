"""Read-only fresh evidence replay and publication artifacts; never dispatch."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .additional_experiment import manifest_hash
from .budgeted_experiments import digest,file_digest,save_json,save_table
from .client import BASE_MODEL,ENDPOINT,QUESTION,text_sha256
from .study2 import (STUDY,STRATA,analyze_probability_sample,policy_arrays,
                     contributions,design_totals,summarize_totals,sensitivity_decision)
from .study2_collection import ledger_status


def routing_stratum_table(frame,protocol,design):
    rows=[];arrays=policy_arrays(frame,protocol,design['population_routing_fraction'])
    for stratum in STRATA[1:3]:
        keep=frame.stratum.eq(stratum).to_numpy();part=frame.loc[keep].reset_index(drop=True)
        if part.empty:raise ValueError('Missing difficult stratum.')
        for name,(q,req) in arrays.items():
            total,draws=design_totals(part,contributions(part.y_true,q[keep],req[keep]),replicates=2000)
            m,b=summarize_totals(total),summarize_totals(draws)
            row={'stratum':stratum,'policy':name,'observed_posts_n':len(part),'stratum_population_n':int(part.population_n.iloc[0]),
                 'counts_type':'finite-frame weighted totals; expected totals for random routing',
                 'tn_estimated':float(total[0]),'fp_estimated':float(total[1]),'fn_estimated':float(total[2]),'tp_estimated':float(total[3])}
            for k,v in m.items():
                row[k]=float(v);row[k+'_ci_low'],row[k+'_ci_high']=np.quantile(b[k],[.025,.975]).tolist()
            rows.append(row)
    return pd.DataFrame(rows)


def replay_fresh_if_complete(root):
    root=Path(root);output=root/'research/results'/STUDY
    package=output/'frozen_v1';private=root/'artifacts'/STUDY
    ledger=private/'budget.sqlite3';status=ledger_status(ledger)
    files={'protocol':package/'protocol.json','manifest':package/'manifest.json',
           'design':package/'design.json','source':private/'evaluation_private.csv.gz'}
    if not any(p.exists() for p in files.values()) and not ledger.exists():
        return {**status,'analysis_status':'NOT COLLECTED','fresh_effect':None,
                'sensitivity_preservation':'not adjudicated; no fresh reference data or justified margin',
                'selective_beats_equal_cost_random':'not estimable'}
    missing=[name for name,p in files.items() if not p.exists()]
    if missing:
        return {**status,'analysis_status':'INCOMPLETE PACKAGE','missing_files':missing,'fresh_effect':None}
    protocol=json.loads(files['protocol'].read_text());manifest=pd.read_json(files['manifest'],precise_float=True)
    design=json.loads(files['design'].read_text())
    if protocol.get('status')!='FROZEN_INTERNAL_PROTOCOL' or manifest_hash(manifest)!=protocol['evaluation_manifest_sha256'] or digest(design)!=protocol['design_sha256']:
        raise ValueError('Fresh evidence package is not the frozen collection design.')
    if not ledger.exists() or status['actual_new_attempts']!=len(manifest):
        return {**status,'analysis_status':'COLLECTION INCOMPLETE','planned_sample_n':len(manifest),'fresh_effect':None}
    with sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY id')]
        saved=dict(db.execute('SELECT key,value FROM metadata'))
    if saved.get('evaluation_decisions_sha256')!=digest(protocol):raise ValueError('Ledger protocol seal mismatch.')
    by_id=manifest.set_index('sample_id');seen=set();records=[]
    for r in rows:
        if r['sample_id'] in seen or r['sample_id'] not in by_id.index:raise ValueError('Duplicate/unselected dispatch in ledger.')
        seen.add(r['sample_id']);s=by_id.loc[r['sample_id']]
        expected=digest({'sample_id':r['sample_id'],'text_sha256':s.text_sha256,'model':BASE_MODEL,
                         'question_sha256':digest(QUESTION),'endpoint':ENDPOINT})
        if r['request_key']!=expected or r['text_sha256']!=s.text_sha256 or r['instruction_sha256']!=digest(QUESTION) or r['model']!=BASE_MODEL:
            raise ValueError('Fresh request evidence does not match its frozen authorization.')
        if datetime.fromisoformat(r['started_at'])<datetime.fromisoformat(protocol['frozen_at_utc']):
            raise ValueError('Evaluation request preceded protocol freeze.')
        if r['status'] not in ['succeeded','failed','pending']:raise ValueError('Unknown outcome status.')
        if r['status']=='succeeded':
            if isinstance(r['score'],bool) or not isinstance(r['score'],(float,int)) or not np.isfinite(r['score']) or not 0<=r['score']<=1:
                raise ValueError('Invalid score in fresh ledger.')
        elif r['score'] is not None:raise ValueError('Unresolved outcome has a stale score.')
        records.append({'sample_id':r['sample_id'],'jev_score':r['score'],'jev_status':r['status']})
    source=pd.read_csv(files['source'],keep_default_na=False,float_precision='round_trip').set_index('text_sha256')
    if source.index.duplicated().any():raise ValueError('Duplicate private source texts.')
    frame=manifest.merge(pd.DataFrame(records),on='sample_id',validate='one_to_one')
    truth=[]
    for r in frame.itertuples(index=False):
        s=source.loc[r.text_sha256]
        if text_sha256(s.text)!=r.text_sha256 or s.row_id!=r.row_id or not np.isclose(s.p_xgb,r.p_xgb,rtol=0,atol=1e-14):
            raise ValueError('Fresh private source identity changed.')
        truth.append(s.y_true)
    frame['y_true']=truth
    if 'author_id' in source:frame['author_id']=source.loc[frame.text_sha256,'author_id'].to_numpy()
    population,paired,transitions=analyze_probability_sample(frame,protocol['candidate_policy'],design)
    strata=routing_stratum_table(frame,protocol['candidate_policy'],design)
    for name,table in [('study2_main_population_results',population),('study2_paired_comparisons',paired),
                       ('study2_transitions',transitions),('study2_routing_stratum_results',strata)]:save_table(output,name,table)
    save_table(output,'fresh_prediction_records',frame.drop(columns=['author_id'],errors='ignore'))
    primary=paired.loc[paired.policy.eq('Selective JEV')&paired.reference.eq('XGBoost tuned')].iloc[0]
    decision=sensitivity_decision(float(primary.delta_sensitivity_ci_low),protocol.get('noninferiority_margin'),protocol.get('margin_justification'))
    random=paired.loc[paired.policy.eq('Selective JEV')&paired.reference.eq('Budget-matched random')].iloc[0]
    return {**status,'analysis_status':'COLLECTED AND REPLAYED','fresh_effect':primary.to_dict(),
            'sensitivity_preservation':decision,
            'selective_beats_equal_cost_random':bool(random.delta_false_alarms_per_1000_ci_high<0),
            'independence':design['independence'],'collection_first_started_utc':min(r['started_at'] for r in rows),
            'collection_last_finished_utc':max((r['finished_at'] for r in rows if r['finished_at']),default=None),
            'uncertainty_scope':'stratified finite-frame sampling conditional on fixed ratings/policy; not future-population or provider variability'}


def reviewer_table():
    rows=[
      ('Are the labels expert judgments of individual posts?','Owner supplied IEEE 9591887 link; repository transformation audited','Primary paper inaccessible; no record mapping or verified annotation unit','Supplied binary labels; specialist-reference claim remains unverified'),
      ('Does Study 2 provide independent validation?','Historical population exclusions enumerated; label-blind sampler tested','No fresh data collected; old test labels were previously aggregated; author/time metadata absent','Study 1 exploratory development evidence; Study 2 independence not yet achieved'),
      ('Is preserved sensitivity established?','Paired historical intervals and explicit decision-bound function','No justified clinical noninferiority margin or fresh expert outcome','Estimate sensitivity differences; nonsignificance is not preservation'),
      ('Does selective routing beat random at equal cost?','Same JEV threshold/fallback; population referral fraction, weights and expected random result implemented','No outside-range JEV scores or fresh common sample','Random-routing superiority is not estimable from existing cache'),
      ('Are uncertain failures and charges counted?','Atomic retained cap tests, failure/restart tests; separate 10000-attempt ledger','No live credential, observed price or provider usage supplied','Zero new real attempts; 10000 remains unused; monetary cost unknown'),
      ('Are repeat outputs stable?','21 overlap pairs; decisions recalculated at three thresholds','Error-enriched repeat set; mutable alias; n=21','1/21 changed at .03 and 0/21 at .08/.50; wide descriptive intervals'),
      ('Would a strong local model perform similarly?','Training-only word/character TF-IDF logistic pipeline and leakage test','Original raw upstream training/development data absent','Local baseline implemented and unrun on real research data'),
      ('What explains psychologically consequential errors?','Separate probability/enriched review samples and blinded forms','Primary construct definitions, text and qualified reviewers unavailable','New human coding pending; no generated expert annotations'),
      ('Are source ethics and external transfer permitted?','Provenance table distinguishes source and current review','Source approval, current approval, consent, license and transfer terms unverified','No current clinical/ethics approval or external-transfer permission is asserted'),
      ('Is reporting clinically interpretable?','Per-1000 flags/errors/requests and paired design-aware analysis specified','No future suicide outcome, intervention effect or clinical deployment validation','Post-label prediction under a pending construct definition; not individual future-suicide risk'),
    ]
    return pd.DataFrame(rows,columns=['likely_reviewer_concern','evidence_supplied','unresolved_limitation','wording_the_manuscript_can_defend'])


def make_figures(study):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    out=study['results']/'figures';out.mkdir(exist_ok=True)
    execution=json.loads((study['results']/'execution_manifest.json').read_text())
    collected=execution['analysis_status']=='COLLECTED AND REPLAYED'
    fig,ax=plt.subplots(figsize=(10,9));ax.set(xlim=(0,10),ylim=(0,11));ax.axis('off')
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#334155',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    p=study['planning']
    box(2.1,9.5,5.8,.85,'Original saved test population: 46,342 text records\nSource post/expert-label unit and linkage unverified','#e2e8f0')
    box(.3,7.6,4.2,1.0,'Historical probability cohort: 5,000\nUsed for development; no fresh validation','#dbeafe')
    box(5.5,7.6,4.2,1.0,'Additional earlier error-selected test: 19\nPrior failures are excluded, never resent','#dbeafe')
    for x in [2.4,7.6]:ax.annotate('',xy=(x,8.7),xytext=(5,9.4),arrowprops={'arrowstyle':'->','color':'#475569'})
    box(1.4,5.4,7.2,1.1,f'Potential residual test frame: {p["inferred_remaining_test_n_before_other_independence_exclusions"]:,}\nResidual [0.25,0.80) route: {p["inferred_remaining_eligible_n_before_other_exclusions"]:,}\nAggregate-derived counts; author/near-duplicate exclusions pending','#dcfce7')
    for x in [2.4,7.6]:ax.annotate('',xy=(5,6.65),xytext=(x,7.5),arrowprops={'arrowstyle':'->','color':'#475569'})
    readiness=('Frozen source, mapping, transfer and frame evidence retained\nSee frozen protocol and execution manifest' if collected else
               'Collection prerequisites unresolved\nPrimary annotation source + record linkage + transfer terms\nOriginal score/text frame + live credential + frozen manifest')
    box(1.4,3.0,7.2,1.25,readiness,'#dcfce7' if collected else '#fef3c7')
    ax.annotate('',xy=(5,4.4),xytext=(5,5.25),arrowprops={'arrowstyle':'->','color':'#475569'})
    last=('Study 2: collected and replayed' if collected else 'Study 2: NOT COLLECTED')
    last+=f'\nActual new JEV attempts: {execution["actual_new_attempts"]:,} / authorized ceiling 10,000'
    box(1.4,.9,7.2,1.0,last,'#dcfce7' if collected else '#fee2e2')
    ax.annotate('',xy=(5,2.05),xytext=(5,2.85),arrowprops={'arrowstyle':'->','color':'#475569'})
    ax.set_title('Text-record evidence flow and current Study 2 status',fontsize=14,pad=18)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(out/('post_flow.'+ext),dpi=250,bbox_inches='tight')
    plt.close(fig)
    # Missing Study 2 outcomes are never drawn as zeros. In the development
    # figure, compare the two historical primary operating points; do not label
    # an independently tuned OR cutoff as OR under the frozen comparator.
    if collected:
        table=pd.read_json(study['results']/'study2_main_population_results.json',precise_float=True)
        selected=table.set_index('policy').loc[['XGBoost tuned','Selective JEV','Budget-matched random','One-sided OR']]
        labels=['Tuned XGBoost','Selective JEV','Equal-cost random','One-sided OR']
        columns=['false_alarms_per_1000','missed_per_1000','requests_per_1000']
        title='Study 2: weighted finite-frame outcomes and policy request burden'
        footer=f'Paired stratified finite-frame intervals; actual new collection attempts = {execution["actual_new_attempts"]:,}.'
    else:
        source=study['root']/'research/results/jev_probability_hybrid_budgeted_experiments/matched_recall_evaluation.json'
        table=pd.read_json(source,precise_float=True)
        selected=table.loc[table.target.eq(.95)].set_index('policy').loc[['XGB threshold (target)','JEV replacement']]
        labels=['Tuned XGBoost','JEV replacement'];columns=['fp','fn','prospective_requests']
        title='Study 1 exploratory 3,000-text subset: thresholds selected for 0.95 development recall'
        footer='Historical cache replay only. Study 2 selective/random outcomes have not been collected. New live attempts = 0.'
    colors=['#64748b','#0f766e','#2563eb','#b45309'][:len(labels)]
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    for ax,column,panel_title in zip(axes,columns,['False alarms / 1,000','Missed positives / 1,000','Prospective JEV requests / 1,000']):
        values=selected[column] if collected else selected[column]/selected.n*1000
        bars=ax.bar(np.arange(len(labels)),values,color=colors)
        if collected:
            ax.vlines(np.arange(len(labels)),selected[column+'_ci_low'],selected[column+'_ci_high'],color='black',lw=1)
        for bar,val in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values.max()*.025,1),f'{val:.1f}',ha='center',fontsize=10)
        ax.set_xticks(np.arange(len(labels)),labels,rotation=25,ha='right');ax.set_title(panel_title,fontsize=11)
        upper=max(values.max(),selected[column+'_ci_high'].max()) if collected else values.max()
        ax.set_ylim(0,max(upper*1.2,1));ax.spines[['top','right']].set_visible(False)
    fig.suptitle(title,fontsize=12)
    fig.text(.5,.005,footer,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.93))
    for ext in ['png','pdf','svg']:fig.savefig(out/('policy_outcomes_and_request_burden.'+ext),dpi=250,bbox_inches='tight')
    plt.close(fig)
    return out


def write_report(study,status):
    r=study['results'];reviewer=reviewer_table();save_table(r,'reviewer_readiness',reviewer)
    if status['analysis_status']=='COLLECTED AND REPLAYED':
        return write_collected_report(study,status,reviewer)
    checks=study['tables']['historical_reference_checks'];rep=study['tables']['historical_repeat_audit']
    date=study['protocol']['created_at_tehran'][:10]
    text=f'''# Selective JEV assistance for suicide-related text flags: development evidence and a validation design

**Manuscript draft dated {date}. Study 2 results are pending, not a completed validation study.**

## Structured abstract

**Objective.** Evaluate whether a selective external-model policy reduces false-positive flags per 1,000 posts relative to threshold-tuned XGBoost, while estimating sensitivity against a construct-compatible human reference. Preservation requires a defensible margin and confidence bound.

**Methods.** We inspected the two executed notebooks, source pipeline, historical 5,000-case prediction export and earlier error-only cohort. We reproduced paired exploratory comparisons, audited repeated scores, specified a stratified common-sample validation design, and tested atomic collection accounting and weighted paired policy replay. The owner supplied IEEE document 9591887 as the proposed primary annotation source. Its annotation unit, specialist qualifications and record linkage to Kaggle version 14 could not be verified in this environment.

**Results.** In the reused 3,000-post exploratory subset at the 0.90 development recall target, XGBoost and JEV each missed 181 positive posts; false alarms were 1,010 and 410. At the 0.95 target, misses were 74 and 64 and false alarms 1,193 and 625. The paired sensitivity difference at the latter target was +0.62 percentage points (95% conditional row-bootstrap interval -0.80 to +1.94), excluding neither a small loss nor a gain. Of 21 repeated, error-enriched texts, 11 had different scores but only 1 changed a decision at 0.03 (4.76%; Wilson 95% interval 0.85%–22.67%); none changed at 0.08 or 0.50 (upper Wilson limit 15.46%). Study 2 has no collected posts, estimated primary effect, equal-cost random comparison or new expert judgments. Actual additional live attempts: {status['actual_new_attempts']}; unused authorized allowance: {status['unused_allowance']}.

**Conclusions.** Cached evidence supports fewer false alarms at similar achieved sensitivity on previously inspected texts. It does not establish expert-reference validity, clinical sensitivity preservation, independent replication or superiority of selective over budget-matched random routing. Those claims require the blocked source verification and fresh collection.

## Methods

### Reference construct and provenance

Primary source supplied by the owner: {study['protocol'].get('primary_source', 'https://ieeexplore.ieee.org/document/9591887')}. Article metadata, full text and supplements were not retrievable because the enforced network policy returned proxy 403 responses. No title, DOI, rater count, agreement statistic, ethics approval or license is invented. The owner did not know the paper-to-CSV mapping. The repository identifies Kaggle `nikhileswarkomati/suicide-watch` version 14 and SHA-256 `1ec16b065a2fae36488de16e8abfc326586a576a25901e09467223e545ddd295`, but this is not evidence of specialist labeling. Matched/unmatched source-record counts are not estimable without source records. The provenance table separates missing evidence from negative evidence.

The current model input is a single exported text record. In descriptions of the intended review task, "post" operationally denotes that record; equivalence to a single original source post remains unverified. If the primary paper labels authors after reviewing multiple posts, assigning an author label to each post changes the target and context; even an exact identity join would not establish post-level expert assessment. No future suicide outcome or clinical diagnosis is observed here. Verified existing specialist labels would be retained as the primary reference; bulk reannotation is not proposed as a publication prerequisite.

### Development and primary candidate

All 5,000 historical probability-selected cases are available for development; they are not independent evaluation. Earlier known-error selections remain separate. A conditional candidate uses routing [0.25,0.80), XGBoost cutoff {study['candidate']['xgb_threshold']:.15g}, JEV cutoff {study['candidate']['jev_threshold']:.15g}, no primary calibration, the original fixed risk-rating question and mutable alias `jev-latest`. An unavailable required response retains the tuned comparator decision. This differs explicitly from Study 1's XGB@0.50 fallback. The development recall target refers to the historical score-range population, not guaranteed global recall. These parameters are a dated candidate, **not a collection-frozen evaluation protocol**, until reference compatibility, transfer terms and sampling data are verified.

The target review task is hypothetical prioritization of posts for qualified human assessment. A flag would request contextual review; it is not a diagnosis, evidence of current intent, a verified future event or an implemented intervention. No acceptable clinical noninferiority margin has been supplied or verified from literature. Sensitivity differences and bounds will be reported; illustrative 1/2/5-percentage-point margins cannot supply the primary preservation claim.

### Sampling, budget and independence

The preferred frame is new authors or a later, compatible expert-labeled collection. If unavailable, use unqueried original test posts after all prior training/development/JEV/error-study and duplicate exclusions. Author/near-duplicate checks require matching source metadata. Aggregate arithmetic leaves a potential 41,323-post old-test frame and 1,062 original-interval posts before further exclusions, radically reducing the residual referral fraction to about 2.57%. These are planning counts, not a recovered sampling frame. Prior use of test labels in baseline aggregate reports remains disclosed.

The tested fallback design uses disjoint outside-low, inside-below-comparator, inside-above-comparator and outside-high score strata, known positive inclusion probabilities, a census of the remaining difficult route where feasible, and probability sampling of both exterior strata. Selection uses scores/identities only. Historical estimates support routed-population precision scenarios; unobserved exterior disagreements prevent a defensible final random-comparison precision calculation. Hence no final sample size or 10,000-row manifest is fabricated. The ceiling is 10,000 cumulative attempts, not a quota. No paid repeats are currently planned because the existing repeat observations should be used first.

### Comparators and statistical analysis

One response per sampled fresh post supports replay of the primary selective policy, tuned XGBoost, one-sided OR and budget-matched random routing. The random policy uses the same JEV cutoff and fallback. Its marginal referral probability equals the **target-frame** selective referral fraction, not the oversampled subset fraction. Expected confusion totals over a uniform fixed-size random referral set identify the expected policy result; ratios are computed from expected totals and are not the average of randomized ratios. A local word/character TF-IDF logistic model is implemented, but no real-data fit was possible without upstream texts. Calibration/fusion remain secondary.

Inverse-inclusion weighted paired totals provide prevalence, sensitivity, precision, specificity, false alarms, misses, flags and requests per 1,000, with transition decomposition and benefit per 100 requests. Stratified paired bootstrap perturbations retain fixed stratum sizes and population weights and apply finite-frame correction, including zero finite-frame sampling variance in a census stratum. These intervals condition on the observed reference labels, provider responses and fixed policy. They do not quantify future-population generalization, annotation error, policy-selection uncertainty or provider variability. Repeated known authors trigger refusal of an unsupported row analysis until an appropriate cluster design is supplied. Ordinary McNemar is not applied to weighted or expected-random confusion totals.

### Focused human review

The package prepares a probability sample of 100 historical cache texts and an enriched sample of 40 transitions, with any overlap disclosed. The probability sample can audit this cache's label fidelity; enriched frequencies cannot estimate prevalence. Model predictions, labels and transition names are omitted from private blinded forms. A provisional descriptive guide covers perspective, temporal context, ideation versus expressed intent, quotation, negation, help-seeking and insufficient evidence/context. Source-paper definitions must ground final coding once retrieved. Qualified reviewers, independent coding, adjudication and scale-appropriate agreement are pending. No language model has generated expert judgments.

## Results

The historical reference checks and repeat audit above are the measured results available here. `study2_main_population_results.json` and `study2_routing_stratum_results.json` explicitly mark fresh estimates **NOT COLLECTED**, rather than filling them with historical results or zeros. No Study 2 sensitivity-preservation criterion has been met or adjudicated, and selective routing versus equal-cost random remains unestimated. The post-flow figure describes these exclusions and blockers. The performance figure is explicitly labeled Study 1, so its favorable false-alarm result cannot be mistaken for replication. The separate 10,000-attempt collection client has only been exercised with mock transport.

## Psychology-focused discussion

Fewer false-positive flags may reduce unnecessary review, but a small or uncertain change in sensitivity must be carried alongside that benefit. An extremely low raw score cutoff such as 0.03 does not imply 3% individual suicide risk. The distinction between ideation, discussion, expressed intent and human-assessed risk is central: improvements against inherited subreddit/category/author labels would answer a different question from improvements against verified post-level specialist judgments. The current documentation does not resolve which construct the reference measures.

Model disagreement can motivate focused contextual review without replacing the existing reference labels. Repeated-score differences should be judged at actual operating points: the 11 numerical discrepancies overstate observed binary instability at 0.08/0.50, but zero changes among 21 nonrepresentative repeats is weak evidence of global stability. A negative fresh routing result, meaningful sensitivity loss or no selective advantage over random would be reported directly. The modest prior incremental fusion benefit does not justify centering complex fusion without new evidence.

## Ethics, privacy and reporting

Source-study approval/exemption, consent, dataset access/license, current-study review and permission to transfer text to JEV remain unverified. A source approval would not automatically cover the present study. No assertion of ethical approval, licensed transfer or clinical readiness is made. Sensitive text, author identifiers, credentials and raw provider payloads stay outside public artifacts. Frozen hashes and aggregate/public-safe manifests support reproducibility. TRIPOD+AI domains relevant to data source, target, reference, validation, missingness, calibration and uncertainty are assessed in the reviewer table; the package does not claim completed clinical-prediction reporting or validation. There is no measured future behavior, diagnosis or intervention benefit.

## Limitations and data/code availability

Required source-paper access, record mapping, complete raw score/text frames, source terms and live authentication are missing. Source annotation unit and expert coverage cannot be verified; no independent Study 2 population or outcomes have been collected. Historical confidence intervals condition on frozen policies and row-level sampling and do not remove prior test reuse. `jev-latest` has no immutable backend revision in the exports. Actual currency cost and historical per-request usage are unavailable.

Code, executed notebook, dated design draft/digest, provenance, historical checks, review sampling and reviewer-readiness table are supplied in this repository. Original private datasets/models/ledgers and new raw text are not redistributed. Collection and fresh offline replay commands are documented in `research/study2_collection_and_reporting.md`. Notebook reruns make zero paid calls. This is a reproducible development-and-protocol package, not a completed expert-reference validation manuscript.
'''
    (r/'manuscript_report.md').write_text(text)
    # Compact main historical table with precise population labeling.
    main=checks[['policy','development_target_recall','n','recall','precision','fn','fp','prospective_requests']].copy()
    save_table(r,'main_historical_results',main)
    save_json(r/'execution_manifest.json',status)
    return reviewer


def write_collected_report(study,status,reviewer):
    """Do not overwrite collected evidence with the pending-results template."""
    r=study['results'];primary=status['fresh_effect']
    frozen=json.loads((r/'frozen_v1/protocol.json').read_text())
    population=pd.read_json(r/'study2_main_population_results.json',precise_float=True)
    paired=pd.read_json(r/'study2_paired_comparisons.json',precise_float=True)
    save_json(r/'execution_manifest.json',status)
    ref=frozen['reference_evidence']
    effect=-primary['delta_false_alarms_per_1000']
    low=-primary['delta_false_alarms_per_1000_ci_high'];high=-primary['delta_false_alarms_per_1000_ci_low']
    text=f'''# Selective JEV assistance: frozen Study 2 policy replay

## Structured abstract

**Objective:** Estimate false alarms avoided per 1,000 posts and the paired sensitivity difference against the documented reference.

**Methods:** {ref['source_citation']}. Reference construct: {ref['construct_definition']}. The frozen expert-reference linkage and transfer evidence are retained with the protocol. A probability sample with common JEV outputs supports inverse-inclusion weighted selective, expected budget-matched random, tuned XGBoost and OR policy replay. Paired stratified bootstrap intervals condition on the observed finite frame, fixed policy and realized ratings.

**Results:** {status['actual_new_attempts']} actual live attempts; {status['successes']} successes, {status['failed']} failures and {status['pending_uncertain']} uncertain outcomes. Unavailable required responses retain the comparator. Selective minus comparator false-alarm reduction: {effect:.3f} per 1,000 (95% interval {low:.3f} to {high:.3f}). Sensitivity difference: {100*primary['delta_sensitivity']:+.3f} percentage points (95% interval {100*primary['delta_sensitivity_ci_low']:+.3f} to {100*primary['delta_sensitivity_ci_high']:+.3f}). Sensitivity decision: {status['sensitivity_preservation']}. Equal-cost random comparison and its paired interval appear in `study2_paired_comparisons.json`; no unweighted McNemar claim is substituted.

**Conclusions:** These estimates apply to {status['independence']}. A positive false-alarm reduction does not by itself establish preserved sensitivity, clinical validity or intervention benefit. The predeclared sensitivity rule and random-comparison intervals must determine the claim; nonsignificance alone does not.

## Methods, ethics and availability

See the dated frozen protocol, design, source-reference evidence and manuscript development draft history for source definitions, raters, sampling exclusions, fallbacks and limitations. Source approval is distinct from current-study ethics review. The documented external-transfer interpretation is `{frozen['transfer_evidence']['interpretation']}`; it is not an assertion of clinical deployment approval. Qualified new human coding remains pending unless separately supplied. No generated expert judgments are included. Public records omit raw text, author IDs, credentials and payloads; new provider costs remain unknown without verified usage/pricing.

This automatically generated results draft requires a source-informed manuscript review before submission. It does not invent source-study details absent from the verified evidence packet.
'''
    (r/'manuscript_report.md').write_text(text)
    return reviewer
