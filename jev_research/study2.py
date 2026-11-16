"""Study 2 design, weighted policy replay and historical development evidence.

No inference occurs here. A missing fresh sample is never replaced with reused
Study 1 data. Expert-reference, transfer permission and noninferiority claims
require documentary evidence external to the historical binary label exports.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

from .budgeted_experiments import (choose_threshold, digest, file_digest, metrics,
    paired_bootstrap, policy_output, save_json, save_table, split_cache, validate_cache)
from .client import BASE_MODEL, ENDPOINT, QUESTION, text_sha256
from .reporting import wilson_interval

STUDY = 'jev_selective_routing_study2_validation'
ATTEMPT_CEILING = 10_000
SEED = 42
PRIMARY_SOURCE = 'https://ieeexplore.ieee.org/document/9591887'
STRATA = ('outside_low', 'inside_below_comparator', 'inside_above_comparator', 'outside_high')


def load_historical(root):
    root = Path(root)
    frame = pd.read_json(root/'research/results/jev_probability_hybrid/predictions.json', precise_float=True)
    prior = pd.read_json(root/'research/results/jev_error_study/base_predictions.json', precise_float=True)
    validate_cache(frame)
    return frame, prior


def historical_checks(frame):
    _, dev, evaluation = split_cache(frame)
    rows = []
    for target in [.9, .95]:
        baseline = None
        for name, spec in [('XGBoost', {'kind': 'xgb'}), ('JEV replacement', {'kind': 'replacement'})]:
            choice = choose_threshold(dev, spec, target)
            score, pred, needs = policy_output(evaluation, spec, choice['threshold'])
            if baseline is None:
                baseline = pred
            row = {'evidence': 'Study 1 reused exploratory 3,000 text records; source post unit unverified',
                   'policy': name, 'development_target_recall': target,
                   'threshold': choice['threshold'], **metrics(evaluation.y_true, pred, score),
                   'prospective_requests': int(needs.sum())}
            if name == 'JEV replacement':
                ci = paired_bootstrap(evaluation.y_true, baseline, pred)
                for metric in ['recall', 'precision', 'specificity']:
                    row[f'delta_{metric}'] = ci[metric]['difference']
                    row[f'delta_{metric}_ci_low'], row[f'delta_{metric}_ci_high'] = ci[metric]['ci95']
            rows.append(row)
    return pd.DataFrame(rows)


def repeat_audit(frame, prior):
    repeat = frame.merge(prior, on='text_sha256', suffixes=('_probability', '_error'))
    rows = []
    for t in [.03, .08, .50]:
        a, b = repeat.jev_score.to_numpy() >= t, repeat.base_score.to_numpy() >= t
        count = int((a != b).sum())
        low, high = wilson_interval(count, len(repeat))
        rows.append({'threshold': t, 'repeated_text_n': len(repeat),
                     'different_score_n': int((repeat.jev_score-repeat.base_score).abs().gt(1e-12).sum()),
                     'different_binary_decision_n': count, 'different_binary_decision_rate': count/len(repeat),
                     'wilson95_low': low, 'wilson95_high': high,
                     'probability_run_positive_n': int(a.sum()), 'error_run_positive_n': int(b.sum()),
                     'probability_positive_error_negative_n': int((a & ~b).sum()),
                     'probability_negative_error_positive_n': int((~a & b).sum()),
                     'scope': '21 repeated, error-enriched texts; descriptive, not a population stability estimate'})
    return pd.DataFrame(rows)


def candidate_policy(frame):
    # All historical cases are now development, not a new held-out evaluation.
    baseline = choose_threshold(frame, {'kind': 'xgb'}, .95)
    tx = baseline['threshold']
    ok = frame.jev_status.eq('succeeded').to_numpy()
    x = frame.p_xgb.to_numpy()
    j = frame.jev_score.fillna(frame.p_xgb).to_numpy()
    y = frame.y_true.to_numpy()
    trials = []
    for t in np.unique(np.r_[0., j[ok], np.nextafter(1., 2.)]):
        pred = np.where(ok, j >= t, x >= tx)
        m = metrics(y, pred)
        if m['recall'] >= .95:
            trials.append((m['fp'], -m['precision'], -float(t), float(t)))
    if not trials:
        raise ValueError('Historical cache cannot support candidate recall target.')
    tj = min(trials)[-1]
    return {'status': 'conditional_candidate_not_a_frozen_evaluation_protocol',
            'comparator': 'threshold-tuned XGBoost', 'primary_assisted': 'selective replacement',
            'xgb_threshold': tx, 'jev_threshold': tj, 'routing_interval': [.25, .8],
            'calibration': 'none in primary policy; sigmoid/fusion are secondary',
            'fallback': 'retain comparator decision at frozen XGBoost threshold',
            'development': 'all 5,000 historical cache rows; supplied labels not verified expert reference',
            'development_target_recall': .95,
            'target_scope': 'historical routed score-range only; not a guaranteed global recall target',
            'model': BASE_MODEL, 'immutable_provider_revision': None, 'question': QUESTION,
            'question_sha256': digest(QUESTION), 'endpoint': ENDPOINT,
            'score_parser': 'answers.department.score; finite numeric, non-boolean, [0,1]',
            'truncation': 'no client truncation; provider context handling unknown'}


def provenance_table():
    fields = [
        ('Citation / DOI', PRIMARY_SOURCE, 'IEEE article title, authors and DOI not retrieved; owner supplied document URL'),
        ('Release', 'Kaggle nikhileswarkomati/suicide-watch version 14', 'Repository README and saved dataset fingerprint; primary release documentation blocked'),
        ('Dataset checksum', '1ec16b065a2fae36488de16e8abfc326586a576a25901e09467223e545ddd295', 'Saved provenance only; original CSV absent'),
        ('Language / platform / period', 'Not independently verified from primary source', 'Pipeline assumes English; no source collection-period documentation supplied'),
        ('Sampling / access / license', 'Not verified', 'Kaggle/IEEE access blocked; do not infer a license from public discoverability'),
        ('Annotation unit / rater context', 'Unknown: posts versus authors or post collections unresolved', 'Owner does not know the mapping or rating unit'),
        ('Construct / operational definition', 'Supplied binary suicide/non-suicide labels only', 'Expert risk, expressed intent, ideation and future suicide outcomes are not interchangeable'),
        ('Qualifications / raters per item', 'Not reported in available project artifacts', 'Primary paper unread; absence from exports does not establish absence from the source study'),
        ('Instructions / adjudication / agreement', 'Not ascertainable', 'No primary annotation protocol or supplementary material retrieved'),
        ('Coverage of specialist labels', 'Unverified', 'Cannot establish entire release, a subset or a different version'),
        ('Paper-to-CSV record linkage', 'Matched and unmatched counts NOT ESTIMABLE', 'No source record IDs, mapping table or source records supplied; not zero confirmed matches'),
        ('Repository transformations', 'Binary mapping; normalized raw deduplication/conflict removal; stratified splits; cleaned deduplication with training priority', 'Source code audited; raw CSV and original full split caches absent'),
        ('Context mismatch', 'Not assessable', 'If author-level labels are inherited by posts, a post classifier has a different target; this possibility is unresolved'),
        ('Ethics / consent / current review', 'Not verified for source study or current study', 'Source and current-study approvals must be documented separately'),
        ('External JEV transfer', 'Permission not established from data terms', 'No live transfer until compatible source terms and target are documented'),
    ]
    return pd.DataFrame(fields, columns=['provenance_item', 'finding', 'evidence_and_limit'])


def strata_for(scores, xgb_threshold):
    p = np.asarray(scores, float)
    if not .25 < xgb_threshold < .8 or not np.isfinite(p).all() or np.any((p < 0)|(p > 1)):
        raise ValueError('Invalid sampling threshold or scores.')
    return np.select([p < .25, (p >= .25)&(p < xgb_threshold),
                      (p >= xgb_threshold)&(p < .8)], STRATA[:3], default=STRATA[3])


def build_sample(test_frame, exclusions, protocol, *, requested_n, seed=SEED):
    """Label-blind stratified SRS by keyed hash; census the remaining route.

    Requires complete frame. Original test rows only; every requested stratum
    must have nonzero population/support. All training/dev and previous-study
    row, raw-text and, where supplied, normalized-text identities are excluded.
    """
    if isinstance(requested_n, bool) or not isinstance(requested_n, int) or not 4 <= requested_n <= ATTEMPT_CEILING:
        raise ValueError('Sample size must be 4..10,000 and justified before collection.')
    fields = ['row_id', 'text_sha256', 'p_xgb', 'source_split', 'dataset_sha256', 'baseline_fingerprint']
    if not set(fields).issubset(test_frame) or not test_frame.source_split.eq('test').all():
        raise ValueError('Require complete original test sampling frame with provenance.')
    if test_frame.row_id.duplicated().any() or test_frame.text_sha256.duplicated().any():
        raise ValueError('Duplicate sampling-frame identities.')
    pool = test_frame.copy()
    for column in ['row_id', 'text_sha256', 'cleaned_text', 'author_id']:
        if column in pool:
            seen = set()
            for excluded in exclusions.values():
                if column in excluded:
                    seen.update(excluded[column].dropna())
            if column=='author_id':
                seen={v for v in seen if str(v).strip()}
            pool = pool.loc[~pool[column].isin(seen)].copy()
    if len(pool) < requested_n:
        raise ValueError('Insufficient untouched frame; do not refill with historical examples.')
    if pool.dataset_sha256.nunique()!=1 or pool.baseline_fingerprint.nunique()!=1:
        raise ValueError('Ambiguous frame provenance.')
    pool['stratum'] = strata_for(pool.p_xgb, protocol['xgb_threshold'])
    populations = pool.stratum.value_counts().reindex(STRATA, fill_value=0).to_dict()
    if any(populations[s] < 2 for s in STRATA):
        raise ValueError('All four strata need >=2 records for population inference; revise upstream design explicitly.')
    # First allocate a census to difficult routing strata where affordable,
    # retaining >=2 sampled records in every exterior stratum.
    inside = STRATA[1:3]
    inside_goal = sum(populations[s] for s in inside)
    if inside_goal + 4 > requested_n:
        raise ValueError('Selected precision design cannot census the routing strata; revise protocol/sample size before collection.')
    quotas = {s: populations[s] for s in inside}
    outside = [STRATA[0], STRATA[3]]
    remainder = requested_n-inside_goal
    quota_low = max(2, min(populations[outside[0]], int(round(remainder*populations[outside[0]]/sum(populations[s] for s in outside)))))
    quota_high = remainder-quota_low
    if quota_high > populations[outside[1]]:
        quota_low += quota_high-populations[outside[1]]; quota_high=populations[outside[1]]
    if quota_high < 2:
        quota_low -= 2-quota_high; quota_high=2
    quotas.update({outside[0]: quota_low, outside[1]: quota_high})
    selected = []
    for s in STRATA:
        candidates = pool.loc[pool.stratum.eq(s), fields].copy()  # discard labels before ranking
        candidates['sample_id'] = [text_sha256(f'{d}:{int(r)}:{h}') for d,r,h in zip(candidates.dataset_sha256,candidates.row_id,candidates.text_sha256)]
        candidates['priority'] = candidates.sample_id.map(lambda i: digest([seed,'study2',s,i]))
        part=candidates.sort_values(['priority','sample_id']).head(quotas[s]).copy()
        part['stratum']=s; part['population_n']=populations[s]; part['sample_n']=quotas[s]
        part['inclusion_probability']=quotas[s]/populations[s]; part['study_partition']='evaluation'
        selected.append(part)
    manifest=pd.concat(selected,ignore_index=True)
    assert len(manifest)==requested_n and manifest.sample_id.nunique()==requested_n
    metadata={'population_n':len(pool),'strata':[{'stratum':s,'population_n':populations[s],'sample_n':quotas[s],'inclusion_probability':quotas[s]/populations[s]} for s in STRATA],
              'population_routing_fraction':inside_goal/len(pool),'design':'census of residual route + score-stratified outside SRS',
              'independence':'unqueried original test posts; test labels previously used in aggregate; author/time separation only where metadata permits'}
    return manifest,metadata


def policy_arrays(frame, protocol, population_route_fraction):
    """Expected random-routing decisions at the same POPULATION request budget.

    Failure fallback is the frozen tuned comparator. For random routing,
    confusion totals are expectations over a uniform fixed-size random referral
    set. Ratios use expected totals, not the mean of randomized ratios.
    """
    x=frame.p_xgb.to_numpy(float); ok=frame.jev_status.eq('succeeded').to_numpy()
    if not np.isfinite(x).all() or np.any((x<0)|(x>1)) or not frame.jev_status.isin(['succeeded','failed','pending','not_attempted']).all():
        raise ValueError('Invalid source score or response status.')
    succeeded=frame.loc[ok,'jev_score'].to_numpy(float)
    if not np.isfinite(succeeded).all() or np.any((succeeded<0)|(succeeded>1)) or frame.loc[~ok,'jev_score'].notna().any():
        raise ValueError('Contradictory or invalid source JEV score.')
    j=frame.jev_score.fillna(frame.p_xgb).to_numpy(float)
    b=(x>=protocol['xgb_threshold']).astype(float)
    g=(j>=protocol['jev_threshold']).astype(float)
    lo,hi=protocol['routing_interval']; route=(x>=lo)&(x<hi)
    r=float(population_route_fraction)
    if not 0<=r<=1:
        raise ValueError('Invalid target-population random-routing budget.')
    use=route & ok
    selective=np.where(use,g,b)
    random=b+r*ok*(g-b)
    need_or=route & (b==0)
    one_sided=np.maximum(b, np.where(need_or&ok,g,0))
    return {'XGBoost tuned':(b,np.zeros(len(x))),
            'Selective JEV':(selective,route.astype(float)),
            'Budget-matched random':(random,np.full(len(x),r)),
            'One-sided OR':(one_sided,need_or.astype(float))}


def contributions(y,q,requests):
    y=np.asarray(y,float); q=np.asarray(q,float)
    if not np.isin(y,[0,1]).all() or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
        raise ValueError('Invalid labels/expected decisions.')
    return np.column_stack([(1-y)*(1-q),(1-y)*q,y*(1-q),y*q,requests])


def summarize_totals(totals):
    tn,fp,fn,tp,req=np.moveaxis(np.asarray(totals,float),-1,0)
    n=tn+fp+fn+tp
    def div(a,b):
        return np.divide(a,b,out=np.full_like(np.asarray(a,float),np.nan),where=np.asarray(b)!=0)
    return {'sensitivity':div(tp,tp+fn),'precision':div(tp,tp+fp),'specificity':div(tn,tn+fp),
            'prevalence':div(tp+fn,n),'missed_per_1000':1000*div(fn,n),
            'false_alarms_per_1000':1000*div(fp,n),'flags_per_1000':1000*div(tp+fp,n),
            'requests_per_1000':1000*div(req,n),'referred_fraction':div(req,n)}


def design_totals(frame, values, *, replicates=2000, seed=SEED):
    """Paired stratified bootstrap with finite-frame correction and fixed N_h.

    Each contribution vector uses exactly the same sampled row indexes for all
    policies. sqrt(1-n_h/N_h) perturbations make a census stratum's finite-frame
    sampling variance zero. No superpopulation/author-time generalization is
    supplied by this correction. Clusters require a different design.
    """
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    estimate=np.zeros(values.shape[1]); draws=np.zeros((replicates,values.shape[1]))
    for s,indices in frame.groupby('stratum').indices.items():
        indices=np.asarray(indices); rows=frame.iloc[indices]
        if rows.population_n.nunique()!=1 or rows.sample_n.nunique()!=1:
            raise ValueError('Stratum design metadata inconsistent.')
        n,N=len(rows),int(rows.population_n.iloc[0])
        if n!=int(rows.sample_n.iloc[0]) or not 2<=n<=N:
            raise ValueError('Require a completed probability sample with >=2 observations per stratum.')
        if not np.allclose(rows.inclusion_probability,n/N,rtol=0,atol=1e-14):
            raise ValueError('Inclusion probabilities do not match the sampling design.')
        data=values[indices]; mean=data.mean(axis=0); estimate+=N*mean
        fpc=np.sqrt(1-n/N)
        for start in range(0,replicates,50):
            k=min(50,replicates-start)
            sampled=data[rng.integers(0,n,size=(k,n))].mean(axis=1)
            draws[start:start+k]+=N*(mean+fpc*(sampled-mean))
    return estimate,draws


def analyze_probability_sample(frame, protocol, design, *, replicates=2000):
    if 'author_id' in frame:
        authors=frame.author_id.dropna().astype(str).str.strip()
        if authors.loc[authors.ne('')].duplicated().any():
            raise ValueError('Repeated authors require a prespecified cluster sampling/variance design; row bootstrap refused.')
    if frame.sample_id.duplicated().any() or frame.text_sha256.duplicated().any():
        raise ValueError('Evaluation rows must be unique.')
    inferred_N=sum(int(group.population_n.iloc[0]) for _,group in frame.groupby('stratum'))
    if inferred_N!=design['population_n'] or set(frame.stratum)!=set(STRATA):
        raise ValueError('A full-population claim requires all frozen score strata and the correct frame denominator.')
    arrays=policy_arrays(frame,protocol,design['population_routing_fraction'])
    names=list(arrays); y=frame.y_true.to_numpy()
    matrix=np.column_stack([contributions(y,*arrays[name]) for name in names])
    total,draw=design_totals(frame,matrix,replicates=replicates)
    rows=[]; summaries={}
    for i,name in enumerate(names):
        t=total[5*i:5*i+5]; d=draw[:,5*i:5*i+5]
        m,boot=summarize_totals(t),summarize_totals(d); summaries[name]=(m,boot)
        row={'policy':name,'sampled_posts_n':len(frame),'target_frame_n':design['population_n'],
             'observed_successful_responses_n':int(frame.jev_status.eq('succeeded').sum()),
             'evidence':'weighted finite-frame policy replay; random row is expected-confusion-total result'}
        for metric,value in m.items():
            row[metric]=float(value)
            row[metric+'_ci_low'],row[metric+'_ci_high']=np.quantile(boot[metric],[.025,.975]).tolist()
        rows.append(row)
    paired=[]
    for name in names[1:]:
        m,boot=summaries[name]
        for ref in ['XGBoost tuned','Budget-matched random']:
            if name==ref: continue
            b,bs=summaries[ref]
            row={'policy':name,'reference':ref}
            for metric in ['sensitivity','precision','false_alarms_per_1000','missed_per_1000','flags_per_1000','requests_per_1000']:
                row['delta_'+metric]=float(m[metric]-b[metric])
                row['delta_'+metric+'_ci_low'],row['delta_'+metric+'_ci_high']=np.quantile(boot[metric]-bs[metric],[.025,.975]).tolist()
            req=float(m['requests_per_1000'])
            row['false_alarms_avoided_per_100_requests']=(float(b['false_alarms_per_1000']-m['false_alarms_per_1000'])/req*100) if req else None
            row['additional_positive_detections_per_100_requests']=(float(b['missed_per_1000']-m['missed_per_1000'])/req*100) if req else None
            paired.append(row)
    transition_rows=[]; base=arrays['XGBoost tuned'][0]
    weights=1/frame.inclusion_probability.to_numpy(); N=design['population_n']
    for name,(q,req) in arrays.items():
        vals={'recovered_false_negatives':y*(1-base)*q,'introduced_false_negatives':y*base*(1-q),
              'corrected_false_positives':(1-y)*base*(1-q),'introduced_false_positives':(1-y)*(1-base)*q}
        transition_rows.append({'policy':name,**{k+'_per_1000':float(np.sum(weights*v)/N*1000) for k,v in vals.items()}})
    return pd.DataFrame(rows),pd.DataFrame(paired),pd.DataFrame(transition_rows)


def sensitivity_decision(lower_bound, margin=None, justification=None):
    if margin is None:
        return {'status':'not_adjudicated_no_defensible_margin','lower_one_sided_97_5_bound':lower_bound,
                'preservation_established':False}
    if not justification or not 0<margin<1:
        raise ValueError('A defensible external justification and absolute recall margin are required.')
    return {'status':'criterion_met' if lower_bound>-margin else 'criterion_not_met',
            'margin':margin,'justification':justification,'lower_one_sided_97_5_bound':lower_bound,
            'preservation_established':lower_bound>-margin}


def fit_local_text_baseline(train,development):
    """Optional strong sparse baseline, training-only features; no test input.

    This is not called on missing real data or on synthetic text as a substitute
    for empirical results. The returned model stays in private artifacts.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion,Pipeline
    from sklearn.linear_model import LogisticRegression
    for column in ['row_id','text_sha256','cleaned_text','author_id']:
        if column in train and column in development:
            if set(train[column].dropna())&set(development[column].dropna()):
                raise ValueError('Local baseline training/development overlap: '+column)
    if set(train.y_true.unique())!={0,1} or set(development.y_true.unique())!={0,1}:
        raise ValueError('Both upstream classes are required.')
    features=FeatureUnion([
        ('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50_000,sublinear_tf=True)),
        ('character',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=100_000,sublinear_tf=True))])
    model=Pipeline([('features',features),('classifier',LogisticRegression(C=1.,solver='liblinear',max_iter=1000,random_state=SEED))])
    model.fit(train.text,train.y_true)
    probabilities=model.predict_proba(development.text)[:,1]
    dev=pd.DataFrame({'p_xgb':probabilities,'y_true':development.y_true.to_numpy(),
                      'jev_status':'not_attempted','jev_score':np.nan})
    threshold=choose_threshold(dev,{'kind':'xgb'},.95)
    return model,{'status':'fitted upstream only','threshold':threshold,'features':'training-only word 1-2 and character 3-5 TF-IDF',
                  'C':1.,'seed':SEED,'train_ids_sha256':digest(sorted(train.text_sha256)),
                  'development_ids_sha256':digest(sorted(development.text_sha256)),
                  'evaluation_results':'not computed by fitting helper'}


def precision_planning(frame,candidate,prior_manifest):
    # Historical eligible count and full-test denominator are documented, but
    # actual fresh strata and their labels are not reconstructed from aggregates.
    extras=prior_manifest.loc[prior_manifest.source_split.eq('test') & ~prior_manifest.text_sha256.isin(frame.text_sha256)]
    extra_inside=int(((extras.p_xgb>=.25)&(extras.p_xgb<.8)).sum())
    remaining_N=46342-5000-len(extras)
    remaining_inside=6067-5000-extra_inside
    y=frame.y_true.to_numpy(); b=(frame.p_xgb.to_numpy()>=candidate['xgb_threshold'])
    j=frame.jev_score.fillna(frame.p_xgb).to_numpy(); ok=frame.jev_status.eq('succeeded').to_numpy()
    a=np.where(ok,j>=candidate['jev_threshold'],b)
    pos=y==1
    d=(a.astype(int)-b.astype(int))[pos]
    neg=(y==0)
    fp_diff=(b.astype(int)-a.astype(int))[neg]
    p=pos.mean(); disagreement=float(np.mean(d!=0)); delta=float(d.mean())
    rows=[]
    for n in [2000,5000,8000,10000]:
        for deff in [1.,1.5,2.,4.]:
            npos=n*p
            se=np.sqrt(max(0,disagreement-delta**2)/npos*deff)
            fp_se=np.sqrt(max(0,float(np.mean(fp_diff**2)-np.mean(fp_diff)**2))/(n*(1-p))*deff)
            rows.append({'planned_common_sample_n':n,'assumed_design_effect':deff,
                         'assumed_positive_fraction':float(p),'assumed_positive_disagreement':disagreement,
                         'approx_sensitivity_halfwidth_95_pp':float(norm.ppf(.975)*se*100),
                         'approx_specificity_halfwidth_95_pp':float(norm.ppf(.975)*fp_se*100),
                         'scope':'illustrative routed-population iid approximation; not a fresh full-population power calculation'})
    details={'known_original_test_n':46342,'known_original_eligible_n':6067,
             'previous_probability_cohort_n':5000,'additional_previous_error_test_n':len(extras),
             'additional_previous_error_test_inside_interval_n':extra_inside,
             'inferred_remaining_test_n_before_other_independence_exclusions':remaining_N,
             'inferred_remaining_eligible_n_before_other_exclusions':remaining_inside,
             'inferred_remaining_routing_fraction':remaining_inside/remaining_N,
             'warning':'Excluding the historical cache changes the population route fraction drastically. These aggregate counts do not supply row IDs, labels or final author/near-duplicate exclusions.',
             'allocation':'Historical development; zero planned paid repeats; census residual route if feasible, probability sample both exterior strata. Final n<=10000 selected only after complete frame and design-aware precision review.',
             'n_status':'no final quota or sample size frozen while outside-stratum disagreements/frame are unknown',
             'outside_score_regions':'No JEV evidence. Random-routing contrast uncertainty outside coverage cannot be estimated from cached inside-range outcomes.',
             'repeated_author_effect':'Design-effect scenarios illustrate loss of precision; not measured intraclass correlation.',
             'unused_authorization':ATTEMPT_CEILING}
    return pd.DataFrame(rows),details


def create_protocol(root,candidate,planning):
    now=datetime.now(timezone.utc)
    return {'study_id':STUDY,'created_at_utc':now.isoformat(),'created_at_tehran':now.astimezone(ZoneInfo('Asia/Tehran')).isoformat(),
            'protocol_schema_version':1,'protocol_revision':1,'primary_source':PRIMARY_SOURCE,
            'status':'DATED_INTERNAL_DESIGN_DRAFT_NOT_FROZEN_FOR_COLLECTION',
            'registration':'No public preregistration claimed',
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'question':'Does selective JEV reduce false-positive flags per 1000 posts relative to tuned XGBoost, with an estimated paired sensitivity difference against a verified compatible reference?',
            'annotation_unit':'pending primary-source verification; intended model input is one post',
            'positive_construct':'pending source definition and paper-to-release record mapping; current cache is supplied suicide/non-suicide labels',
            'population':'candidate: unqueried original test posts after training/development/history/duplicate exclusions; aggregate test exposure disclosed',
            'primary_effect':'comparator false alarms per 1000 minus selective false alarms per 1000; positive favors assistance',
            'primary_sensitivity':'assisted minus comparator sensitivity with paired design-appropriate interval; no preservation claim without justified margin',
            'noninferiority_margin':None,'margin_justification':None,
            'illustrative_margins_secondary_only':[.01,.02,.05],
            'candidate_policy':candidate,'seed':SEED,'budget_ceiling':ATTEMPT_CEILING,
            'owner_live_authorization':'Already granted up to 10000 additional attempts; no repeat financial approval needed',
            'routine_notebooks':'cache-only; separate resumable collection command',
            'ledger':'artifacts/'+STUDY+'/budget.sqlite3',
            'sampling':planning,'evaluation_manifest_sha256':None,'design_sha256':None,
            'inference':'paired stratified bootstrap of weighted contribution totals; finite-frame correction for census strata; expected random-routing outcome at exact population referral fraction; no ordinary McNemar for weighted/random expectations',
            'author_rule':'exclude prior authors if identifiers exist; repeated sampled authors require a cluster design before analysis',
            'secondary':['budget-matched random referral using identical JEV cutoff/fallback','one-sided OR','local word/character TF-IDF logistic baseline if raw training/development texts available','sigmoid calibration and fusion as supplementary only'],
            'human_review':'separate probability and enriched disagreement samples; blinded descriptive coding; no LLM-generated expert labels',
            'gates':{'primary_source_and_construct_verified':False,'expert_record_mapping_verified':False,
                     'external_transfer_terms_permit':False,'complete_sampling_frame_verified':False,
                     'frozen_manifest_and_protocol':False,'live_credential_available':False},
            'new_live_attempts':0,'study2_results':'NOT COLLECTED'}


def prepare_human_review(frame,root,results):
    """Independent samples; private key keeps model/label information off forms."""
    root,results=Path(root),Path(results)
    private=root/'artifacts'/STUDY/'human_review';private.mkdir(parents=True,exist_ok=True)
    probability=frame.copy()
    probability['priority']=probability.sample_id.map(lambda i:digest([SEED,'label-fidelity-probability',i]))
    probability=probability.sort_values('priority').head(100).copy()
    probability['sample_kind']='probability';probability['inclusion_probability']=100/len(frame)
    y=frame.y_true.to_numpy();b=frame.p_xgb.to_numpy()>=.5
    g=np.where(frame.jev_status.eq('succeeded'),frame.jev_score.fillna(frame.p_xgb).to_numpy()>=.5,b)
    groups={'recovered_positive':(y==1)&~b&g,'lost_positive':(y==1)&b&~g,
            'corrected_false_alarm':(y==0)&b&~g,'new_false_alarm':(y==0)&~b&g}
    enriched=[]
    for name,mask in groups.items():
        part=frame.loc[mask].copy();part['priority']=part.sample_id.map(lambda i:digest([SEED,'enriched-review',name,i]))
        part=part.sort_values('priority').head(10).copy();part['sample_kind']='enriched';part['transition']=name
        part['inclusion_probability']=np.nan;enriched.append(part)
    selected=pd.concat([probability,*enriched],ignore_index=True)
    # Keep stable local blind IDs. Salt is not exported to the public package.
    salt_path=private/'blind_salt'
    if not salt_path.exists():
        import secrets
        salt_path.write_text(secrets.token_hex(32));salt_path.chmod(0o600)
    salt=salt_path.read_text()
    selected['blind_id']=[hashlib.sha256(f'{salt}:{kind}:{sid}'.encode()).hexdigest()[:20]
                          for kind,sid in zip(selected.sample_kind,selected.sample_id)]
    private_key=selected[['blind_id','sample_id','row_id','text_sha256','sample_kind','y_true','p_xgb','jev_score','transition']].copy()
    private_key.to_json(private/'private_key.json',orient='records',indent=2)
    (private/'private_key.json').chmod(0o600)
    forms=pd.DataFrame({'blind_id':selected.blind_id,'source_text':'PENDING: matching private text required',
                        'first_or_third_person':'','present_or_past':'','ideation_or_intent':'',
                        'quotation':'','negation':'','help_seeking':'','insufficient_context':'',
                        'source_construct_code':'','insufficient_evidence':'','reviewer_id':''})
    forms.to_json(private/'blinded_forms.json',orient='records',indent=2)
    public=selected[['sample_id','row_id','text_sha256','sample_kind','inclusion_probability']].copy()
    public['scope']='historical Study 1 only; not a fresh Study 2 sample'
    save_table(results,'human_review_sampling_manifest',public)
    return {'status':'pending source construct, matching text and qualified reviewers','probability_n':len(probability),
            'probability_scope':'historical 5000-case cache only','enriched_n':sum(map(len,enriched)),
            'overlap_between_samples_n':len(set(probability.sample_id)&set(pd.concat(enriched).sample_id)),
            'blinding':'Forms exclude model identity, labels, scores and transition categories; private key is ignored by Git',
            'coding':'two independent qualified reviewers, blinded coding, adjudication after independent forms; Cohen kappa for nominal codes and weighted kappa for justified ordinal codes; report agreement denominators and uncertain cases',
            'paper_grounding':'pending primary paper; proposed linguistic fields are provisional, not verified source definitions',
            'new_expert_labels_created':0}


def run_offline(root):
    root=Path(root).resolve();results=root/'research/results'/STUDY;results.mkdir(parents=True,exist_ok=True)
    frame,prior=load_historical(root)
    checks=historical_checks(frame);repeat=repeat_audit(frame,prior);candidate=candidate_policy(frame)
    previous=pd.read_json(root/'research/results/jev_error_study/selection_manifest.json',precise_float=True)
    planning,assumptions=precision_planning(frame,candidate,previous)
    protocol_path=results/'protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        if protocol.get('candidate_policy')!=candidate:
            raise ValueError('Existing protocol candidate differs; version the design explicitly.')
        if 'protocol_schema_version' not in protocol:
            protocol.update({'protocol_schema_version':1,'protocol_revision':1,'primary_source':PRIMARY_SOURCE})
            save_json(protocol_path,protocol)
    else:
        protocol=create_protocol(root,candidate,assumptions);save_json(protocol_path,protocol)
    save_json(results/'protocol_digest.json',{'sha256':digest(protocol),'status':protocol['status'],
                                            'evaluation_manifest_sha256':None,'public_preregistration':False})
    review=prepare_human_review(frame,root,results)
    tables={'historical_reference_checks':checks,'historical_repeat_audit':repeat,
            'provenance':provenance_table(),'illustrative_precision_planning':planning}
    for name,table in tables.items():save_table(results,name,table)
    save_json(results/'planning_assumptions.json',assumptions);save_json(results/'human_review_status.json',review)
    pending=pd.DataFrame([{'policy':name,'Study 2 population estimates':'NOT COLLECTED',
                           'fresh_sample_n':None,'false_alarms_per_1000':None,'sensitivity':None,
                           'actual_new_attempts':0,'reason':'source/reference mapping, transfer terms, raw frame and live credential unavailable'}
                          for name in ['XGBoost tuned','Selective JEV','Budget-matched random','One-sided OR','Strong local text baseline']])
    save_table(results,'study2_main_population_results',pending)
    save_table(results,'study2_routing_stratum_results',pd.DataFrame([{'stratum':s,'Study 2 result':'NOT COLLECTED','observed_count':None} for s in STRATA]))
    return {'root':root,'results':results,'frame':frame,'tables':tables,'candidate':candidate,
            'protocol':protocol,'planning':assumptions,'review':review,'pending_results':pending}
