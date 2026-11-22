"""Paired episode statistics for a predeclared diagnostic panel.

No model/data loading, output selection, plotting or file writes happen here.
The caller supplies one explicitly qualified source cohort and frozen points.
An absent/nonfinite metric is missing evidence, never a zero-valued result.
"""
import math

ANALYSIS_SEED=20261114
BOOTSTRAP_RESAMPLES=2000


def _finite_error(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def paired_bootstrap(matrix,*,resamples=BOOTSTRAP_RESAMPLES,seed=ANALYSIS_SEED):
    """Resample whole source rows, sharing one draw across every curve point."""
    import numpy as np
    values=np.asarray(matrix,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]==0 or not np.isfinite(values).all():
        raise ValueError('A finite source-by-point matrix is required')
    if type(resamples)is not int or resamples<2 or type(seed)is not int:
        raise ValueError('Explicit finite bootstrap count and analysis seed required')
    n,p=values.shape
    if n==0:
        return {'n_sources':0,'means':[None]*p,'lower':[None]*p,'upper':[None]*p,
            'uncertainty_status':'no_paired_sources'}
    means=values.mean(axis=0).tolist()
    if n==1:
        return {'n_sources':1,'means':means,'lower':[None]*p,'upper':[None]*p,
            'uncertainty_status':'one_source_no_bootstrap_interval'}
    generator=np.random.default_rng(seed)
    # A single episode draw is reused across all points. Batching bounds memory
    # without treating layer/ratio aliases as additional experimental units.
    draws=np.empty((resamples,p),dtype=np.float64)
    for first in range(0,resamples,64):
        stop=min(first+64,resamples)
        indexes=generator.integers(0,n,size=(stop-first,n))
        draws[first:stop]=values[indexes].mean(axis=1)
    limits=np.quantile(draws,[.025,.975],axis=0,method='linear')
    return {'n_sources':n,'means':means,'lower':limits[0].tolist(),'upper':limits[1].tolist(),
        'uncertainty_status':'paired_episode_percentile_95',
        'interval_scope':'pointwise_percentile_95_not_a_simultaneous_curve_band',
        'resamples':resamples,'analysis_seed':seed,
        'inference_scope':'conditional on this fixed filtered source panel; not a uniformly sampled Bridge population'}


def build_paired_panel(records,requested_source_ids,qualified_source_ids,points):
    """Keep all coverage; form a common complete metric cohort without padding.

    records[source] = {'baseline': nonnegative_error,
                       'points': {frozen_point: nonnegative_error}}
    A source with any missing point stays in the exclusion ledger. Its remaining
    points are not silently used to change the curve's cohort from point to point.
    """
    requested=list(requested_source_ids);qualified=list(qualified_source_ids);points=list(points)
    if (not requested or len(requested)!=len(set(requested))or not points
            or len(points)!=len(set(points))or len(qualified)!=len(set(qualified))
            or not set(qualified)<=set(requested)):
        raise ValueError('Unique original source identities and frozen points required')
    if not isinstance(records,dict)or not set(records)<=set(qualified):
        raise ValueError('Unrequested or unqualified source outputs cannot enter the panel')
    qualified_set=set(qualified)
    if qualified!=[s for s in requested if s in qualified_set]:
        raise ValueError('Qualified sources must preserve original requested order')
    included=[];excluded=[];absolute=[];deltas=[];baselines=[]
    for source in qualified:
        row=records.get(source);reasons=[]
        if not isinstance(row,dict):
            excluded.append({'sample_id':source,'reason':'missing_source_metric_record'});continue
        baseline=row.get('baseline')
        if not _finite_error(baseline):reasons.append({'point':'baseline','reason':'missing_or_invalid_nonnegative_metric'})
        measured=row.get('points')
        if not isinstance(measured,dict):measured={}
        if set(measured)-set(points):raise ValueError('Unexpected point outside the frozen panel')
        for point in points:
            if not _finite_error(measured.get(point)):
                reasons.append({'point':point,'reason':'missing_or_invalid_nonnegative_metric'})
        if reasons:
            excluded.append({'sample_id':source,'reason':'incomplete_metric_panel','details':reasons});continue
        values=[float(measured[point])for point in points]
        included.append(source);baselines.append(float(baseline));absolute.append(values)
        deltas.append([value-float(baseline)for value in values])
    return {'requested_source_ids':requested,'qualified_source_ids':qualified,'paired_source_ids':included,
        'annotation_or_protocol_blocked_source_ids':[s for s in requested if s not in qualified_set],
        'excluded_qualified_sources':excluded,'points':points,
        'requested_sources':len(requested),'qualified_sources':len(qualified),'paired_sources':len(included),
        'all_qualified_sources_metric_complete':bool(qualified)and len(included)==len(qualified),
        'absolute_values':absolute,'signed_baseline_differences':deltas,'baseline_values':baselines,
        'experimental_unit':'source_episode','aliases_layers_and_ratios_are_new_replicates':False,
        'negative_error_differences_preserved':True,'missing_metrics_replaced_with_zero':False}


def summarize_paired_panel(panel):
    """No intervals/means from unequal point-specific source sets."""
    import numpy as np
    points=panel['points'];n=panel['paired_sources']
    absolute=np.asarray(panel['absolute_values'],dtype=np.float64).reshape(n,len(points))
    differences=np.asarray(panel['signed_baseline_differences'],dtype=np.float64).reshape(n,len(points))
    return {'points':points,'requested_sources':panel['requested_sources'],
        'qualified_sources':panel['qualified_sources'],'paired_sources':n,
        'all_qualified_sources_metric_complete':panel['all_qualified_sources_metric_complete'],
        'status':'complete_metric_panel'if n and panel['all_qualified_sources_metric_complete']else'incomplete_metric_coverage',
        'absolute_error':paired_bootstrap(absolute),'signed_baseline_difference':paired_bootstrap(differences),
        'requested_source_ids':panel['requested_source_ids'],'qualified_source_ids':panel['qualified_source_ids'],
        'annotation_or_protocol_blocked_source_ids':panel['annotation_or_protocol_blocked_source_ids'],
        'source_ids':panel['paired_source_ids'],'excluded_qualified_sources':panel['excluded_qualified_sources'],
        'source_weighting':'equal episode weight; no layer/ratio/alias replication'}


def summarize_point_differences(records, source_ids, point_pairs):
    """Subtract registered point pairs within a source before resampling."""
    import numpy as np

    pairs = list(point_pairs)
    values = [
        [records[sample_id]["points"][left] - records[sample_id]["points"][right]
         for left, right in pairs]
        for sample_id in source_ids
    ]
    matrix = np.asarray(values, dtype=np.float64).reshape(len(source_ids), len(pairs))
    return paired_bootstrap(matrix)
