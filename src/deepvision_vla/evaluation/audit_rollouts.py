"""Read native evidence only; never import a model/simulator or replay a request."""
import argparse,collections,hashlib,json,pathlib

INSTRUCTIONS={
 'close_box':'close box','close_fridge':'close fridge','close_laptop_lid':'close laptop lid',
 'phone_on_base':'put the phone on the base','place_wine_at_rack_location':'put the wine on the middle',
 'sweep_to_dustpan':'sweep the dirt up','take_frame_off_hanger':'take frame off hanger',
 'take_umbrella_out_of_umbrella_stand':'get the umbrella','toilet_seat_down':'toilet seat down','water_plants':'water plant',
}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(root,task,episode_ids):
    root=pathlib.Path(root);episode_ids=list(episode_ids);wanted=set(episode_ids)
    if task not in INSTRUCTIONS or len(wanted)!=len(episode_ids):raise ValueError('invalid fixed request set')
    files=[root/'launcher_receipt.json',root/'run.log',root/'results/predict_results'/task/'full32_h1/rollout_summary.json']
    issues=[]
    def read_json(path,label):
        if not path.is_file():return None
        try:
            value=json.loads(path.read_text())
            if not isinstance(value,dict):raise ValueError('expected object')
            return value
        except (OSError,ValueError):issues.append('unreadable_'+label);return None
    launcher=read_json(files[0],'launcher')
    log=files[1].read_text(errors='replace') if files[1].is_file() else ''
    summary=read_json(files[2],'summary')
    policy=[];forward=[];protocol=[]
    for line in log.splitlines():
        for marker,target in [('POLICY_REQUEST ',policy),('MODEL_FORWARD_ENTRY ',forward),('RESET_PROTOCOL ',protocol)]:
            if line.startswith(marker):
                try:target.append(json.loads(line[len(marker):]))
                except (ValueError,TypeError):issues.append('invalid_telemetry_json:'+marker.strip())
    for name,events in [('policy',policy),('forward',forward)]:
        for event in events:
            if event.get('episode_id') not in wanted:issues.append(name+':unknown_episode')
            if event.get('task')!=task:issues.append(name+':wrong_task')
            if event.get('instruction')!=INSTRUCTIONS[task]:issues.append(name+':wrong_instruction')
            if type(event.get('step')) is not int or not 0<=event['step']<10:issues.append(name+':invalid_step')
    if len(protocol)>1:issues.append('duplicate_reset_protocol')
    if protocol and (protocol[0].get('policy')!='single_seed_episode_identity_v1' or protocol[0].get('outer_attempts')!=1 or protocol[0].get('wrapper_attempts')!=1):issues.append('wrong_reset_protocol')
    if policy and not protocol:issues.append('missing_reset_protocol')
    records=summary.get('episode_records',[]) if summary else []
    counts=collections.Counter(row.get('episode_id') for row in records)
    if any(n>1 for n in counts.values()):issues.append('duplicate_episode_record')
    if set(counts)-wanted:issues.append('unrequested_episode_record')
    by_id={row['episode_id']:row for row in records if row.get('episode_id') in wanted}
    attempted={event.get('episode_id') for event in policy}&wanted
    forwarded={event.get('episode_id') for event in forward}&wanted
    if forwarded-attempted:issues.append('forward_without_policy_entry')
    if launcher and (launcher.get('task')!=task or launcher.get('requested_episodes')!=len(episode_ids)):issues.append('launcher_request_mismatch')
    if summary and (summary.get('task_name')!=task or summary.get('requested_episodes')!=len(episode_ids) or summary.get('eval_seed')!=42 or summary.get('max_steps')!=10):issues.append('summary_contract_mismatch')
    result_rows=[]
    for episode in episode_ids:
        row=by_id.get(episode);has_policy=episode in attempted;seed=42+episode*1009
        if row is None:status='outcome_missing'
        elif row.get('reset_status')=='failed':status='pre_policy_failure'
        elif has_policy:status='model_success' if row.get('success') is True else 'model_failure'
        else:status='pre_policy_failure'
        if row:
            if row.get('episode_seed')!=seed or row.get('reset_attempts')!=1 or row.get('reset_seeds_attempted')!=[seed]:issues.append(f'episode{episode}:reset_identity_mismatch')
            if row.get('reset_status')=='completed' and (row.get('actual_reset_seed')!=seed or row.get('reset_seed_trace')!=[seed]):issues.append(f'episode{episode}:actual_reset_mismatch')
            if row.get('reset_status')=='failed' and has_policy:issues.append(f'episode{episode}:policy_after_failed_reset')
            if row.get('success') is True and not has_policy:issues.append(f'episode{episode}:success_without_policy')
            if type(row.get('success')) is not bool:issues.append(f'episode{episode}:unknown_success_flag')
        result_rows.append({'episode_id':episode,'requested_seed':seed,'model_attempted':has_policy,'model_forward_observed':episode in forwarded,
                            'status':status,'native_record':row})
    success=sum(row.get('success') is True for row in by_id.values())
    if summary and summary.get('successes')!=success:issues.append('native_success_counter_mismatch')
    missing=sum(row['status']=='outcome_missing' for row in result_rows)
    prepolicy=sum(row['status']=='pre_policy_failure' for row in result_rows)
    model_fail=sum(row['status']=='model_failure' for row in result_rows)
    unresolved=len(attempted-set(by_id))
    terminal=launcher and launcher.get('status') in ['process_completed','process_failed','launcher_failed']
    complete=bool(terminal and launcher.get('returncode')==0 and summary and summary.get('status')=='COMPLETE' and missing==0 and not issues)
    return {'schema_version':1,'task':task,'status':'complete' if complete else 'invalid' if issues else 'incomplete' if root.exists() else 'not_started',
        'requested':len(episode_ids),'actual_attempts':len(attempted),'model_forward_episodes':len(forwarded),
        'success':success,'model_fail':model_fail,'pre_policy_fail':prepolicy,'missing':missing,'unresolved_model_attempts':unresolved,
        'requested_fail':len(by_id)-success,'success_rate_requested_pct':100*success/len(episode_ids) if episode_ids and by_id and not issues else None,
        'score_is_complete':complete,
        'policy_calls':len(policy),'model_forward_calls':len(forward),'instruction':INSTRUCTIONS[task],
        'failure_counters':{key:summary.get(key) if summary else None for key in ['infrastructure_error_count','policy_action_error_count','skipped_episodes','unattempted_episodes']},
        'launcher_status':launcher.get('status') if launcher else None,'launcher_returncode':launcher.get('returncode') if launcher else None,
        'reset_protocol':protocol[0] if len(protocol)==1 else None,'issues':issues,'episodes':result_rows,
        'evidence_sha256':{str(p.relative_to(root)):sha(p) for p in files if p.is_file()},
        'physical_state_identity_claim':False}

def aggregate(rows):
    names=['requested','actual_attempts','model_forward_episodes','success','model_fail','pre_policy_fail','missing','unresolved_model_attempts','requested_fail','policy_calls','model_forward_calls']
    result={key:sum(row[key] for row in rows) for key in names}
    result.update(status='complete' if rows and all(row['status']=='complete' for row in rows) else 'incomplete',tasks=rows,
                  success_rate_requested_pct=100*result['success']/result['requested'] if result['requested'] and result['success']+result['requested_fail'] else None,
                  score_is_complete=bool(rows and all(row['status']=='complete' for row in rows)),physical_state_identity_claim=False)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--task',choices=INSTRUCTIONS,required=True);p.add_argument('--episodes',type=int,default=50);p.add_argument('--receipt',required=True);a=p.parse_args()
    if not 1<=a.episodes<=50:p.error('episode budget must be1..50')
    result=audit(a.output_root,a.task,range(a.episodes))
    with pathlib.Path(a.receipt).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','evidence_sha256']},indent=2))
if __name__=='__main__':main()
