"""Explicit strict reset adapter; model, observation and action functions are unchanged."""
import ast,hashlib,json
POLICY='single_seed_episode_identity_v1'
def transform_evaluator(source):
    tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    retry=[];seeds=[]
    for n in ast.walk(main):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAX_RESET_RETRIES' for t in n.targets):retry.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ep_seed' for t in n.targets):seeds.append(n)
    if len(retry)!=1 or len(seeds)!=1 or not isinstance(retry[0].value,ast.Constant) or retry[0].value.value!=5:raise ValueError('reset adapter source shape changed')
    original=ast.parse('None if args.eval_seed < 0 else args.eval_seed + i * 1009',mode='eval').body
    if ast.dump(seeds[0].value)!=ast.dump(original):raise ValueError('reference seed formula changed')
    retry[0].value=ast.Constant(1)
    seeds[0].value=ast.parse('None if args.eval_seed < 0 else args.eval_seed + int(episode_id) * 1009',mode='eval').body
    ast.fix_missing_locations(tree)
    return tree
def single_reset(self,seed=None):
    self.cur_step=0
    self.env.reset(seed=seed)
    obs=self.get_obs();obs['actual_reset_seed']=seed;obs['reset_seed_trace']=[seed]
    if self.cinematic_flag:self.tr.take_snap(obs=None)
    return obs
def install_reset_adapter():
    from lift3d.envs.rlbench_env import RLBenchEnv
    RLBenchEnv.reset=single_reset
def execution_fingerprint(tree):
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
