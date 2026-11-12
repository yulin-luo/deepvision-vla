"""Retrieve only the pinned twenty-file H1 bundle, then verify every byte."""
import argparse,hashlib,json,pathlib,re,shutil,subprocess,sys,time
from deepvision_vla.paths import ASSETS_ROOT,LICENSES_ROOT,require_source_checkout

def load_source():
    here=ASSETS_ROOT
    spec=json.loads((here/'h1_checkpoint_source.json').read_text())
    inventory=json.loads((here/spec['inventory']).read_text())
    if spec['schema_version']!=1 or spec['repo_type']!='model' or not re.fullmatch('[0-9a-f]{40}',spec['revision']):raise ValueError('unversioned checkpoint source')
    prefix=pathlib.PurePosixPath(spec['checkpoint_subdirectory'])
    if prefix.is_absolute() or '..' in prefix.parts or len(inventory)!=20:raise ValueError('invalid checkpoint inventory')
    for rel,item in inventory.items():
        p=pathlib.PurePosixPath(rel)
        if p.is_absolute() or '..' in p.parts or not re.fullmatch('[0-9a-f]{64}',item['sha256']) or item['size']<=0:raise ValueError('unsafe or unsealed checkpoint member')
    return here,spec,inventory

def plan():
    here,spec,inventory=load_source()
    return {'repo_type':spec['repo_type'],'repo_id':spec['repo_id'],'revision':spec['revision'],'checkpoint_subdirectory':spec['checkpoint_subdirectory'],
            'required_files':len(inventory),'required_bytes':sum(x['size'] for x in inventory.values()),
            'allow_patterns':[spec['checkpoint_subdirectory']+'/'+rel for rel in sorted(inventory)],
            'anonymous_access':spec['anonymous_access_observed_20261111'],'publishes_or_changes_access':False}

def download(output,cache):
    root=pathlib.Path(output).expanduser().resolve();cache=pathlib.Path(cache).expanduser().resolve()
    if root.exists():raise FileExistsError('fresh output root required; existing checkpoints are never replaced')
    here,spec,inventory=load_source();request=plan()
    from huggingface_hub import HfApi,snapshot_download
    try:info=HfApi().model_info(spec['repo_id'],revision=spec['revision'],token=False)
    except Exception as error:raise RuntimeError('Pinned public checkpoint is not reachable ('+type(error).__name__+'); check network access. No output files were created.') from None
    if info.sha!=spec['revision']:raise ValueError('Hugging Face revision differs from the pinned checkpoint')
    if not set(request['allow_patterns']).issubset({x.rfilename for x in info.siblings}):raise ValueError('pinned repository lacks required H1 files')
    root.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=True)
    record={'status':'downloading','request':request,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'authorization_tokens_recorded':False}
    receipt=root/'download_receipt.json';receipt.write_text(json.dumps(record,indent=2))
    try:
        snapshot_download(repo_id=spec['repo_id'],repo_type='model',revision=spec['revision'],allow_patterns=request['allow_patterns'],local_dir=str(root),cache_dir=str(cache),max_workers=4,token=False)
        checkpoint=root/spec['checkpoint_subdirectory']
        subprocess.run([sys.executable,'-m','deepvision_vla.tools.verify','--checkpoint',str(checkpoint),'--receipt',str(root/'checkpoint_verification.json')],check=True)
        # Sidecars stay outside the original twenty-file checkpoint directory.
        shutil.copy2(LICENSES_ROOT/'DINOv3_LICENSE.md',root/'DINOv3_LICENSE.md')
        shutil.copy2(require_source_checkout()/'LICENSE',root/'QWEN_APACHE2_LICENSE.txt')
        from deepvision_vla.runtime.checkpoint import checkpoint_binding
        binding=checkpoint_binding(checkpoint,root/'checkpoint_verification.json')
        record.update(status='downloaded_and_sha256_verified',checkpoint_root=str(checkpoint),required_files=20,
                      checkpoint_verification_receipt=binding['checkpoint_verification_receipt'],
                      checkpoint_verification_receipt_sha256=binding['checkpoint_verification_receipt_sha256'],
                      checkpoint_inventory_sha256=binding['checkpoint_inventory_sha256'])
    except BaseException as error:
        record.update(status='failed_partial_output_preserved',error_type=type(error).__name__);raise
    finally:
        record['ended_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());receipt.write_text(json.dumps(record,indent=2))
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root');p.add_argument('--cache');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if a.plan_only:print(json.dumps(plan(),indent=2));return
    if not a.output_root or not a.cache:p.error('--output-root and --cache are required for a real download')
    print(json.dumps(download(a.output_root,a.cache),indent=2))
if __name__=='__main__':main()
