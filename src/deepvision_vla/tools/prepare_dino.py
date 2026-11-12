"""Create a new local config/processor view over the saved fine-tuned DINO."""
import argparse,pathlib,shutil,os,json
from deepvision_vla.paths import ASSETS_ROOT
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    checkpoint=pathlib.Path(a.checkpoint).expanduser().resolve(strict=True)
    weight=checkpoint/'dinov3_model/pytorch_model.bin'
    if not weight.is_file() or weight.stat().st_size!=1681376319:raise ValueError('unexpected saved DINO artifact; run full checkpoint verification first')
    out=pathlib.Path(a.output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    assets=ASSETS_ROOT/'dino_runtime'
    for name in ['config.json','preprocessor_config.json']:shutil.copy2(assets/name,out/name)
    os.symlink(weight,out/'pytorch_model.bin')
    print(json.dumps({'runtime_view':str(out),'uses_saved_finetuned_dino':True,'additional_foundation_weights_required':False,'checkpoint_modified':False}))
if __name__=='__main__':main()
