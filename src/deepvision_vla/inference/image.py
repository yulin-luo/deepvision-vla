"""Label-free B1 input and dispatch, preserving scientific model/codec functions."""
import ast,hashlib,json,math,pathlib

def file_sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def read_request(path):
    path=pathlib.Path(path).expanduser().resolve(strict=True)
    data=json.loads(path.read_text())
    if not isinstance(data,dict):raise ValueError('request must be a JSON object')
    if set(data)-{'schema_version','image','instruction','current_ee_position_xyz'}:raise ValueError('unknown request fields; labels/GT actions are not inputs')
    if data.get('schema_version')!=1:raise ValueError('unsupported request schema')
    instruction=data.get('instruction')
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('explicit nonempty instruction required')
    if any(token in instruction for token in ['<|im_start|>','<|im_end|>','<|vision_start|>','<|vision_end|>','<|image_pad|>','<action_']):raise ValueError('instruction contains reserved model tokens')
    image=data.get('image')
    if not isinstance(image,str) or not image:raise ValueError('image path required')
    image=pathlib.Path(image).expanduser()
    image=(path.parent/image).resolve(strict=True) if not image.is_absolute() else image.resolve(strict=True)
    if not image.is_file():raise ValueError('image must be a file')
    position=data.get('current_ee_position_xyz')
    if position is not None and (not isinstance(position,list) or len(position)!=3 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in position)):raise ValueError('measured world-EEF position must have three finite numbers')
    return {'schema_version':1,'image':str(image),'instruction':instruction,'current_ee_position_xyz':position,'request_path':str(path)}

def image_dispatch_tree(source,request_path):
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_dataset_predict']
    if len(functions)!=1 or [x.arg for x in functions[0].args.args]!=['args']:raise ValueError('dataset dispatcher source shape changed')
    functions[0].body=[
        ast.ImportFrom(module='deepvision_vla.inference.image',names=[ast.alias(name='run_request',asname='_dv_image_request')],level=0),
        ast.Return(value=ast.Call(func=ast.Name(id='_dv_image_request',ctx=ast.Load()),args=[ast.Name(id='args',ctx=ast.Load()),ast.Call(func=ast.Name(id='globals',ctx=ast.Load()),args=[],keywords=[]),ast.Constant(str(pathlib.Path(request_path).resolve(strict=True)))],keywords=[]))]
    return ast.fix_missing_locations(tree)

def run_request(args,namespace,request_path):
    # These dependencies are imported only for actual image processing/decoding.
    import numpy as np
    from PIL import Image
    request=read_request(request_path)
    if args.use_robot_state!=0 or args.action_chunk!=1 or args.action_dim!=7 or args.action_representation!='xyz_delta_euler_absolute_v1':raise ValueError('image API requires the accepted no-state absolute-Euler H1 contract')
    out=pathlib.Path(args.result_dir)/'image_prediction';out.mkdir(parents=True,exist_ok=False)
    image=Image.open(request['image']).convert('RGB')
    if image.size!=(224,224):raise ValueError('accepted profile expects uncropped224x224 front RGB; do not silently substitute another image protocol')
    components=namespace['model_load'](args)
    prediction=namespace['model_predict'](args,components['vl_gpt'],components['processor'],components['action_tokenizer'],components['action_head'],components['statistic'],
        request['instruction'],[image],None,None,str(out),step=0,
        dinov3_model=components['dinov3_model'],dinov3_processor=components['dinov3_processor'],dinov3_qwen_mapping=components['dinov3_qwen_mapping'])
    prediction=np.asarray(prediction,dtype=np.float32)
    # The preserved denormalize_action_chunks returns actions[0] for H1.
    # Do not silently flatten arbitrary batches/chunks into a valid-looking row.
    if prediction.shape!=(7,) or not np.isfinite(prediction).all():raise ValueError(f'expected one finite H1 action [7], got {prediction.shape}')
    command=None
    if request['current_ee_position_xyz'] is not None:
        from lift3d.helpers.graphics import EEpose
        execution=namespace['decode_action_for_execution'](prediction.copy(),None,args.action_representation)
        execution[:3]+=np.asarray(request['current_ee_position_xyz'],dtype=np.float64)
        command=np.append(EEpose.pose_6DoF_to_7DoF(execution[:-1]),execution[-1]).tolist()
        if len(command)!=8 or not all(math.isfinite(v) for v in command):raise ValueError('nonfinite execution command')
    result={'schema_version':1,'status':'prediction_completed','instruction':request['instruction'],'image_size':list(image.size),
        'image_sha256':file_sha(request['image']),'request_sha256':file_sha(request_path),'policy_conditioned_on_robot_state':False,
        'ground_truth_action_required':False,'simulator_episodes':0,'rlbench_success_rate':None,
        'action_representation':args.action_representation,'action_normalization_contract':args.action_normalization_contract,
        'predicted_action_xyz_delta_euler_grip':prediction.tolist(),'current_ee_position_xyz':request['current_ee_position_xyz'],
        'execution_xyz_xyzw_grip':command,'execution_performed':False,
        'checkpoint_training_config_sha256':file_sha(pathlib.Path(args.model_path).parent/'training_config.json'),
        'checkpoint_stats_sha256':file_sha(pathlib.Path(args.model_path).parent/'stats_data.json')}
    with (out/'prediction.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print('IMAGE_PREDICTION_COMPLETED '+json.dumps(result,ensure_ascii=False),flush=True)
    return result
