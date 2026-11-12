"""Reconstruct the fixed RLBench simulator dependencies from public upstream artifacts."""
import argparse,hashlib,json,os,pathlib,shutil,tarfile,urllib.request,uuid
ARCHIVES=[
 {'name':'PyRep','filename':'PyRep-8f420be8064b1970aae18a9cfbc978dfb15747ef.tar.gz','url':'https://codeload.github.com/stepjam/PyRep/tar.gz/8f420be8064b1970aae18a9cfbc978dfb15747ef','sha256':'7ead8fc6101ab76d0aebf142b0503130fe8b30a5a69e58a67f6d298fad85c064','bytes':35839883},
 {'name':'RLBench','filename':'RLBench-841fd5c550e3ed6c196c40cb6ea636be3735ced4.tar.gz','url':'https://codeload.github.com/CSCSX/RLBench/tar.gz/841fd5c550e3ed6c196c40cb6ea636be3735ced4','sha256':'03ff4487aa683ff3eea710dd365546234c074b05fd342a506d4b6474ae0a2ae3','bytes':304397944},
 {'name':'CoppeliaSim','filename':'CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz','url':'https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz','sha256':'512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8','bytes':156375216}]
WINE=[
 {'path':'rlbench/tasks/place_wine_at_rack_location.py','sha256':'7cbf111eb26e5c8f5b130a618b421d03ab349085ec196e476f4b804b8b8b95a5'},
 {'path':'rlbench/task_ttms/place_wine_at_rack_location.ttm','sha256':'1ddd9f531c8d8ed7f54dd8ad7e68b03ade87ee0ed181a9a647b314e974612caf'}]
PERACT='ad991951bc53e4f3b73b803a75cf4b7d55295cf7'
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def extract(archive,target):
 target.mkdir(parents=True,exist_ok=False);pending=[];total=0
 with tarfile.open(archive,'r:*') as tf:
  for m in tf:
   parts=pathlib.PurePosixPath(m.name).parts
   if m.name.startswith('/') or '..' in parts:raise ValueError('unsafe archive member')
   if len(parts)<=1:continue
   dest=target.joinpath(*parts[1:])
   if not dest.resolve().is_relative_to(target.resolve()):raise ValueError('archive target escape')
   if m.isdir():dest.mkdir(parents=True,exist_ok=True)
   elif m.isfile():
    total+=m.size
    if total>3*2**30:raise ValueError('archive unpacked byte budget')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(m) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
    dest.chmod(m.mode&0o777)
   elif m.issym():
    if pathlib.Path(m.linkname).is_absolute() or not (dest.parent/m.linkname).resolve().is_relative_to(target.resolve()):raise ValueError('unsafe archive link')
    pending.append((dest,m.linkname))
   else:raise ValueError('unsupported archive entry')
 for dest,link in pending:dest.parent.mkdir(parents=True,exist_ok=True);os.symlink(link,dest)
 return total
def download(url,path,expected_sha,expected_bytes=None):
 if path.exists():
  if sha(path)!=expected_sha:raise ValueError('existing cache hash mismatch: '+str(path))
  return True
 part=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial');n=0
 request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.coppeliarobotics.com/'})
 with urllib.request.urlopen(request,timeout=90) as response,part.open('xb') as f:
  while b:=response.read(2**20):
   n+=len(b)
   if n>650*2**20:raise ValueError('download exceeds byte budget')
   f.write(b)
 if expected_bytes is not None and n!=expected_bytes:raise ValueError('download size mismatch')
 if sha(part)!=expected_sha:raise ValueError('download SHA mismatch')
 # Atomic no-clobber publication: a concurrent verified download may win.
 try:os.link(part,path)
 except FileExistsError:
  if sha(path)!=expected_sha:raise ValueError('concurrent cache hash mismatch: '+str(path))
  part.unlink()
  return True
 part.unlink()  # only this successful attempt; failed attempts remain intact
 return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--cache',required=True);a=p.parse_args()
 output=pathlib.Path(a.output_root).expanduser().resolve();cache=pathlib.Path(a.cache).expanduser().resolve()
 output.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=True);records=[]
 for item in ARCHIVES:
  archive=cache/item['filename'];reused=download(item['url'],archive,item['sha256'],item['bytes']);count=extract(archive,output/item['name'])
  records.append(dict(item,reused_cache=reused,unpacked_bytes=count));print('PREPARED',item['name'],flush=True)
 for item in WINE:
  url=f'https://raw.githubusercontent.com/MohitShridhar/RLBench/{PERACT}/'+item['path'];path=cache/pathlib.Path(item['path']).name
  reused=download(url,path,item['sha256']);target=output/'RLBench'/item['path']
  if target.exists():raise FileExistsError('wine overlay would overwrite a base task')
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
  records.append(dict(item,url=url,reused_cache=reused));print('PREPARED',item['path'],flush=True)
 with (output/'dependency_sources.json').open('x') as f:json.dump({'artifacts':records,'gpu_calls':0,'license_note':'third-party runtime/source terms apply; binaries are downloaded from upstream, not redistributed in this code repository'},f,indent=2)
 print(json.dumps({'status':'prepared_sources_not_yet_built_or_simulator_validated','root':str(output),'items':len(records)}))
if __name__=='__main__':main()
