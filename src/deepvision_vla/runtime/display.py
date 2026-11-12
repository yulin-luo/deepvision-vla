"""Authenticated task-local headless display; no fixed display numbers or global edits."""
import argparse,hashlib,json,os,pathlib,secrets,selectors,shutil,signal,subprocess,sys,time
def main():
    p=argparse.ArgumentParser();p.add_argument('--xvfb');p.add_argument('--log-dir');p.add_argument('--startup-timeout',type=float,default=30);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not cmd:p.error('a child command is required after --')
    bindir=pathlib.Path(sys.executable).resolve().parent
    xvfb=pathlib.Path(a.xvfb) if a.xvfb else bindir/'Xvfb'
    xauth=bindir/'xauth'
    if not xvfb.is_file() or not xauth.is_file():raise FileNotFoundError('install the pinned Xvfb and xauth conda packages in this environment')
    logdir=pathlib.Path(a.log_dir or pathlib.Path(os.environ['DEEPVISION_RUN_ROOT'])/'display').resolve()
    logdir.mkdir(parents=True,exist_ok=False)
    authority=logdir/'session.Xauthority';authority.touch(mode=0o600,exist_ok=False);cookie=secrets.token_hex(16)
    def authorize(number):
        result=subprocess.run([str(xauth),'-f',str(authority),'source','-'],input=f'add :{number} MIT-MAGIC-COOKIE-1 {cookie}\n',text=True,capture_output=True,timeout=10)
        if result.returncode:raise RuntimeError('xauth failed: '+result.stderr)
    authorize(0)
    readfd,writefd=os.pipe();display_env=dict(os.environ)
    for key in ['LD_LIBRARY_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM']:display_env.pop(key,None)
    display_env['PATH']=str(bindir)+os.pathsep+display_env.get('PATH','');display_env['CUDA_VISIBLE_DEVICES']=''
    dri=bindir.parent/'lib/dri'
    if dri.is_dir():
        display_env.setdefault('DRI_DRIVER_PATH',str(dri))
        display_env.setdefault('LIBGL_DRIVERS_PATH',str(dri))
    child=None;server=None
    def interrupt(signum,frame):raise KeyboardInterrupt('display wrapper interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        with (logdir/'xvfb.log').open('x') as serverlog:
            server=subprocess.Popen([str(xvfb),'-displayfd',str(writefd),'-screen','0','1024x768x24','-nolisten','tcp','-noreset','-auth',str(authority)],env=display_env,stdout=serverlog,stderr=subprocess.STDOUT,pass_fds=(writefd,))
        os.close(writefd);writefd=None
        deadline=time.monotonic()+a.startup_timeout;data=b'';selector=selectors.DefaultSelector();selector.register(readfd,selectors.EVENT_READ)
        try:
            while b'\n' not in data:
                if server.poll() is not None:raise RuntimeError(f'Xvfb exited {server.returncode}; inspect {logdir}/xvfb.log')
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Xvfb startup deadline')
                if selector.select(min(remaining,1)):
                    block=os.read(readfd,32)
                    if not block:raise RuntimeError('Xvfb displayfd closed without a display')
                    data+=block
        finally:selector.close()
        number=data.decode().strip()
        if not number.isdigit():raise ValueError('invalid Xvfb displayfd reply')
        authorize(number)
        runtime=logdir/'runtime';runtime.mkdir(mode=0o700)
        env=dict(os.environ,DISPLAY=':'+number,XAUTHORITY=str(authority),XDG_RUNTIME_DIR=str(runtime))
        if dri.is_dir():
            env.setdefault('DRI_DRIVER_PATH',str(dri));env.setdefault('LIBGL_DRIVERS_PATH',str(dri))
        info={'display':':'+number,'server_pid':server.pid,'xvfb':str(xvfb),'xvfb_sha256':hashlib.sha256(xvfb.read_bytes()).hexdigest(),'authenticated':True,'child_command':cmd,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        (logdir/'display_receipt.json').write_text(json.dumps(info,indent=2));print('X_DISPLAY_READY '+json.dumps(info),flush=True)
        child=subprocess.Popen(cmd,env=env);rc=child.wait();info.update(child_returncode=rc);(logdir/'display_receipt.json').write_text(json.dumps(info,indent=2));return rc
    finally:
        for proc in [child,server]:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        os.close(readfd)
        if writefd is not None:os.close(writefd)
        authority.unlink(missing_ok=True)
if __name__=='__main__':raise SystemExit(main())
