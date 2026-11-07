"""Route an authorized Quest to the local simulator; XR/Enable remain deliberate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request


def find_adb(explicit=None):
    chosen=explicit or os.environ.get('TELEOP_ADB') or shutil.which('adb')
    if chosen:return chosen
    candidates=[Path(os.environ[key])/'platform-tools/adb' for key in ('ANDROID_HOME','ANDROID_SDK_ROOT') if os.environ.get(key)]
    home=Path.home()
    candidates.extend((home/'Android/Sdk/platform-tools/adb',home/'Library/Android/sdk/platform-tools/adb'))
    candidates.extend(sorted((home/'.local/opt').glob('teleop-platform-tools-*/adb'),reverse=True))
    return next((str(p) for p in candidates if p.is_file() and os.access(p,os.X_OK)),None)


def connect(*, port=18883, adb=None, serial=None):
    executable=find_adb(adb)
    if not executable:
        raise RuntimeError('adb was not found; install Android platform-tools and add adb to PATH, or pass --adb /path/to/adb')
    if not 1<=port<=65535:raise ValueError('Quest port must be between1and65535')
    def command(*args):
        result=subprocess.run([executable,*args],capture_output=True,text=True,timeout=10)
        if result.returncode:raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'ADB command failed')
        return result.stdout
    def read(route):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/{route}',timeout=5) as response:return json.load(response)
    cfg,health=read('config'),read('health')
    if cfg.get('application')!='robotics-assembly-3.0/teleop' or cfg.get('execution')!='simulation' or set(cfg.get('tools',{}))!={'left','right'}:
        raise RuntimeError('Port is not the repository SO101 simulator; no Quest route changed')
    if not health.get('owner_alive') or not health.get('render_alive') or health.get('owner_error') or health.get('render_error'):
        raise RuntimeError('Simulator owner/renderer is not healthy; no Quest route changed')
    rows=[line.split() for line in command('devices','-l').splitlines()[1:] if line.strip()]
    quests=[row for row in rows if len(row)>=2 and row[1]=='device' and any(token.lower().startswith('model:quest') for token in row)]
    selected=[row for row in quests if row[0]==serial] if serial else quests
    if len(selected)!=1:
        raise RuntimeError('Connect and authorize exactly one Quest, or pass --serial; unlock the headset and accept USB debugging')
    device=selected[0][0];endpoint=f'tcp:{port}'
    mappings=[row.split() for row in command('-s',device,'reverse','--list').splitlines()]
    matches=[row for row in mappings if len(row)>=3 and row[-2]==endpoint]
    if any(row[-1]!=endpoint for row in matches):raise RuntimeError('Quest port already has another route; no route changed')
    if not matches:command('-s',device,'reverse','--no-rebind',endpoint,endpoint)
    url=f'http://localhost:{port}/'
    result=command('-s',device,'shell','am','start','-a','android.intent.action.VIEW','-d',url,'-p','com.oculus.browser')
    if 'Error:' in result or 'Exception' in result:raise RuntimeError(result.strip())
    print(f'Quest browser opened at {url} — select Enter Quest VR, release Y/grips, then press Y.',flush=True)
    return url


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=18883)
    p.add_argument('--adb');p.add_argument('--serial');a=p.parse_args()
    try:connect(port=a.port,adb=a.adb,serial=a.serial)
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:p.exit(1,f'Quest connection failed: {error}\n')


if __name__=='__main__':main()
