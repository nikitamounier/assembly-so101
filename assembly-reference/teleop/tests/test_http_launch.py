"""Launch the public CLI on a real ephemeral port and exercise its browser API."""
import gzip
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import pytest


@pytest.mark.parametrize('quest_without_usb',[False,True])
def test_module_cli_serves_shared_physics_stereo_assets_and_fresh_views(quest_without_usb):
    root=Path(__file__).resolve().parents[2]
    command=[sys.executable,'-m','teleop.serve','--port','0','--seconds','8']
    if quest_without_usb:command+=['--quest','--adb',str(root/'missing-test-adb')]
    process=subprocess.Popen(command,cwd=root,
                             env=dict(os.environ,MUJOCO_GL='cgl' if sys.platform=='darwin' else 'egl'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    lines=queue.Queue();threading.Thread(target=lambda:lines.put(process.stdout.readline()),daemon=True).start()
    try:
        ready=json.loads(lines.get(timeout=60));base=ready['url'].replace('localhost','127.0.0.1')
        def read(path):
            with urllib.request.urlopen(base+path,timeout=5) as response:
                content=response.read()
                if response.headers.get('Content-Encoding')=='gzip':content=gzip.decompress(content)
                return content
        config=json.loads(read('/api/config'));state=json.loads(read('/api/state'))
        assert config['application']=='robotics-assembly-3.0/teleop'
        assert config['environment_identity']['profile']=='shared-side-v14-horn-support'
        assert not state['enabled'] and not state['faults']
        assert b'immersive.mjs' in read('/app.js') and b'<html' in read('/')
        scene=json.loads(read('/api/scene'));assert scene['body_count']==27
        deadline=time.monotonic()+5
        while True:
            health=json.loads(read('/api/health'))
            if health['render_alive']:break
            assert time.monotonic()<deadline
            time.sleep(.05)
        views=json.loads(read('/api/views'));assert set(views['frames'])=={'main','left','right'}
        assert read('/api/frame.jpg?camera=main').startswith(b'\xff\xd8')
    finally:
        process.terminate()
        _,errors=process.communicate(timeout=10)
        assert process.returncode==0,errors
