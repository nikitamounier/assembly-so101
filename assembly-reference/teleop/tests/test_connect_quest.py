"""USB connection preserves routes and refuses wrong/unready simulator targets."""
import io
import json
from types import SimpleNamespace
import pytest
from teleop import connect_quest


@pytest.mark.parametrize('location',['Android/Sdk/platform-tools/adb','.local/opt/teleop-platform-tools-test/adb'])
def test_discovers_existing_sdk_without_changing_path(monkeypatch,tmp_path,location):
    for key in ('TELEOP_ADB','ANDROID_HOME','ANDROID_SDK_ROOT'):monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(connect_quest.shutil,'which',lambda _:None)
    monkeypatch.setattr(connect_quest.Path,'home',lambda:tmp_path)
    executable=tmp_path/location;executable.parent.mkdir(parents=True);executable.write_text('test executable');executable.chmod(0o755)
    assert connect_quest.find_adb()==str(executable)


def fixture(monkeypatch,*,application='robotics-assembly-3.0/teleop',devices=None,routes=''):
    calls=[]
    def read(url,timeout):
        value=({'application':application,'execution':'simulation','tools':{'left':{},'right':{}}}
               if url.endswith('/config') else {'owner_alive':True,'render_alive':True})
        return io.BytesIO(json.dumps(value).encode())
    def run(args,**kwargs):
        calls.append(args[1:])
        result=(devices if devices is not None else 'List of devices attached\nquest-test device model:Quest_3S\n') if args[1:3]==['devices','-l'] else routes if args[-2:]==['reverse','--list'] else ''
        return SimpleNamespace(returncode=0,stdout=result,stderr='')
    monkeypatch.setattr(connect_quest.urllib.request,'urlopen',read)
    monkeypatch.setattr(connect_quest.subprocess,'run',run)
    return calls


def test_refuses_other_service_before_any_adb_action(monkeypatch):
    calls=fixture(monkeypatch,application='another-app')
    with pytest.raises(RuntimeError,match='not the repository'):connect_quest.connect(adb='adb-test')
    assert calls==[]


def test_refuses_unauthorized_headset_without_changing_routes(monkeypatch):
    calls=fixture(monkeypatch,devices='List of devices attached\nquest-test unauthorized model:Quest_3S\n')
    with pytest.raises(RuntimeError,match='authorize'):connect_quest.connect(adb='adb-test')
    assert calls==[['devices','-l']]


def test_preserves_conflicting_reverse_mapping(monkeypatch):
    calls=fixture(monkeypatch,routes='transport tcp:18883 tcp:19000\n')
    with pytest.raises(RuntimeError,match='another route'):connect_quest.connect(adb='adb-test')
    assert not any('--no-rebind' in call or 'shell' in call for call in calls)


def test_custom_port_routes_and_opens_only_authorized_quest(monkeypatch):
    calls=fixture(monkeypatch,routes='transport tcp:18883 tcp:18883\n')
    assert connect_quest.connect(port=18884,adb='adb-test',serial='quest-test')=='http://localhost:18884/'
    assert ['-s','quest-test','reverse','--no-rebind','tcp:18884','tcp:18884'] in calls
    assert calls[-1]==['-s','quest-test','shell','am','start','-a','android.intent.action.VIEW','-d','http://localhost:18884/','-p','com.oculus.browser']
