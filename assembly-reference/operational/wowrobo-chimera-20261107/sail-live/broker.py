"""Pinned-TLS broker; learner authentication is preserved end to end."""
import asyncio, json, subprocess
from pathlib import Path
from gateway import broker_servers

folder = Path('/run/assembly')
folder.mkdir(exist_ok=True, mode=0o700)
cert, key = folder/'cert.pem', folder/'key.pem'
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=assembly-so101','-keyout',str(key),'-out',str(cert)],check=True,capture_output=True)
key.chmod(0o600)
token = json.loads((folder/'broker.json').read_text())['token']
asyncio.run(broker_servers(cert, key, token, lambda: (folder/'ready').touch(), 7*86400))
