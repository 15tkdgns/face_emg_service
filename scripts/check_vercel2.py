# -*- coding: utf-8 -*-
import urllib.request, json, os, sys, datetime
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
token   = os.getenv('Vercel_Token')
proj_id = os.getenv('Vercel_API')

req = urllib.request.Request(
    f'https://api.vercel.com/v6/deployments?projectId={proj_id}&limit=6',
    headers={'Authorization': f'Bearer {token}'}
)
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read())

print('=== 최근 배포 6개 (최신순) ===')
for d in data['deployments']:
    ts = d.get('createdAt', 0)
    dt = datetime.datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M:%S')
    commit = d.get('meta', {}).get('githubCommitRef','') + ' | ' + d.get('meta', {}).get('githubCommitMessage','').split('\n')[0][:50]
    print(f"  [{d.get('state'):10}] {dt}  {commit}")
    print(f"    URL: {d.get('url')}")
