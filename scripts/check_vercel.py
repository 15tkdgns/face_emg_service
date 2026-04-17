# -*- coding: utf-8 -*-
import urllib.request, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
token   = os.getenv('Vercel_Token')
proj_id = os.getenv('Vercel_API')

# 최신 배포 목록 (3개)
req = urllib.request.Request(
    f'https://api.vercel.com/v6/deployments?projectId={proj_id}&limit=3',
    headers={'Authorization': f'Bearer {token}'}
)
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read())

print('=== 최근 배포 목록 ===')
for d in data['deployments']:
    ts = d.get('createdAt', 0)
    import datetime
    dt = datetime.datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M')
    commit = d.get('meta', {}).get('githubCommitMessage', '').split('\n')[0][:60]
    print(f"  [{d.get('state')}] {dt}  {commit}")

# 최신 배포 이벤트 (에러 원인)
uid = data['deployments'][0]['uid']
req2 = urllib.request.Request(
    f'https://api.vercel.com/v3/deployments/{uid}/events?limit=200',
    headers={'Authorization': f'Bearer {token}'}
)
with urllib.request.urlopen(req2) as r2:
    events = json.loads(r2.read())

print(f'\n=== 최신 배포 이벤트 마지막 40줄 ===')
texts = [e.get('text','').strip() for e in events if e.get('text','').strip()]
for t in texts[-40:]:
    print(' ', t[:160])
