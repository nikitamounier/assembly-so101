"""Read-only live provider accounting; queued cores never count as running."""
from pathlib import Path
import asyncio,datetime,json,time
from modal.client import _Client
from modal_proto import api_pb2
HERE=Path(__file__).resolve().parent
async def main():
 receipt=json.loads((HERE/'active.json').read_text());client=await _Client.from_env()
 layout=await client._stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=receipt['app_id']))
 rows=[]
 for name,identifier in layout.app_layout.function_ids.items():
  stats=await client._stub.FunctionGetCurrentStats(api_pb2.FunctionGetCurrentStatsRequest(function_id=identifier))
  rows.append(dict(name=name,function_id=identifier,queued=stats.backlog,containers=stats.num_total_tasks,running_inputs=stats.num_running_inputs))
 by={r['name']:r for r in rows};large=by.get('worker',{}).get('running_inputs',0);small=by.get('worker_remainder',{}).get('running_inputs',0)
 stamp=Path(receipt['receipt']).parent.name
 start=datetime.datetime.strptime(stamp,'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()
 continuous=receipt.get('mode')=='continuous'
 expires=None if continuous else start+receipt['seconds']
 result=dict(action='keep_monitor',app_id=receipt['app_id'],time_unix=time.time(),pool_status=receipt['status'],functions=rows,requested_rollout_physical_cores=receipt['requested_physical_cores'],running_rollout_physical_cores=60*large+12*small,running_rollout_processes=6*large+small,cumulative_initialized_workers=receipt.get('initialized_workers',0),cumulative_first_control_workers=receipt.get('first_control_workers',0),packed_smoke_passed=receipt.get('packed_smoke_passed',False),local_controller_age_s=time.time()-receipt.get('controller_heartbeat_unix',time.time()),conservative_pool_expiry_unix=expires,expiry_notice_due=False if continuous else time.time()>=expires-900,continuous=continuous,automatic_lease_renewal=receipt.get('automatic_lease_renewal',False),elapsed_time_cutoff=receipt.get('elapsed_time_cutoff',not continuous))
 if continuous:
  funding=receipt.get('credit_snapshot',{});elapsed=max(0,time.time()-funding.get('checked_unix',time.time()));rate=max(receipt.get('estimated_full_pool_usd_per_hour',[245]));credit=max(0,funding.get('remaining_usd',0)-elapsed*rate/3600);headroom=max(0,funding.get('usage_limit_usd',0)-funding.get('usage_usd',0)-elapsed*rate/3600)
  result.update(credit_snapshot=funding,estimated_remaining_credits_usd=credit,estimated_usage_cap_headroom_usd=headroom,funding_estimate_only=True,funding_notice_due=min(credit,headroom)<=rate*.25)
 if receipt['status']=='workers_dispatched' and result['local_controller_age_s']>120:result['action']='ACTION cloud_controller_stale'
 if receipt['status']=='failed':result['action']='ACTION cloud_pool_failed'
 (HERE/'cloud-status.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':asyncio.run(main())
