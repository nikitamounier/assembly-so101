# Native rollback — November7

Latest verified human rejects simplified MuJoCo transitions and requests the checkpoint from before the change. Active native continuation is ROOT/run-v9, from original checkpoint25/966743transitions, no bare descendant imported. **Rollback complete: native update26 committed and its matching durable checkpoint independently verified.**

Restoredcheckpoint SHA `df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b`, copied byte-for-byte from ROOT/run-v6/checkpoint.before-bare-v8.pt. Actuallearnerinitial-restored.json PASS verifies policy/Adam/all1143SGS history/reset-sampler RNG exact. Original nativephysics/model `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`: SDFcontacts/supplements/frictioncallbacks,.25mstimestep/160substeps,originalpenetration/velocity/numericalguards. OriginalsourcehashesPASS; all five tasks/rewards/actions/reset/CADinputs unchanged.

Barev8 stoppedandpreserved at581/7985663transitions, archivecheckpoint SHA `0dbbb3f59b039fc9ea2e258b55108023661654d88f3441c3ab6ab91e897f5cb0`. All40CPUleases/400schedulerCPUs and GPU3598717/3606264 retained; newnativelearner3598717,idlefallback3606264, no paidcloudwaitingdependency. Config9600envcapacity/960slots/32steps/307200targetbatch/5epochs/4minibatches/micro1024, actualvalidmaskdependsrunningretainedworkers.

ROOT `/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4`; ops-v9/active-inputs-v9.json, dashboardaliasv9. Local target-v9/start_v9/monitor_entry_v9. SameheartbeatACTIVE30s withnewnativepins; bare/earlierrunhistoriesneverrestarted, PRIMARYintentionallypaused/CUBEretired. Dashboardfollowsv9/nativeidentity and nativeoriginv6<=25 only; rejectedbaremetrics excludedfromnativechart. Autoscale-v9armedafternewnativeupdateproof; protectednear-startprojections/greedyscaling up to1000ChimeraCPUs restored. Firstcyclefoundnone<=180s afterexcludingownedhosts,submittednojobs.

## Completion evidence

Native update26 /978743totaltransitions adds12000validtransitions, sourcevalid/finite,unstable0. Rollout530.04s,optimizer.426s,375meanactiveworlds acrossretained-leasepreemptions. Matchingdurablecheckpoint SHA `822a39f28f100cf09a889f26c06cf770f864e277b6306182420fa7c19a2948eb`, finitepolicy/Adam,1143SGSrows,rollbackorigindf907...verified; no bare descendant imported. [Independent proof](live-proof.json). Currentloggedsuccess46.8% is mostlypartial/near cover; it doesnotmean completeassemblylearned. Sixcompleteactualnativeepisodeclips exist, identitiesnative8c... .
