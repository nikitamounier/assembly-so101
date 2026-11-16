# Distributed reset seed groups

User rejected the90minute estimate and requested fastercompletion. Same exact25×(60TRAIN+12EVAL) bank, geometry, noise, passivehinges and .08prep+.2holdnative overlap/retentionadmission. No trainingjobs changed.

Originaljob3620194 remains the finalbankowner; its exactcoordinator2650930 ispaused, and its8provenstagechildren terminatedafteratomiccheckpointpreservation. All acceptedreceipts preserved. `prepare-result.json` recordsassignment:82unfinishedgroups, fourdisjointglobal-slotpartitions perstage,4alreadycomplete stagebanks.

`partition_worker.py` imports the unchanged frozen generator and changes only its slot iterator:globalTRAINindices part::4of60 and EVALpart::4of12; attemptlimits20/40/80/160 unchanged. ASTasserts exactlytwo rangecalls(slot,attempt), fulltargetsandseedmappingunchanged, same source/modelidentityasserted againstmaster. Each group has a separateoutputfolder, inheritedexclusivelease and exact18owned-slotcheck. Existing admittedslots areseeded with byte-identical referencedarchives. `partition-proof.json` recordscodehashes/proof.

Arrayjobs3621555(GPU1142),3621556(GPU11AE),3621557(GPU36DC),3621558(GPU720E),3621560(GPUC972),2CPU/16GiB pergroup.57initialjobsstarted; per-userQOSmaxjobs100blockedtherest. Only19stillpendingarraytasks werecanceled/replaced bypackedjobs3621644(10groups/20CPU160GiB,GPU720E) and3621645(9groups/18CPU144GiB,GPUC972); runningarraytasksuntouched. See submitted-arrays.json andpacked-submissions.json. GPUsrequested:zero. Targetnodeshadnoowntrainingjobs atsubmission.

`farm_control.py` fencesclaims globally, mergesonlyfull72-slotstageunions andbyte-verifiesallartifacts. Requeuedoriginalcoordinatorshitlaunchgatebeforeclaimingstages. Supervisorpublishesstageswithallpartleasesinherited; onlyafter25fullstagebanksandnoheldleases closesfence+resumesoriginalcoordinator toseal1800andexport. LocaldurablesupervisorPID74642 retriesallocatedsrunafterpreemption, status outputs/reset-bank-farm-20261107/supervisor-status.json. OriginalcompletioncollectorPID65735/job3620194 stillverifiesandreturnsfullbanktoMacviewer.

Firstobserveddistributedproof:743uniqueacceptedglobal(stage,split,slot)keys;57runninggroups,6complete,63initialized. Countsunionmaster+partitionreceipts; copiedexistingreceiptsneverdoublecounted. Moregroupsstillqueuedatthischeck. CurrentCLI helpers: `farm_control.py supervise-partitions --root ROOT --timeout5400`; do notmanuallyresume whileexternalleasesareactive.

Packedlaunchamendment:3621644/3621645 failedbeforeworkers because SlurmSMT-adjusted SLURM_CPUS_PER_TASK conflicted withSLURM_TRES_PER_TASK. Fixedexplicit srun--cpus-per-task toTRESrequestedcount; replacement3621781(10groups,GPU720E)/3621782(9groups,GPUC972). Aplannedextra consolidationwasnotexecuted: firstpending-cancelreturnedinvalidalreadyfailedIDsandstoppedbeforeanyrunningjobcancel. Originalrunningarrayworkersuntouched.

Currentverifiedproof:all82assignedgroupsinitializedacross5nodes,66running/16complete;962uniqueadmittedglobal-slotkeys at1794096325.46. Packed3621781and3621782RUNNING; source/modelidentityandnativechecksretainoriginalphysics. Originalsupervisornativepublished6complete72-slotstagebanks atthischeck; remaininggroupscontinue. RevisedETA20–30minutes is throughput/tail estimate, notcompletionclaim. Evidencepartition-generation-proof.json.


## 2026-11-07 [codex] Full1800 reset bank completed and viewer verified

- All25×(60TRAIN+12EVAL)=1800 native-admitted variations sealed, finite fullnative payloads rechecked; immutableLinuxmodel1b071d9b92dc5e3e3e9ba33234a25f24bfc3b3049081a52ac673b1df423b952d. Native minimum remainsno unintendedoverlap+heldgripretention,motionallowed. Jobs3620194/3621781/3621782 COMPLETED0:0. Farm supervisor resumedoriginalonlyafter25complete72statecanonicalbanks; finalverification succeeded.
- Export69,931,190bytes SHAa9b7d7cf706d2ec47b0141d04534799b5078ff3f71496295751918f2e0b02af6 verifiedandreturned. Untouchednativebank:outputs/reset-bank-chimera-completion-20261107/linux-qualified-result/native-bank. Mac display assets_local/wowrobo/randomization-bank-1800-valid-chimera-v4; local-relocation.json preservesqualificationplatformLinux andlocal_native_admission_repeated=false,151rawassets/448pointcontractverified. No active trainingrun changed.
- LIVEhttp://localhost:8878/,viewerPID89326. FreshHTTPcatalog1825entries=1800trainingeligiblevariations+25nominals;25distinctstages exactly72each. Collectorstatus complete_and_published, viewer_readytrue. Payloadquota rechecked25×60/12,allfinite.
- Retiredonlyownredundantlocalgenerator supervisor45153/coordinator44994/children93066,93369,94200 afterexactcommandchecks. Alllocalartifacts preserved, retirement.json underlocalv4plan; liveviewer untouched. Farm74642andcollector65735finishednaturally.
