"""Small five-axis controller: clutch mapping, bounded IK and local simulation Hold.

Position is primary. Rotation uses only the remaining position-nullspace axes;
unachievable full orientation is reported, never advertised as six-DoF tracking.
This module has no USB, network, camera or physical SDK dependency.
"""
from dataclasses import dataclass
import numpy as np
import mujoco

def vector(value,n=3):
    result=np.asarray(value,dtype=float)
    if result.shape!=(n,)or not np.isfinite(result).all():raise ValueError('finite_vector_required')
    return result.copy()

def rotation(value):
    r=np.asarray(value,dtype=float)
    if r.shape!=(3,3)or not np.isfinite(r).all()or not np.allclose(r.T@r,np.eye(3),atol=1e-7)or np.linalg.det(r)<.999999:
        raise ValueError('proper_rotation_required')
    return r.copy()

@dataclass
class Pose:
    position:np.ndarray
    rotation:np.ndarray

class ClutchMap:
    """One controller goal marker and one measured tool marker, in one metric frame."""
    def __init__(self,gain=1.,acquire_radius=.05):
        if not np.isfinite(gain)or not 0<gain<=1:raise ValueError('bounded_gain_required')
        self.gain=float(gain);self.acquire_radius=acquire_radius;self.R=np.eye(3);self.t=np.zeros(3);self.grabs={}
    def aligned(self,p):return self.R@vector(p)+self.t
    def acquire(self,side,p,r,tool):
        point=self.aligned(p);r=self.R@rotation(r)
        if self.acquire_radius is not None and np.linalg.norm(point-tool.position)>self.acquire_radius:return False
        self.grabs[side]=(point,r,Pose(tool.position.copy(),tool.rotation.copy()))
        return True
    def target(self,side,p,r):
        anchor,orient,tool=self.grabs[side]
        return Pose(tool.position+self.gain*(self.aligned(p)-anchor),self.R@rotation(r)@orient.T@tool.rotation)
    def release(self,side):self.grabs.pop(side,None)
    def recenter(self,R,t,controllers):
        # Snapshot every existing goal before changing frame. Repeated recenter
        # changes subsequent deltas, never the current arm reference.
        R,t=rotation(R),vector(t)
        if set(self.grabs)-set(controllers):raise ValueError('active_controller_pose_required')
        goals={s:self.target(s,*controllers[s])for s in self.grabs}
        self.R,self.t=R,t
        for s,goal in goals.items():
            p,r=controllers[s];self.grabs[s]=(self.aligned(p),R@rotation(r),goal)

class Kinematics:
    def __init__(self,model,description):
        self.model=model;self.data=mujoco.MjData(model);self.arms={};self.previous={}
        for side,a in description['arms'].items():
            j=np.array([model.joint(n).id for n in a['joint_names']],dtype=int)
            self.arms[side]=dict(q=model.jnt_qposadr[j],v=model.jnt_dofadr[j],
                limits=model.jnt_range[j].copy(),site=model.site(a['tool_site']).id)
    def forward(self,qpos,side):
        self.data.qpos[:]=qpos;mujoco.mj_kinematics(self.model,self.data)
        a=self.arms[side];s=a['site']
        return Pose(self.data.site_xpos[s].copy(),self.data.site_xmat[s].reshape(3,3).copy())
    @staticmethod
    def orientation_error(actual,desired):
        # Shortest SO(3) logarithm, including half turns where sin(theta)=0.
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,(desired@actual.T).reshape(-1))
        if quat[0]<0:quat=-quat
        length=np.linalg.norm(quat[1:])
        return quat[1:]*(2*np.arctan2(length,quat[0])/length) if length>1e-12 else np.zeros(3)
    def solve(self,qpos,side,goal,iterations=55):
        previous=self.previous.pop(side,None)
        position=vector(goal.position);desired=rotation(goal.rotation)
        a=self.arms[side];self.data.qpos[:]=qpos
        if np.linalg.norm(position-self.forward(qpos,side).position)>.6:
            return dict(accepted=False,reason='unreachable_distance',iterations=0)
        # Re-seed from measured posture. Translation and attainable rotation
        # are solved together; a moving controller must not suppress rotation.
        # Continue from the measured posture; previous is retained only as
        # an acquisition/continuation witness for the secondary trust region.
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        best=None;best_angle=np.inf;secondary_used=0.
        # First geometric solve uses the full work budget. Continuation
        # secondary pursuit stays near the measured posture instead of banking
        # a distant orientation target behind the position-reference limiter.
        secondary_budget=np.inf if previous is None else .12
        for count in range(max(1,min(55,int(iterations)))):
            mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data)
            error=position-self.data.site_xpos[a['site']]
            actual=self.data.site_xmat[a['site']].reshape(3,3).copy()
            orient=self.orientation_error(actual,desired)
            settled=np.linalg.norm(error)<.0005
            if settled and np.linalg.norm(orient)<best_angle:
                best=self.data.qpos[a['q']].copy();best_angle=np.linalg.norm(orient)
            mujoco.mj_jacSite(self.model,self.data,jp,jr,a['site'])
            J=jp[:,a['v']];u,s,vt=np.linalg.svd(J,full_matrices=True)
            inverse=vt[:3].T@np.diag(s/(s*s+1e-6))@u.T
            delta=inverse@error
            null=vt[int(np.sum(s>1e-4)):].T
            if not settled and null.shape[1]:
                q=self.data.qpos[a['q']];lo=a['limits'][:,0];hi=a['limits'][:,1]
                gradient=1/np.maximum(q-lo,.01)-1/np.maximum(hi-q,.01)
                avoidance=.8*null@(null.T@gradient)
                ratio=min(1.,np.linalg.norm(delta)/max(np.linalg.norm(avoidance),1e-12))
                delta+=ratio*avoidance
            if null.shape[1]:
                # Position has priority; remaining axes compensate rotation
                # caused by translation and pursue the controller orientation.
                N=jr[:,a['v']]@null
                residual=orient-jr[:,a['v']]@delta
                correction=null@np.linalg.lstsq(N,residual,rcond=1e-4)[0]
                norm=np.linalg.norm(correction)
                remaining=min(.08,secondary_budget-secondary_used)
                if norm>=1e-5 and remaining>=1e-5:
                    if norm>remaining:correction*=remaining/norm
                    secondary_used+=np.linalg.norm(correction)
                    delta+=correction
                elif settled:break
            elif settled:break
            magnitude=np.max(abs(delta))
            if magnitude>.08:delta*=.08/magnitude
            q=self.data.qpos[a['q']].copy();lower=a['limits'][:,0]+1e-5;upper=a['limits'][:,1]-1e-5
            if settled:
                # Scale the complete nullspace step at limits; clipping its
                # individual joints would destroy position cancellation.
                ratios=np.ones(5)
                positive=delta>0;negative=delta<0
                ratios[positive]=(upper[positive]-q[positive])/delta[positive]
                ratios[negative]=(lower[negative]-q[negative])/delta[negative]
                fraction=max(0.,min(1.,float(np.min(ratios))))
                if fraction<1e-6:break
                self.data.qpos[a['q']]=q+fraction*delta
            else:self.data.qpos[a['q']]=np.clip(q+delta,lower,upper)
        mujoco.mj_kinematics(self.model,self.data)
        residual=float(np.linalg.norm(position-self.data.site_xpos[a['site']]))
        angle=float(np.linalg.norm(self.orientation_error(self.data.site_xmat[a['site']].reshape(3,3),desired)))
        if best is not None and (residual>=.0005 or best_angle<angle):
            self.data.qpos[a['q']]=best;mujoco.mj_kinematics(self.model,self.data)
            residual=float(np.linalg.norm(position-self.data.site_xpos[a['site']]))
            angle=float(np.linalg.norm(self.orientation_error(self.data.site_xmat[a['site']].reshape(3,3),desired)))
        if residual<.004:self.previous[side]=self.data.qpos[a['q']].copy()
        return dict(accepted=residual<.004,reason='reachable_position'if residual<.004 else'position_unreachable',
            target=self.data.qpos[a['q']].copy(),position_error_m=residual,orientation_error_rad=angle,
            iterations=count+1,orientation_scope='position-priority with simultaneous achievable rotation; full orientation may be infeasible')

class SimulationController:
    """Position controls on actual MuJoCo dynamics, with no plant qpos actuation."""
    native_substeps=1
    stop_on_joint_velocity=True
    def __init__(self,model,description):
        self.model=model;self.description=description;self.data=mujoco.MjData(model)
        # The only assignment to plant qpos is this explicit initial state.
        self.data.qpos[:]=description['home_qpos'];mujoco.mj_forward(model,self.data)
        self.ik=Kinematics(model,description);self.mapper=ClutchMap();self.enabled=False;self.stopped=False
        # Simulation effort release, not an electrical power cut. Native disable
        # removes position-servo gain AND bias forces without changing force caps.
        self._actuation_disable_bit=int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.torque_enabled=not bool(int(model.opt.disableflags)&self._actuation_disable_bit)
        self._torque_disable_owned=False
        self.recovery_source_ns=None;self.recovery_active=False
        self.reason='startup_disabled';self.goals={};self.admitted_goals={};self.admitted_sources={};self.submit_status={};self.samples={};self.faults=[];self.hazards=[];self.fault_events=[]
        self.qids=np.array([model.jnt_qposadr[int(j)]for j in model.actuator_trnid[:,0]],dtype=int)
        self.dofs=np.array([model.jnt_dofadr[int(j)]for j in model.actuator_trnid[:,0]],dtype=int)
        self.reference=self.data.qpos[self.qids].copy();self.data.ctrl[:]=self.reference
        self.targets=self.reference.copy();self.cartesian_speed=.25;self.speed=.8;self.acceleration=4.;self.reference_velocity=np.zeros(model.nu)
        self.ids={s:np.array([model.actuator(n).id for n in a['actuator_names']],dtype=int)for s,a in description['arms'].items()}
        self.reference_groups=list(self.ids.values())+[np.array([model.actuator(a['gripper_actuator']).id])for a in description['arms'].values()]
    def tool(self,side):
        s=self.model.site(self.description['arms'][side]['tool_site']).id
        return Pose(self.data.site_xpos[s].copy(),self.data.site_xmat[s].reshape(3,3).copy())
    def enable(self):
        if self.stopped or self.faults:return False
        if not self.torque_enabled:
            if not self._torque_disable_owned:
                self.reason='actuation_disabled_outside_controller';return False
            if not self._observed_rest():
                self.reason='torque_enable_requires_finite_observed_rest';return False
            measured=self.data.qpos[self.qids]
            lower,upper=self.model.jnt_range[self.model.actuator_trnid[:,0],0],self.model.jnt_range[self.model.actuator_trnid[:,0],1]
            target=self.data.qpos[self.qids].copy()
            if (np.any(measured<lower-.005)or np.any(measured>upper+.005)
                or np.any(target<self.model.actuator_ctrlrange[:,0])or np.any(target>self.model.actuator_ctrlrange[:,1])):
                self.reason='torque_enable_measured_position_out_of_bounds';return False
            self.hold('torque_enable_measured_seed')
            self.reference=target.copy();self.targets=target.copy();self.reference_velocity[:]=0.
            self.data.ctrl[:]=target
            self.model.opt.disableflags=int(self.model.opt.disableflags)&~self._actuation_disable_bit
            self._torque_disable_owned=False;self.torque_enabled=True
            if self.collisions():
                self.enabled=False;self.hold('torque_restored_contact_hold');return False
        elif self.collisions():return False
        self.enabled=True;self.reason='enabled';return True
    def _observed_rest(self):
        return (np.isfinite(self.data.qpos).all()and np.isfinite(self.data.qvel).all()
                and np.max(abs(self.data.qvel[self.dofs]))<=.02)
    def release_torque(self):
        # The owner validates the original uninterrupted three-second gesture,
        # both released grips and fresh input/feedback/video before this call.
        if (self.enabled or self.stopped or self.faults or self.mapper.grabs
            or self.recovery_active or not self._observed_rest()):return False
        if not self.torque_enabled:return True
        self.hold('torque_released_passive_simulation')
        self.model.opt.disableflags=int(self.model.opt.disableflags)|self._actuation_disable_bit
        self._torque_disable_owned=True;self.torque_enabled=False
        return True
    def recovery_held(self,now_ns):
        return self.recovery_source_ns is not None and 0<=now_ns-self.recovery_source_ns<200_000_000
    def recovery_valid(self,now_ns):
        return self.recovery_active and bool(self.mapper.grabs) and self.recovery_held(now_ns)
    def set_recovery(self,held,source_ns,now_ns):
        # Fresh Y is a witness, not permission for an existing ordinary grab.
        # Only a subsequent successful direct fresh grip activates recovery.
        if held and self.torque_enabled and not self.stopped and not self.faults and 0<=now_ns-source_ns<200_000_000:
            self.recovery_source_ns=int(source_ns);return True
        was_recovery=self.recovery_active
        self.recovery_source_ns=None;self.recovery_active=False
        if was_recovery and self.collisions() and self.enabled:
            self.enabled=False;self.hold('collision_recovery_released_in_contact')
        return False
    def hold(self,reason='release'):
        self.recovery_source_ns=None;self.recovery_active=False
        self.targets=self.data.qpos[self.qids].copy();self.goals.clear();self.admitted_goals.clear();self.admitted_sources.clear();self.samples.clear();self.mapper.grabs.clear();self.ik.previous.clear();self.reason=reason
    def stop(self):
        self.stopped=True;self.enabled=False;self.hold('stop_latched')
    def reset_stop(self):
        if np.max(abs(self.data.qvel[self.dofs]))>.02:return False
        self.stopped=False;self.reason='disabled_after_stop_reset';return True
    def grab(self,side,p,r,source_ns,now_ns):
        if not self.torque_enabled or side not in self.ids or self.stopped or self.faults or side in self.mapper.grabs or not (self.enabled or self.recovery_held(now_ns)) or not 0<=now_ns-source_ns<200_000_000:return False
        # Validate direct proximity/rotation before releasing any other clutch.
        tool=self.tool(side)
        if self.mapper.acquire_radius is not None and np.linalg.norm(self.mapper.aligned(p)-tool.position)>self.mapper.acquire_radius:return False
        rotation(r)
        if self.recovery_held(now_ns) and not self.recovery_active:
            witness=self.recovery_source_ns
            self.hold('collision_recovery_arming')
            self.recovery_source_ns=witness
        acquired=self.mapper.acquire(side,p,r,tool)
        if acquired:
            self.ik.previous.pop(side,None);self.goals.pop(side,None);self.submit_status.pop(side,None)
            self.admitted_goals[side]=Pose(tool.position.copy(),tool.rotation.copy());self.admitted_sources[side]=int(source_ns)
            self.enabled=True
            if self.recovery_held(now_ns):
                self.recovery_active=True;self.reason='collision_recovery_active'
        return acquired
    def release(self,side):
        self.mapper.release(side)
        if not self.mapper.grabs:self.recovery_active=False
        self.samples.pop(side,None);self.goals.pop(side,None);self.admitted_goals.pop(side,None);self.admitted_sources.pop(side,None);self.ik.previous.pop(side,None)
        ids=np.append(self.ids[side],self.model.actuator(self.description['arms'][side]['gripper_actuator']).id)
        self.targets[ids]=self.data.qpos[self.qids[ids]];self.reason='released_'+side
    def submit(self,side,p,r,source_ns,now_ns):
        if not self.enabled or self.stopped or side not in self.mapper.grabs:return False
        previous=self.samples.get(side,-1)
        if source_ns<=previous or not 0<=now_ns-source_ns<200_000_000 or (previous>=0 and now_ns-previous>=200_000_000):
            self.release(side);self.reason='stale_or_reordered';self.submit_status[side]=dict(reason='stale_or_reordered',source_ns=int(source_ns));return False
        try:
            goal=self.mapper.target(side,p,r)
            prior=self.goals.get(side,self.admitted_goals[side])
            raw_step=float(np.linalg.norm(goal.position-prior.position))
            if raw_step>.15:raise ValueError('excessive_pose_jump')
        except (ValueError,np.linalg.LinAlgError)as error:
            self.release(side);self.reason='target_rejected';self.submit_status[side]=dict(reason=str(error),source_ns=int(source_ns));return False
        # This is the newest genuine acquisition, even when IK locally Holds.
        # Spend only its interval: blocked time never banks catch-up motion.
        dt=min(.2,max(0.,(source_ns-self.admitted_sources[side])/1e9))
        admitted=self.admitted_goals[side];delta=goal.position-admitted.position
        distance=float(np.linalg.norm(delta));budget=self.cartesian_speed*dt
        position=admitted.position+delta*min(1.,budget/max(distance,1e-12))
        reference=Pose(position,goal.rotation.copy())
        self.samples[side]=int(source_ns);self.admitted_sources[side]=int(source_ns);self.goals[side]=goal
        detail=dict(source_ns=int(source_ns),raw_step_m=raw_step,requested_speed_m_s=raw_step/dt if dt else None,
            admitted_step_m=float(np.linalg.norm(position-admitted.position)))
        try:
            result=self.ik.solve(self.data.qpos,side,reference)
        except (ValueError,np.linalg.LinAlgError)as error:
            result=dict(accepted=False,reason='ik_error:'+str(error))
        if not result['accepted']:
            self.ik.previous.pop(side,None);self.targets[self.ids[side]]=self.data.qpos[self.qids[self.ids[side]]];self.reason='target_held'
            self.submit_status[side]=dict(detail,reason='ik_hold',ik_reason=result['reason'],**{k:result[k] for k in ('position_error_m','orientation_error_rad','orientation_scope') if k in result});return True
        self.admitted_goals[side]=reference;self.targets[self.ids[side]]=result['target'];self.reason='active'
        self.submit_status[side]=dict(detail,reason='reference_limited'if distance>budget+1e-12 else'active',ik_reason=result['reason'],**{k:result[k] for k in ('position_error_m','orientation_error_rad','orientation_scope') if k in result});return True
    def gripper(self,side,openness,now_ns):
        if not self.enabled or self.stopped or side not in self.mapper.grabs:return False
        if side not in self.samples or not 0<=now_ns-self.samples[side]<200_000_000:return False
        value=float(openness)
        if not np.isfinite(value)or not 0<=value<=1:return False
        a=self.model.actuator(self.description['arms'][side]['gripper_actuator']).id
        arm=self.description['arms'][side]
        lo,hi=arm['gripper_closed'],arm['gripper_open'];self.targets[a]=lo+value*(hi-lo);return True
    def collisions(self):
        # Actual native collision contacts, including separate arms and table.
        # Model contact exclusions describe adjacent mechanical connections.
        result=[]
        for c in self.data.contact:
            if c.dist<0:
                a,b=map(int,c.geom);result.append(dict(geoms=[self.model.geom(a).name,self.model.geom(b).name],distance_m=float(c.dist)))
        return result
    def step(self,now_ns):
        if not self.torque_enabled:
            # Gravity, native joint constraints, damping and contacts continue.
            # Passive motion has no active speed/support guarantee or auto-rearm.
            mujoco.mj_step(self.model,self.data,nstep=self.native_substeps)
            if not np.isfinite(self.data.qpos).all()or not np.isfinite(self.data.qvel).all():
                self.faults.append('nonfinite_measured');self.stop()
            return self.tool(next(iter(self.description['arms'])))
        if self.recovery_source_ns is not None and not self.recovery_held(now_ns):
            if self.recovery_active:
                self.enabled=False;self.hold('original_recovery_input_expired')
            else:self.recovery_source_ns=None
        for side,sample in list(self.samples.items()):
            if now_ns-sample>=200_000_000:self.release(side);self.reason='original_input_expired'
        hazards=self.collisions()
        if hazards and self.enabled and not self.recovery_valid(now_ns):
            self.hazards.append(dict(time_s=self.data.time,contacts=hazards));self.enabled=False;self.hold('modeled_collision_hold')
        dt=self.model.opt.timestep*self.native_substeps;distance=self.targets-self.reference
        # One progress rate keeps a multi-joint pose coordinated. Independent
        # saturated rates can overshoot the Cartesian goal as short axes arrive.
        desired_velocity=np.zeros_like(distance)
        for group in self.reference_groups:
            span=float(np.max(abs(distance[group])))
            progress=min(self.speed,np.sqrt(2*self.acceleration*span))
            desired_velocity[group]=distance[group]*(progress/max(span,1e-12))
        if hazards and self.recovery_valid(now_ns):
            # Avoid charging position-servo error against contact. This is a
            # soft 20 mrad braking margin, not a hard lead/stopping guarantee.
            lead=self.reference-self.data.qpos[self.qids]
            braking=np.sqrt(2*self.acceleration*np.maximum(0.,.02-np.sign(desired_velocity)*lead))
            desired_velocity=np.sign(desired_velocity)*np.minimum(abs(desired_velocity),braking)
        velocity_change=desired_velocity-self.reference_velocity
        for group in self.reference_groups:
            self.reference_velocity[group]+=velocity_change[group]*min(1.,self.acceleration*dt/max(float(np.max(abs(velocity_change[group]))),1e-12))
        self.reference+=self.reference_velocity*dt
        self.reference=np.clip(self.reference,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
        self.data.ctrl[:]=self.reference;mujoco.mj_step(self.model,self.data,nstep=self.native_substeps)
        if not np.isfinite(self.data.qpos).all()or not np.isfinite(self.data.qvel).all():
            self.faults.append('nonfinite_measured');self.stop()
        if self.stop_on_joint_velocity and np.max(abs(self.data.qvel[self.dofs]))>1.5:
            if len(self.fault_events)<4:self.fault_events.append(dict(reason='simulation_joint_velocity',time_s=float(self.data.time),
                velocity=self.data.qvel.copy().tolist(),position=self.data.qpos.copy().tolist(),
                reference=self.reference.copy().tolist(),target=self.targets.copy().tolist(),effort=self.data.actuator_force.copy().tolist()))
            self.faults.append('simulation_joint_velocity');self.stop()
        lower,upper=self.model.jnt_range[self.model.actuator_trnid[:,0],0],self.model.jnt_range[self.model.actuator_trnid[:,0],1]
        measured=self.data.qpos[self.qids]
        if np.any(measured<lower-.005)or np.any(measured>upper+.005):
            self.faults.append('simulation_joint_position');self.stop()
        effort=self.data.actuator_force
        if np.any(effort<self.model.actuator_forcerange[:,0]-1e-8)or np.any(effort>self.model.actuator_forcerange[:,1]+1e-8):
            self.faults.append('simulation_actuator_effort');self.stop()
        return self.tool(next(iter(self.description['arms'])))
