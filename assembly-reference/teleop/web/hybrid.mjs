// Display policy only. Phone pixels have no robot-control authority.
export const defaultAlignment=()=>({opacity:.45,scale:1,x:0,y:0});
export function validAlignment(value){return !!value&&Number.isFinite(value.opacity)&&value.opacity>=0&&value.opacity<=1&&Number.isFinite(value.scale)&&value.scale>=.5&&value.scale<=2&&['x','y'].every(k=>Number.isFinite(value[k])&&Math.abs(value[k])<=.3);}
export function editAlignment(previous,patch,locked){const next={...previous,...patch};return !locked&&validAlignment(next)?next:previous;}
export function containImage(width,height,planeWidth,planeHeight){
 if(![width,height,planeWidth,planeHeight].every(v=>Number.isFinite(v)&&v>0))throw Error('Positive image and plane dimensions required');
 const scale=Math.min(planeWidth/width,planeHeight/height);
 return {width:width*scale,height:height*scale,aspectMismatch:Math.abs((width/height)/(planeWidth/planeHeight)-1)>.02};
}
export function phoneFresh(sample,receivedMs,roundtripMs,now){
 if(!sample?.available||sample.sensor_time_known!==false||!Number.isInteger(sample.generation)||!Number.isInteger(sample.sequence)||!Number.isFinite(sample.age_upper_ms)||sample.age_upper_ms<0)return false;
 if(![receivedMs,roundtripMs,now].every(Number.isFinite)||roundtripMs<0||now<receivedMs)return false;
 return sample.age_upper_ms+roundtripMs+now-receivedMs<500;
}
export function nativeCaptureFresh(acquiredMs,offsetUpperMs,now){
 return [acquiredMs,offsetUpperMs,now].every(Number.isFinite)&&now+offsetUpperMs>=acquiredMs&&now+offsetUpperMs-acquiredMs<500;
}
export function hybridViewState({selected,phoneReady,nativeReady,suspended=false,refreshUntil=null,now}){
 const waiting=suspended||(Number.isFinite(refreshUntil)&&now<refreshUntil);
 return {background:!!selected&&!!phoneReady,overlay:!!selected&&!!phoneReady&&!!nativeReady&&!suspended,
         hold:!!selected&&(suspended||!phoneReady||!nativeReady),drop:!!selected&&!phoneReady&&!waiting};
}
export class PhoneGuard{
 constructor(){this.token=0;this.generation=-1;this.sequence=-1;}
 begin(){return ++this.token;}
 invalidate(){this.token++;}
 accept(token,sample,receivedMs,roundtripMs,now){
  if(token!==this.token||!phoneFresh(sample,receivedMs,roundtripMs,now)||typeof sample.jpeg!=='string'||!sample.jpeg.startsWith('data:image/jpeg;base64,'))return false;
  if(sample.generation<this.generation||sample.generation===this.generation&&sample.sequence<this.sequence)return false;
  this.generation=sample.generation;this.sequence=sample.sequence;return true;
 }
}
