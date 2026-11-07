"""Small deterministic meshes for the photo-reconstructed HOTO tool, in metres."""
import numpy as np


def lathe(profile, segments=96):
    """Closed axial profile: (z, radius) rings with outward winding."""
    angle=np.arange(segments)*2*np.pi/segments
    vertices=np.concatenate([np.column_stack((r*np.cos(angle),r*np.sin(angle),np.full(segments,z))) for z,r in profile])
    faces=[]
    for ring in range(len(profile)-1):
        for i in range(segments):
            j=(i+1)%segments;a=ring*segments+i;b=ring*segments+j;c=b+segments;e=a+segments
            faces.extend(((a,b,c),(a,c,e)))
    for offset,reverse in ((0,True),((len(profile)-1)*segments,False)):
        for i in range(1,segments-1):
            face=(offset,offset+i,offset+i+1)
            faces.append(face[::-1] if reverse else face)
    return vertices,np.asarray(faces)


def capsule_plate(width,length,thickness,segments=24):
    """Rounded capsule on the XZ plane, thickness along local Y."""
    radius=width/2;span=(length-width)/2
    a=np.linspace(0,np.pi,segments+1);b=np.linspace(np.pi,2*np.pi,segments+1)
    xz=np.vstack((np.column_stack((radius*np.cos(a),span+radius*np.sin(a))),
                  np.column_stack((radius*np.cos(b),-span+radius*np.sin(b)))))
    n=len(xz);v=np.vstack((np.column_stack((xz[:,0],np.full(n,-thickness/2),xz[:,1])),
                         np.column_stack((xz[:,0],np.full(n,thickness/2),xz[:,1]))))
    f=[]
    for i in range(n):
        j=(i+1)%n;f.extend(((i,i+n,j+n),(i,j+n,j)))
    for i in range(1,n-1):f.extend(((0,i,i+1),(n,n+i+1,n+i)))
    return v,np.asarray(f)


def rocker_half(width,length,thickness,reverse=False):
    r=width/2;a=np.linspace(0,np.pi,25)
    xz=np.vstack(([[r,-length/2]],np.column_stack((r*np.cos(a),length/2-r+r*np.sin(a))),[[-r,-length/2]]))
    n=len(xz);v=np.vstack((np.column_stack((xz[:,0],np.full(n,-thickness/2),xz[:,1])),
                         np.column_stack((xz[:,0],np.full(n,thickness/2),xz[:,1]))))
    f=[]
    for i in range(n):
        j=(i+1)%n;f.extend(((i,i+n,j+n),(i,j+n,j)))
    for i in range(1,n-1):f.extend(((0,i,i+1),(n,n+i+1,n+i)))
    if reverse:v*=np.array([-1,1,-1])
    return v,np.asarray(f)


def capsule_frame(width,length,border,thickness):
    outer,_=capsule_plate(width,length,thickness);inner,_=capsule_plate(width-2*border,length-2*border,thickness)
    # Equal capsule centres/ring counts give an annular extrusion without fills.
    n=len(outer)//2;v=np.vstack((outer,inner));f=[]
    for i in range(n):
        j=(i+1)%n
        for p,q,r,s in ((i,j,j+n,i+n),(i+2*n,i+3*n,j+3*n,j+2*n),
                        (i,i+2*n,j+2*n,j),(i+n,j+n,j+3*n,i+3*n)):
            f.extend(((p,q,r),(p,r,s)))
    return v,np.asarray(f)[:,::-1]


def shell(rings,rx=.00985,ry=.011,segments=192,fluted=False,capsule_recess=False):
    """Elliptical casing, including a real recess for the physical controls.

    Each ring is (z, scale, front_y_limit); collision uses convex axial pieces.
    Fine fluting belongs to the surface itself, avoiding overlapping decals.
    """
    a=np.arange(segments)*2*np.pi/segments;vertices=[]
    for z,scale,clip in rings:
        factor=np.ones(segments)*scale
        if fluted and z>=.030:
            factor-=np.where(np.sin(a)<.79,.00005/rx*(.5-.5*np.cos(64*a)),0.)
        x=rx*np.cos(a)*factor;y=ry*np.sin(a)*factor
        if clip is not None:
            if capsule_recess:
                dz=max(abs(z-.0085)-.0327,0.)
                half=np.sqrt(max(.0052**2-dz**2,0.))
                y=np.where((np.abs(x)<half)&(y>clip),clip,y)
            else:y=np.minimum(y,clip)
        vertices.extend(np.column_stack((x,y,np.full(segments,z))))
    f=[]
    for k in range(len(rings)-1):
        for i in range(segments):
            j=(i+1)%segments;p=k*segments+i;q=k*segments+j
            f.extend(((p,q,q+segments),(p,q+segments,p+segments)))
    for offset,reverse in ((0,True),((len(rings)-1)*segments,False)):
        for i in range(1,segments-1):
            face=(offset,offset+i,offset+i+1);f.append(face[::-1] if reverse else face)
    return np.asarray(vertices),np.asarray(f)


def ring(z,inner,outer,thickness,segments=96):
    # A thin true annulus, so the central chuck remains visible.
    a=np.arange(segments)*2*np.pi/segments
    v=np.concatenate([np.column_stack((r*np.cos(a),r*np.sin(a),np.full(segments,h)))
                      for h,r in ((z-thickness/2,inner),(z-thickness/2,outer),
                                  (z+thickness/2,outer),(z+thickness/2,inner))])
    f=[]
    for k in range(4):
        next_k=(k+1)%4
        for i in range(segments):
            j=(i+1)%segments;p=k*segments+i;q=k*segments+j;r=next_k*segments+j;s=next_k*segments+i
            f.extend(((p,q,r),(p,r,s)))
    return v,np.asarray(f)


def tapered_wing(hx,hy,z0,z1):
    v=np.array([(x*hx*scale,y*hy*scale,z) for z,scale in ((z0,1),(z1,.32))
                for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))])
    f=np.array([(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
                (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)])
    return v,f


def merge(parts):
    vertices=[];faces=[];offset=0
    for v,f in parts:
        vertices.append(v);faces.append(f+offset);offset+=len(v)
    return np.concatenate(vertices),np.concatenate(faces)


def stroke_mesh(lines,radius=.00010):
    """Short geometric strokes for permanent control icons; no font/texture file."""
    parts=[]
    for start,end in lines:
        start,end=np.asarray(start,float),np.asarray(end,float);direction=end-start;length=np.linalg.norm(direction)
        z=direction/length;x=np.cross([0,1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        v,f=lathe([(-length/2,radius),(length/2,radius)],segments=6)
        parts.append((v@np.column_stack((x,y,z)).T+(start+end)/2,f))
    return merge(parts)
