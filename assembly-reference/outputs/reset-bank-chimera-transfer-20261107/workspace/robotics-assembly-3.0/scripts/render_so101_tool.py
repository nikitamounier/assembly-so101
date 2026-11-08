"""Render the actual shared HOTO asset and native control/bit/USB detail views."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from assembly_policy.so101 import scene
from assembly_policy.so101.tool import add_driver
from assembly_policy.so101.workcell import axes,numbers


def render(output):
    scene.load_plugin();root=ET.Element('mujoco')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',gravity='0 0 0',timestep='.0005')
    asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.10 .13 .16',rgb2='.035 .045 .055',width='256',height='256')
    world=ET.SubElement(root,'worldbody');extension=ET.SubElement(root,'extension');contact=ET.SubElement(root,'contact')
    ET.SubElement(world,'light',pos='-.3 .4 .5',dir='.3 -.4 -.5',diffuse='.85 .85 .85',specular='.8 .8 .8')
    ET.SubElement(world,'light',pos='.3 -.2 .3',dir='-.3 .2 -.3',diffuse='.45 .50 .55')
    add_driver(world,extension,contact,asset=asset,position=(0,0,.12),quaternion=(0,0,1,0))
    cameras={
        'whole':((.065,.26,.14),(0,0,.106),(0,0,1),.235),
        'controls':((0,.16,.1115),(0,0,.1115),(1,0,0),.030),
        'bit':((0,.075,-.025),(0,0,.025),(1,0,0),.040),
        'usb':((.015,.055,.28),(0,0,.2067),(0,1,0),.031),
    }
    for name,(position,target,up,height) in cameras.items():
        ET.SubElement(world,'camera',name=name,pos=numbers(position),xyaxes=numbers(axes(position,target,up)),projection='orthographic',fovy=str(height))
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1400',offheight='1000')
    ET.SubElement(visual,'headlight',ambient='.4 .4 .4',diffuse='.4 .4 .4')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    def view(name,width,height):
        renderer=mujoco.Renderer(m,width=width,height=height)
        try:
            renderer.update_scene(d,camera=name,scene_option=option)
            return Image.fromarray(renderer.render())
        finally:renderer.close()
    canvas=Image.new('RGB',(1400,1000),'#101a22');canvas.paste(view('whole',440,900),(0,70))
    canvas.paste(view('controls',900,300),(475,130))
    canvas.paste(view('bit',435,380),(475,505));canvas.paste(view('usb',435,380),(940,505))
    def font(size):
        try:return ImageFont.truetype('DejaVuSans.ttf',size)
        except OSError:return ImageFont.load_default(size=size)
    draw=ImageDraw.Draw(canvas)
    for text,xy,size,colour in (
        ('HOTO SNAPBLOQ S-A01',(25,20),30,'white'),
        ('Shared native simulation model',(25,58),18,'#aac3d1'),
        ('RECESSED ROCKER · TORQUE SELECTOR · THREE INDICATORS',(490,100),20,'white'),
        ('C4 BIT · COLLET · LED RING',(490,468),20,'white'),
        ('REAR USB-C PORT',(955,468),20,'white'),
        ('181.7 × 19.7 × 22.5 mm · 105 g',(475,925),25,'white'),
        ('Published tool dimensions/mass; installed bit shown. Fine features reconstructed from official photos.',(475,963),15,'#aac3d1')):
        draw.text(xy,text,font=font(size),fill=colour)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=scene.ROOT/'docs/images/hoto-s-a01.png')
    a=p.parse_args();print(render(a.out))


if __name__=='__main__':main()
