import streamlit as st
from PIL import Image
import numpy as np
from load_dataset import c2w_to_rays
import torch
from render import render_rays
from scipy.spatial.transform import Rotation as R

import streamlit.components.v1 as components

x = st.slider('', -7.0, 7.0, 0.0,step=0.1,key="a")
y = st.slider('', -7.0, 7.0, 0.0,step=0.1,key="b")
z = st.slider('', -7.0, 7.0, 0.0,step=0.1,key="c")
rx = st.slider('', -180.0, 180.0, 0.0,step=0.1,key="d")
ry = st.slider('', -180.0, 180.0, 0.0,step=0.1,key="e")
rz = st.slider('', -180.0, 180.0, 0.0,step=0.1,key="f")


device = "cuda"
image_res = 200
model = torch.load("model")

r = R.from_euler('x', rx, degrees=True).as_matrix()
r = R.from_euler('y', ry, degrees=True).as_matrix() @ r
r = R.from_euler('z', rz, degrees=True).as_matrix() @ r

c2w = np.eye(4)
c2w[:3,:3] = r
c2w = c2w @ np.array([
    [1,0,0,x],
    [0,1,0,y],
    [0,0,1,z],
    [0,0,0,1]
])
origins1, dirs1 = c2w_to_rays(c2w,200,200)
origins1 = torch.FloatTensor(origins1).to(device)
dirs1 = torch.FloatTensor(dirs1).to(device)

print("a")
C = render_rays((origins1, dirs1), model, device=device)
print("b")
out_img = C.detach().cpu().numpy().reshape((image_res, image_res, 3))
image = Image.fromarray(np.uint8(np.clip((out_img) * 255, 0, 255))).resize((400,400))

st.image(image)
