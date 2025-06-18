from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from RayTracing import Surface, RayTracing
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def Hello():
    return {"Hello":"!!!!!!fdasfsdafa!!!!!!!!!!!!!"}

class Message(BaseModel):
    enviType: int
    N: int
    tpX: int
    tpY: int
    tpZ: int
    rpX: int
    rpY: int
    rpZ: int

@app.post("/send")
def send_message(msg: Message):
    try:
        surfs = Surface(msg.enviType)
        points, _, surfs_list_idxs = surfs.points, surfs.surf_list, surfs.surf_list_idxs
        tp = np.array([msg.tpX, msg.tpY, msg.tpZ])
        rp = np.array([msg.rpX, msg.rpY, msg.rpZ])
        N = msg.N
        ante_polor = 'TE'
        rt = RayTracing(N=N, surfs=surfs.tri_surf, t_p=tp, r_p=rp, ante_polor=ante_polor)
        rt.do_ray_tracing()
        return {
            "points": points.tolist(),
            # "surfsList": surfs_list,
            "surfsListIdxs": surfs_list_idxs
        }
    except Exception as e:
        return {"error": str(e)}
