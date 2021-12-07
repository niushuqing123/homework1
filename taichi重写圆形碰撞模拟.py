from numpy.ma.core import shape
import taichi as ti
import numpy as np
from taichi.lang.ops import sqrt

ti.init(arch=ti.gpu)

#可调节的窗口的尺寸，不影响效果
# screen_res = (800, 600)
# screen_res = (1000, 200)
screen_res = (400, 900)


max_particles=20000
n=ti.field(dtype=int, shape=())

x = ti.Vector.field(2, dtype=float, shape=max_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=max_particles)  # velocity

r = ti.field(dtype=float, shape=max_particles)  
m = ti.field(dtype=float, shape=max_particles) #质量等于1*半径，并不是很符合物理，但能表示质量随半径增大而增大。如果用0.5*pai*r*r的话会有问题，感觉是受杨氏模量和dt限制了


# 可调节的，与边界碰撞的阻尼,[0-1)
damping=0.1

colors = np.array([0x068587, 0xED553B, 0xEEEEF0,0xFF00FF,0xFFFF00], dtype=np.uint32)
color = np.zeros(shape=max_particles)
for i in  range(max_particles):
    color[i]=colors[i%colors.size]


#以后可以升级成3d版，做一些与其他物理模型的耦合

@ti.kernel
def solve():
    dt=0.001
    Young_modulus=2000
    for i in range(n[None]):
        for j in range(i+1,n[None]):
            direction_vector=x[j]-x[i]
            direction_vector_length=sqrt(direction_vector[0]**2+direction_vector[1]**2)

            if (direction_vector_length<=r[i] + r[j]):
                elastic_force=Young_modulus*(direction_vector/direction_vector_length)*(r[i]+r[j]-direction_vector_length)
                
                elastic_damping = (v[i] - v[j]).dot(direction_vector/direction_vector_length)  # 两个粒子速度之差的模长大小（点积）；相对速度在两粒子方向上的投影。

                v[i] += -elastic_damping*10 * ((direction_vector/direction_vector_length)/m[i])*dt
                v[j] -= -elastic_damping*10 * ((direction_vector/direction_vector_length)/m[j])*dt
                
                v[i]-=(elastic_force/m[i])*dt
                v[j]+=(elastic_force/m[j])*dt


    for i in range(n[None]):
        if(x[i][0]<r[i]):v[i][0]+=(Young_modulus*(r[i]-x[i][0])-damping*v[i][0])*dt
        if(x[i][1]<r[i]):v[i][1]+=(Young_modulus*(r[i]-x[i][1])-damping*v[i][1])*dt
        if(x[i][0]+r[i]>screen_res[0]):v[i][0]+=(Young_modulus*(screen_res[0]-x[i][0]-r[i])-damping*v[i][0])*dt
        if(x[i][1]+r[i]>screen_res[1]):v[i][1]+=(Young_modulus*(screen_res[1]-x[i][1]-r[i])-damping*v[i][1])*dt

        

    for i in range(n[None]):
        v[i][1]-=9.8*dt
        x[i]+=v[i]*dt
        
        
# @ti.func #add_matrix用kernel的时候add用func
@ti.kernel
def add(pos_x: ti.f32, pos_y: ti.f32, r1: ti.f32,vx:ti.f32,vy:ti.f32):
    num=n[None]
    x[num] = ti.Vector([pos_x, pos_y])  # 将新粒子的位置存入x
    r[num]=r1
    v[num]=ti.Vector([vx, vy])
    m[num]=r1
    n[None] += 1  # 粒子数量加一


# @ti.kernel#gpu运行不能用kernel，cpu可以
def add_matrix(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(6):
        for j in range(6):
            add(pos_x+i*20,pos_y+j*20,10,0,0)
# @ti.kernel#gpu运行不能用kernel
def add_tiny(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(6):
        for j in range(6):
            add(pos_x+i*4,pos_y+j*4,2,0,0)

@ti.kernel
def scale(k:ti.i32):
    if r[n[None]-1]<200 and (r[n[None]-1]>4 or k>0):
        r[n[None]-1]+=k
        m[n[None]-1]=r[n[None]-1]

@ti.kernel
def reset():
    for i in range(n[None]):
        x[i] =[0, 0]
        r[i]=0
        v[i]=[0, 0]
        m[i]=0
    n[None]=0

def main():
    gui = ti.GUI("Physical circular", res=screen_res, background_color=0xFFDEAD)
    p=-1
    add(screen_res[0]/2,screen_res[1]/2,50,0,0)
            
    while True:

        R=r.to_numpy()
        X = x.to_numpy()

        for e in gui.get_events(ti.GUI.PRESS):  
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == ti.GUI.LMB:  # 鼠标左键，增加粒子
                add(e.pos[0]*screen_res[0], e.pos[1]*screen_res[1],10,0,0)
            elif gui.is_pressed('w'):#对上一个圆进行缩放
                scale(4)
            elif gui.is_pressed('s'):  
                scale(-4)
            elif gui.is_pressed('p'):#左下角喷射开关
                p*=-1
            elif gui.is_pressed('h'):#添加一堆圆
                add_matrix(e.pos[0]*screen_res[0],e.pos[1]*screen_res[1])
            elif gui.is_pressed('y'):#添加一堆微小的圆
                add_tiny(e.pos[0]*screen_res[0],e.pos[1]*screen_res[1])
            elif gui.is_pressed('r'):#清空
                reset()
        if p==1:
            add(0.1, 0.8,10,10,0)
            

        for i in range(n[None]):
            X[i]/=screen_res

        for i in range(30):
            solve()
 

        gui.circles(X, color=color,radius=R)

        gui.show()


if __name__ == '__main__':
    main()
