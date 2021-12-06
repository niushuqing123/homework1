from numpy.ma.core import shape
import taichi as ti
import numpy as np
from taichi.lang.ops import sqrt

ti.init()

#可调节的窗口的尺寸，不影响效果

screen_res = (800, 600)
# screen_res = (1000, 200)
# screen_res = (400, 900)



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



@ti.kernel
def solve():
    dt=0.001
    杨氏模量=200
    for i in range(n[None]):
        for j in range(i+1,n[None]):
            法向量=x[j]-x[i]
            法向量模长=sqrt(法向量[0]**2+法向量[1]**2)
            
            # print(法向量)

            if (法向量模长<=r[i] + r[j]):
                弹性力=杨氏模量*(法向量/法向量模长)*(r[i]+r[j]-法向量模长)
                
                v_rel = (v[i] - v[j]).dot(法向量/法向量模长)  # 两个粒子速度之差的模长大小（点积）；相对速度在两粒子方向上的投影。

                v[i] += -v_rel*10 * ((法向量/法向量模长)/m[i])*dt
                v[j] -= -v_rel*10 * ((法向量/法向量模长)/m[j])*dt
                

                # v[i] += -v_rel*1 * ((法向量/法向量模长))*dt
                # v[j] -= -v_rel*1 * ((法向量/法向量模长))*dt
                

                v[i]-=(弹性力/m[i])*dt
                v[j]+=(弹性力/m[j])*dt

                # v[i]-=(弹性力)*dt
                # v[j]+=(弹性力)*dt



    for i in range(n[None]):
        if(x[i][0]<r[i]):v[i][0]+=(杨氏模量*(r[i]-x[i][0])-damping*v[i][0])*dt
        if(x[i][1]<r[i]):v[i][1]+=(杨氏模量*(r[i]-x[i][1])-damping*v[i][1])*dt
        if(x[i][0]+r[i]>screen_res[0]):v[i][0]+=(杨氏模量*(screen_res[0]-x[i][0]-r[i])-damping*v[i][0])*dt
        if(x[i][1]+r[i]>screen_res[1]):v[i][1]+=(杨氏模量*(screen_res[1]-x[i][1]-r[i])-damping*v[i][1])*dt

        

    for i in range(n[None]):
        v[i][1]-=9.8*dt
        x[i]+=v[i]*dt
        
@ti.kernel
def add(pos_x: ti.f32, pos_y: ti.f32, r1: ti.f32,vx:ti.f32,vy:ti.f32):
    num=n[None]
    x[n[None]] = ti.Vector([pos_x, pos_y])  # 将新粒子的位置存入x
    r[n[None]]=r1
    v[n[None]]=ti.Vector([vx, vy])
    m[n[None]]=r1
    n[None] += 1  # 粒子数量加一


@ti.kernel
def scale(k:ti.i32):
    if r[n[None]-1]<200 and r[n[None]-1]>4:r[n[None]-1]+=k
    m[n[None]-1]=r[n[None]-1]



def add_(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(6):
        for j in range(6):
            add(pos_x+i*20,pos_y+j*20,10,0,0)
   

def main():
    gui = ti.GUI("test", res=screen_res, background_color=0xFFDEAD)
    p=-1
    # for i in range(5):
        # add(50*i,50*i,10,0,0)

    # add(100,100,50,0,0)
    # add(400,400,50,-100,50)
    # add(400,500,50,0,0)
    add(400,500,50,0,0)
            
    while True:

        R=r.to_numpy()
        X = x.to_numpy()

        for e in gui.get_events(ti.GUI.PRESS):  # 此for只循环一次，为了分离用户操作
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == ti.GUI.LMB:  # 鼠标左键，增加粒子
                add(e.pos[0]*screen_res[0], e.pos[1]*screen_res[1],10,0,0)
            elif gui.is_pressed('w'):
                scale(4)
            elif gui.is_pressed('s'):  
                scale(-4)
            elif gui.is_pressed('p'):
                p*=-1
            elif gui.is_pressed('h'):
                add_(e.pos[0]*screen_res[0],e.pos[1]*screen_res[1])

        if p==1:
            add(0.1, 0.8,10,10,0)
            

        for i in range(n[None]):
            X[i]/=screen_res

        for i in range(100):
            solve()
 

        # colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
        # colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)


        # colors=colors[r.to_numpy()]
        gui.circles(X, color=color,radius=R)

        # print(X)
        # for i in range(n[None]):
            # gui.circle(X[i], color=0X888888, radius=R[i])

            # 给每个物体不同的颜色
            
            # gui.circle(X[i], color=colors[material.to_numpy()], radius=R[i])



            # gui.circle(X[i], color=i*1200, radius=R[i])
            # gui.circle(X[i], color=colors[i%colors.size], radius=R[i])

            # print(i*1200)

            # print(0x068587)
            
        # break
        gui.show()


if __name__ == '__main__':
    main()
