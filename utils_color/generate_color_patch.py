"""
 生成颜色色块图像进行保存,生成方式,BGR 三种颜色进行遍历
 256/16=16 1/16=0.0625 这个值可以用于在梯度更新时候进行判断,以防到了后期由于量化的误差大于梯度的更新误差导致 颜色不更新
 理论上而言 颜色越多 颜色的准确度越高 但是同时也会带来内存的压力 和取色的压力
 以16为间隔,一共可以生成16*16*16=4096 
 21*29.7cm 为A4纸张的大小 按照0.5cm 和 0.9cm尺寸为一个patch进行打印 总共一张值1386种颜色 无边框打印
 1024就够了一张

 如果以32为间隔,则总共8*8*8=512种颜色,依旧是会超出间隔的

"""
import cv2
import numpy as np
import os 
save_path='/data1/yjt/adversarial_attack/myattack/'

# total=[]
# image_hang=[]

# for B in range(0,256,16):
#     for G in range(0,256,16):
#         for R in range(0,256,16):
#             BGR=np.array([[[B,G,R]]],dtype=np.uint8)
#             # print(BGR.shape)
#             BGR=np.tile(BGR,(36,20,1)) #10 18 像素值的图像
#             # print(BGR.shape)
#             # print(BGR)
#             total.append(BGR)
# total=np.asarray(total)  #4096 36 20 3 
# print(total.shape)
# total=np.concatenate(total,axis=0) #147456 20 3 
# print(total.shape)

# res=[]
# for i in range(0,147456,1152):
#     res.append(total[i:i+1152,:,:])
# res=np.asarray(res)
# print(res.shape) # 128 1152 20 3 
# # print(hang.shape)


# for i in range(4):
#     image=res[32*i:32*i+32,:,:]
#     image=np.concatenate(image,axis=1)
#     print(image.shape)
#     cv2.imwrite(save_path+str(i)+'.png',image)


"""
下面是反过来读取颜色的代码，用于进行颜色的映射构建
"""
img=cv2.imread('/data1/yjt/adversarial_attack/myattack/3.png')
file = open('/data1/yjt/adversarial_attack/myattack/3.txt', 'w') #保存颜色值的文件

h,w,c=img.shape
h_step=round(h/32)
w_step=round(w/32)
print(h_step,w_step)

start_w = round(w/32/2)
start_h = round(h/32/2)


for center_h in range(start_h,h,h_step):
    for center_w in range(start_w,w,w_step):

        file.write(str(img[center_h,center_w,2]/255.0)+','+str(img[center_h,center_w,1]/255.0)+','+str(img[center_h,center_w,0]/255.0)+'\n') # RGB 颜色值保存
        # 进行标注，将所在地方转换为全白的像素表明在这个地方采样 用于检查对应的颜色值
        for i in range (center_h-2,center_h+2):
            for j in range ( center_w-2,center_w+2):
                img[i,j,0]=255
                img[i,j,1]=255
                img[i,j,2]=255


cv2.imwrite(save_path+'3_biaozhu.png',img)
file.close()