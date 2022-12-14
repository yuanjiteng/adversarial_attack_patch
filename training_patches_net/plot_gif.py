import imageio
import os
# def compose_gif():
#     img_paths = ["img/1.jpg","img/2.jpg","img/3.jpg","img/4.jpg"
#     ,"img/5.jpg","img/6.jpg"]
#     gif_images = []
#     for path in img_paths:
#         gif_images.append(imageio.imread(path))
#     imageio.mimsave("test.gif",gif_images,fps=1)

# 对于一个文件夹下面的所有文件进行
def compose_gif():
    gif_images=[]
    path='/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/runs/detect/exp126/successed_predict/'
    files=os.listdir(path)
    for file in files:
        img_path=path+file
        gif_images.append(imageio.imread(img_path))
    imageio.mimsave("successed_126.gif",gif_images,fps=1)
    print("finished!")


if __name__=='__main__':
    compose_gif()
    # gif_images=[]
    # for i in range(175):
    #     path='/data1/yjt/adversarial_attack/myattack/training_patches/'+str(i)+' patch.png'
    #     gif_images.append(imageio.imread(path))
    # imageio.mimsave("res_net.gif",gif_images,fps=5)
    # print("finished!")