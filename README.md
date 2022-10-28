对于yolov5代码的改变，将sys.append()代码进行注释，并在所有的文件前面增加

PyTorchYOLOv5. 用于引用，在进行模型加载时候增加代码在 experiment.py 

`with add_path('/data1/yjt/adversarial_attack/myattack/PyTorchYOLOv5/'):`

进行临时的sys使用，从而减少引入包的冲突



对于yolov3，不增加YOLOv3进入系统工作目录，保证所有找包的系统环境只有最外层的myattack文件夹一个，训练时候 在train中修改 sys.path 就行



attack_test.py 文件为测试模型使用
