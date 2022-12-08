import torch 
"""
此文件用于尝试ste 的实现
"""

# matrix1=torch.randn(1,3,2,2) #patch 
# matrix2=torch.randn(4,3,2,2) #用于替换的值
# print(matrix1,matrix2)
# dist=matrix1-matrix2
# dist2=dist**2
# dist2= torch.sum(dist2,1) # 2 2 2 
# dist2=torch.sqrt(dist2) # 2 2 2 
# # 选择小的 
# val,indices=torch.min(dist2,0) #val 2 2  indices[2 2]里面的值代表选择哪些 相当于对于第一个值选择
# # print(val ,indices)
# # 只能得到四个值
# min2=torch.gather(dist2,0,indices.unsqueeze(0))
# # print(min2.shape,min2)
# indices2=indices.expand(3,2,2).unsqueeze(0)
# res= torch.gather(matrix2,0,indices2)
# print(res)
# # matrix3=matrix2[indices]
# print(matrix3.shape)
# 然而上面的东西不可导
# 所以需要进行后续处理？如果是输出结果之后进行量化呢？ 量化测试

# float32->量化-128 128->再进行变换？
# 两个方法，一个使用ste进行更新，一个不断更改 模型进行 只是需要一个映射关系



import torch



class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,another):

        return another 
        # 没问题 但是如果涉及到计算呢 就不行了？
        # return torch.sign(input)
    @staticmethod
    def backward(ctx, grad_output):
        grad_output=grad_output+1
        return grad_output,None

if __name__ == '__main__':

    sign = LBSign.apply
    
    params = torch.randn(4, requires_grad = True)
    another = torch.randn(4)
    print(params)                                                                        
    output = sign(params,another)

    loss = output.mean()
    loss.backward()
    #这样才是进行多次的梯度累加
    print(params)

    # 没有设置优化器
    print(output)
    print(params.grad)
    print(another.grad)
