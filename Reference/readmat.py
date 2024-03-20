import scipy.io as scio
#data_mat = "C://Users//sunyu//Desktop//fake_and_real_peppers_ms.mat"
data_mat = "C://Users//sunyu//Desktop//CAVE//1.mat"
data = scio.loadmat(data_mat)
print(type(data))
print(data.keys())
print(data.get('__header__'))
print(data.get('__version__'))
print(data.get('__global__'))
print(type(data.get('HSI')))
print((data.get('HSI')).size)