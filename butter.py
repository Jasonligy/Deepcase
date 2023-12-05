from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
x=[0.05*i*np.pi for i in range(100)]
y=[0.5*i*np.pi for i in range(100)]
data_x=np.sin(x)
data_y=np.sin(y)
data_z=data_x+data_y

data=np.arange(100)


print(len(y))
b, a = signal.butter(1, 0.5, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
data_f = signal.lfilter(b, a, data_z)  #data为要过滤的信号
data_f = signal.filtfilt(b, a, data_z)  #data为要过滤的信号
# print(filtedData)
plt.plot(data_f)
plt.show()