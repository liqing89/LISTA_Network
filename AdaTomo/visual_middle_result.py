import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('/home/amax/Wcx/wangcx/code/AdaTomo/middle_result/Ada-LISTA/Ada_LISTA_epoch16_output.csv')

# 选择CSV文件中特定的行和列（假设您想选择第2行到第5行和第3列到第6列）
selected_rows = slice(0, 10)  # 选择行（注意Python中索引是从0开始的）
selected_columns = slice( 1400, 1499)  # 选择列（注意Python中索引是从0开始的）
selected_data = df.iloc[selected_rows, selected_columns]
data_array = selected_data.values
plt.figure(figsize=(10, 6))
for i in range(data_array.shape[0]):
    plt.plot(data_array[i, :], label=f'Row {i+2}')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.title('Selected CSV Data Visualization')
plt.legend()
plt.grid()
plt.show()
plt.switch_backend('agg')
plt.savefig("batch_size1.jpg")