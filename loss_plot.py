import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
path_ = './model/proxy/eq/'
csv_file = path_ + 'losses.csv'
df = pd.read_csv(csv_file)

# Calculate the moving average with a window size of N
N = 100  # Adjust the window size as needed
df['loss1_moving_avg'] = df['MAE'].rolling(window=N).mean()
df['loss2_moving_avg'] = df['MRSTFT'].rolling(window=N).mean()

# Plot the moving average of both losses
plt.figure(figsize=(10, 6))
plt.plot(df['loss1_moving_avg'], label='MAE Moving Average')
plt.plot(df['loss2_moving_avg'], label='MRSTFT Moving Average')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Moving Average of Losses')
plt.legend()

# Save the plot as an image file
output_file = path_+ 'losses_moving_average_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
