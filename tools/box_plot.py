import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Optional, but often convenient

# Assuming your data is organized like this:
data = {
    'H1': [99.5, 94.5, 93.6, 100.0, 100.0, 100.0, 95.9, 100.0, 100.0, 100.0, 95.5, 100.0, 85.0, 100.0, 99.5],
    'H2': [100.0, 100.0, 98.2, 100.0, 100.0, 93.6, 99.1, 100.0, 100.0, 100.0, 99.5, 90.5, 100.0, 100.0, 100.0],
    'H3': [97.7, 100.0, 97.7, 100.0, 100.0, 99.1, 100.0, 100.0, 100.0, 100.0, 100.0, 98.6, 100.0, 100.0, 100.0],
    'H4': [77.0, 99.1, 93.2, 80.9, 100.0, 84.1, 88.2, 100.0, 84.5, 97.3, 98.2, 99.1, 90.9, 97.7, 98.6],
    'H5': [90.0, 97.7, 90.9, 88.2, 99.5, 92.7, 85.5, 92.7, 77.3, 78.2, 86.8, 99.5, 97.7, 99.1, 87.3],
    'UNK': [61.8, 74.1, 67.7, 74.5, 78.6, 59.5, 58.2, 52.7, 60.5, 68.2, 45.9, 72.7, 60.0, 65.9, 75.5]
}
# Convert to DataFrame for easier plotting with Seaborn
df = pd.DataFrame(data)

# Create the box plot
plt.figure(figsize=(10, 6)) # Adjust figure size as needed
sns.boxplot(data=df)
plt.title('Distribution of Real-Time Accuracy Across 15 Subjects per Gesture Class')
plt.ylabel('Accuracy (%)')
plt.xlabel('Gesture Class')
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
#plt.savefig('accuracy_boxplot.png') # Optional: Save the figure
plt.show()