import matplotlib.pyplot as plt

marks = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
         55, 60, 65, 70, 7, 80, 85, 90, 95, 100]

plt.hist(marks, bins=5, edgecolor='black')

plt.title("Histogram of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")

plt.grid(True)

plt.show()
