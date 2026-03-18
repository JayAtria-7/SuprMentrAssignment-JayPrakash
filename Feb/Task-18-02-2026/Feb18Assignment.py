'''
today's task:

sales analysis report

create own dataset
columns: month,revenue,marketing spend,profit

task:
create line plot for revenue trend
create scatter pot btn marketing, profit
create correlation map
write insights from your graph
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Revenue": [200, 220, 250, 270, 300, 320],
    "Marketing_Spend": [50, 60, 55, 65, 70, 75],
    "Profit": [80, 90, 110, 120, 140, 150]
}

df = pd.DataFrame(data)


plt.figure()
plt.plot(df["Month"], df["Revenue"])
plt.title("Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()


plt.figure()
plt.scatter(df["Marketing_Spend"], df["Profit"])
plt.title("Marketing Spend vs Profit")
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")
plt.show()


plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()


print("Insights:")
print("1. Revenue shows an increasing trend over the months.")
print("2. Higher marketing spend is associated with higher profit.")
print("3. Revenue and Profit are strongly positively correlated.")