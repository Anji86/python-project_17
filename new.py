import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Objective 1: Data Loading and Preprocessing
data = pd.read_csv('dataset.csv')  # Use read_excel for Excel files

# Objective 2: Exploratory Data Analysis (EDA)
print("\n===== Exploratory Data Analysis =====")
print("\nDataset Dimensions:", data.shape)
print("\nColumn Names:", list(data.columns))

# Objective 3: Data Quality Assessment
print("\nData Types and Missing Values:")
print(data.info(show_counts=True))
print("\nNumerical Columns Statistics:")
print(data.describe().round(2))

# Objective 4: Categorical Data Analysis
categorical_cols = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns Summary:")
for col in categorical_cols:
    print(f"\n{col} - Unique Values:", data[col].nunique())
    print(data[col].value_counts().head())
    
# Objective: Histogram for Numerical Columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    plt.hist(data[col].dropna(), bins=30, color='green', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()

# Objective 5: Outlier Detection
print("\nOutlier Analysis:")
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = len(data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))])
    print(f"{col} - Number of outliers: {outliers}")
# Objective: Shipping Service Level Impact on Sales
if 'ship_service_level' in data.columns and 'Gross_sales' in data.columns:
    shipping_impact = data.groupby('ship_service_level')['Gross_sales'].mean().sort_values(ascending=False)
    print("\nShipping Impact on Average Sales:\n", shipping_impact.round(2))

    # Bar plot for visual analysis
    plt.figure(figsize=(10, 6))
    shipping_impact.plot(kind='bar', color='skyblue')
    plt.title('Average Gross Sales by Shipping Service Level', fontsize=14)
    plt.xlabel('Shipping Service Level', fontsize=12)
    plt.ylabel('Average Gross Sales (INR)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Objective 6: Sales and Revenue Analysis
if 'Gross_sales' in data.columns and 'Net_quantity' in data.columns:
    corr_matrix = data[['Gross_sales', 'Net_quantity']].corr()
    print("\nCorrelation Matrix:\n", corr_matrix.round(3))

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.3f')
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Net_quantity', y='Gross_sales', data=data, scatter_kws={'alpha': 0.3})
    plt.title('Sales Volume vs Revenue', fontsize=14)
    plt.xlabel('Quantity Sold', fontsize=12)
    plt.ylabel('Gross Sales (INR)', fontsize=12)
    plt.show()

# Objective 7: Product Category Performance
if 'Category' in data.columns:
    category_sales = data.groupby('Category')['Gross_sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    category_sales.plot(kind='bar', color='skyblue')
    plt.title('Sales by Category', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Total Sales (INR)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Objective 8: Payment Method Analysis
if 'Payment_Mode' in data.columns:
    payment_mode = data['Payment_Mode'].value_counts()
    plt.figure(figsize=(8, 8))
    payment_mode.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Payment Mode Distribution', fontsize=14)
    plt.ylabel('')
    plt.show()
# Objective: Sales Funnel Analysis - Vertical Bar Chart
if 'Returns' in data.columns and 'ship_service_level' in data.columns:
    stages = ['Total Orders', 'Successful Deliveries', 'Premium Shipping']
    values = [
        len(data),
        len(data[data['Returns'] == 0]),
        len(data[data['ship_service_level'] == 'Premium'])
    ]

    # Vertical Funnel Chart
    plt.figure(figsize=(8, 6))
    plt.bar(stages, values, color=['#2ecc71', '#3498db', '#9b59b6'])
    plt.title('Sales Funnel Analysis', fontsize=14)
    plt.ylabel('Number of Orders', fontsize=12)
    plt.xlabel('Funnel Stage', fontsize=12)
    for i, v in enumerate(values):
        plt.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Funnel Conversion Rates
    print("\nFunnel Conversion Rates:")
    for i in range(len(stages) - 1):
        conversion = (values[i + 1] / values[i]) * 100
        print(f"{stages[i]} → {stages[i + 1]}: {conversion:.1f}%")

# Objective 9: Returns and Shipping Performance
if 'Returns' in data.columns and 'ship_service_level' in data.columns:
    returns_percentage = (data['Returns'].sum() / len(data)) * 100
    shipping_impact = data.groupby('ship_service_level')['Gross_sales'].mean().sort_values(ascending=False)
    print(f"\nReturns Percentage: {returns_percentage:.2f}%")
    print("\nShipping Impact on Average Sales:\n", shipping_impact)

# Objective 10: Time Series Analysis
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.dropna(subset=['Date'], inplace=True)
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    yearly_sales = data.groupby('Year')['Gross_sales'].sum().sort_values(ascending=False)
    monthly_sales = data.groupby('Month')['Gross_sales'].mean().sort_index()

    plt.figure(figsize=(10, 6))
    yearly_sales.plot(kind='bar', color='orange')
    plt.title('Yearly Gross Sales', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Gross Sales')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    monthly_sales.plot(marker='o', linestyle='--', color='green')
    plt.title('Average Monthly Sales', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.show()

# Objective 11: Geographic Distribution
if 'State' in data.columns:
    state_sales = data.groupby('State')['Gross_sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    state_sales.plot(kind='bar', color='lightgreen')
    plt.title('Sales by State', fontsize=14)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Total Sales (INR)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Objective 12: Sales Funnel Metrics
if 'Returns' in data.columns and 'ship_service_level' in data.columns:
    stages = ['Total Orders', 'Successful Deliveries', 'Premium Shipping']
    values = [
        len(data),
        len(data[data['Returns'] == 0]),
        len(data[data['ship_service_level'] == 'Premium'])
    ]

    # Horizontal Funnel Chart
    plt.figure(figsize=(10, 8))
    plt.barh(stages, values, color=['#2ecc71', '#3498db', '#9b59b6'])
    plt.title('Sales Funnel Analysis', fontsize=14)
    plt.xlabel('Number of Orders', fontsize=12)
    plt.gca().invert_yaxis()
    for i, v in enumerate(values):
        plt.text(v, i, f' {v:,}', va='center', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Funnel Conversion Rates
    print("\nFunnel Conversion Rates:")
    for i in range(len(stages) - 1):
        conversion = (values[i + 1] / values[i]) * 100
        print(f"{stages[i]} → {stages[i + 1]}: {conversion:.1f}%")

# Optional: Pairplot
if all(col in data.columns for col in ['Gross_sales', 'Net_quantity', 'Returns']):
    sns.pairplot(data[['Gross_sales', 'Net_quantity', 'Returns']], diag_kind='kde')
    plt.suptitle('Multi-variable Analysis', y=1.00, fontsize=12)
    plt.show()
