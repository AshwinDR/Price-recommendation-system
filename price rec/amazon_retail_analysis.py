import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
%matplotlib inline
retail_data = pd.read_csv('/content/amazon.csv')
retail_data.head()
retail_data.tail()
retail_data.shape
retail_data.columns.values
retail_data.dtypes
retail_data.describe()
retail_data['B2B'].value_counts().plot(kind='bar', figsize=(8, 6))
plt.xlabel(" B2B Category", labelpad=14)
plt.ylabel("Count", labelpad=14)
plt.title("Count of B2B Category", y=1.02);
100*retail_data['B2B'].value_counts()/len(retail_data['B2B'])
retail_data['B2B'].value_counts()
retail_data.info()
retail_data1 = retail_data.copy()
retail_data1.head()
retail_data1['Date'] = pd.to_datetime(retail_data1['Date'])
retail_data1['ship-postal-code'] = retail_data1['ship-postal-code'].astype('category')
retail_data1.isnull().sum()
retail_data1['B2B'] = retail_data1['B2B'].astype(str)
retail_data1['B2B'] = retail_data1['B2B'].replace({'False':'B2C','True':'B2B'})
retail_data1.rename(columns = {'Order ID':'Order_ID','ship-service-level':'Shipping_type','ship-city':'Shipping_city',
                               'ship-state':'Shipping_state','ship-postal-code':'Pincode','fulfilled-by':'Fulfilled_by',
                               'Qty':'Quantity','Courier Status':'Courier_Status','ship-country':'Shipping_country',
                               'B2B':'Business_type','promotion-ids':'Promotion_ids','Status':'Order_status',
                               'Date':'Order_date'},inplace = True)
retail_data1
retail_data1[retail_data1['Pincode'].isnull()]
columns_to_check = ['Pincode','Shipping_city','Shipping_country','Shipping_state']
missing_values = retail_data1[columns_to_check].isnull()
missing_values.corr()
retail_data1[retail_data1['Shipping_city'] == retail_data1['Shipping_city'].mode()[0]]
missing_values.mean()*100
retail_data1['Amount'].isnull().mean()*100
import numpy as np
retail_data1['Amount'] = np.where(retail_data1['Order_status'] == 'Cancelled', 0, retail_data1['Amount'])
retail_data1[retail_data1['Amount'].isnull()]
retail_data1['Amount'].skew()
sns.boxplot(retail_data1['Amount'])
retail_data1['Amount'].describe()
retail_data1[retail_data1['Order_status'] == 'Cancelled']
column_check = ['Order_status','Amount']
missing_values = retail_data1[column_check].isnull()
missing_values.corr()
sns.boxplot(retail_data1['Amount'].fillna(retail_data1['Amount'].median()))
retail_data1['Amount'].isnull()
retail_data1[retail_data1['Courier_Status'].isnull()]
retail_data1['Courier_Status'].isnull().mean()*100
retail_data1.loc[:10,:'Courier_Status']
retail_data1.dtypes
retail_data1['Promotion_ids'].unique()
retail_data1['Promotion_ids']
import pandas as pd
retail_data1['Promotion_type'] = None
for i, pro_id in enumerate(retail_data1['Promotion_ids']):
    if pd.notna(pro_id) and 'Amazon PLCC Free-Financing Universal Merchant' in pro_id:
        retail_data1.at[i, 'Promotion_type'] = 'Amazon PLCC Free-Financing Universal Merchant'
    elif pd.notna(pro_id) and 'IN Core Free Shipping' in pro_id:
        retail_data1.at[i, 'Promotion_type'] = 'IN Core Free Shipping'
print(i)
print(pro_id)
retail_data1
retail_data1['Promotion_type'].unique()
retail_data1['Promotion_type'].isnull().mean()
import numpy as np
def find_anomalies(retail_data1):
    anomalies = []
    random_data_std = np.std(retail_data1)
    random_data_mean = np.mean(retail_data1)
    anomaly_cut_off = random_data_std * 3
    lower_limit  = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    for outlier in retail_data1:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies
find_anomalies(retail_data1['Amount'].sort_values(ascending = True))
retail_data1[retail_data1.isnull() == True]
retail_data1.isnull().sum()
retail_data1.dropna(subset = ['Shipping_city','Shipping_state','Pincode'],inplace = True)
retail_data1.isnull().sum()
col = ['Fulfilled_by','Shipping_type']
missing = retail_data1[col]
missing
retail_data1[retail_data1['Fulfilled_by'].isnull()]
retail_data1['Fulfilled_by'] = np.where(retail_data1['Shipping_type'] == 'Expedited','Fulfillment by Amazon ','Easy Ship')
retail_data1
retail_data1.isnull().sum()
retail_data1[retail_data1['Promotion_type'].isnull()].sample(50)
retail_data1[retail_data1['Quantity'] == 0].sample(50)
retail_data1['Amount'].isnull().mean()*100
retail_data1['Amount'].mean()
retail_data1['Amount'].fillna(retail_data1['Amount'].mean()).skew()
retail_data1['Amount'].fillna(retail_data1['Amount'].median()).skew()
retail_data1['Amount'] = retail_data1['Amount'].fillna(retail_data1['Amount'].mean())
retail_data1[retail_data1['Amount'] == 0].sample(50)
retail_data1['Amount'].skew()
retail_data1['Amount'].describe()
retail_data1[retail_data1['Promotion_type'] == 'IN Core Free Shipping'].sample(50)
retail_data1.loc[15:50,'Fulfilled_by':'Promotion_type']
relation = pd.crosstab(retail_data1['Fulfilled_by'], retail_data1['Promotion_type'])
relation
import numpy as np
condition_merchant = retail_data1['Fulfilled_by'] == 'Easy Ship'
retail_data1['Promotion_type'] = np.where(
    condition_merchant,
    'Amazon PLCC Free-Financing Universal Merchant',
    'IN Core Free Shipping')
retail_data1['Category'].unique()
retail_data1.drop(columns = ['Courier_Status','currency','index','Promotion_ids','ASIN','Shipping_country','Sales Channel '],inplace = True)
retail_data1['Pincode'] = retail_data1['Pincode'].astype(str)
retail_data1['Pincode'] = retail_data1['Pincode'].str.replace('.','').str[:6]
retail_data1['Pincode'] = retail_data1['Pincode'].astype('category')
retail_data1[['Shipping_city', 'Shipping_state']] = retail_data1[['Shipping_city', 'Shipping_state']].apply(lambda x: x.str.capitalize())
retail_data1
retail_data1['Shipping_state'].unique()
retail_data1['Shipping_state'] = retail_data1['Shipping_state'].replace({
                                'Rj':'Rajasthan','Orissa':'Odisha',
                                 'Nl':'Nagaland','Rajshthan':'Rajasthan',
                                   'Pb':'Punjab','Chhattisgarh':'Chattisgarh','Ar':'Arunachal pradesh',
                                   'Punjab/mohali/zirakpur':'Punjab'})
retail_data1['Shipping_city'].unique()
retail_data1[retail_data1.duplicated(keep = False)]
retail_data1.drop_duplicates(inplace = True)
print(f'The earliest date is {retail_data1["Order_date"].min()}')
print(f'The latest date is {retail_data1["Order_date"].max()}')
retail_data1['Month'] = retail_data1['Order_date'].dt.month
retail_data1['Day'] = retail_data1['Order_date'].dt.strftime('%A')
retail_data1['Month_name'] = retail_data1['Order_date'].dt.strftime('%B')
retail_new = retail_data1[['Quantity','Category','Shipping_state','Business_type']]
import matplotlib.pyplot as plt
import seaborn as sns
predictors = retail_new.drop(columns=['Business_type'])
for i, predictor in enumerate(predictors):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.countplot(data=retail_new, x=predictor, hue='Business_type')
    plt.xticks(rotation =90,fontsize =7)
    plt.ylabel("count")
    plt.title(f'Count Plot for {predictor} with Business_type')
    plt.show()
Revenue = retail_data1.groupby('Month_name')['Amount'].sum().sort_values(ascending=False).reset_index()
Revenue['Amount'] = Revenue['Amount']/1000000
for index, value in enumerate(Revenue['Amount']):
    plt.text(index, value + 0.1 , f'{value:.2f}M', ha='center', va='bottom', fontsize=8)
sns.barplot(x=Revenue['Month_name'],y=Revenue['Amount'])
plt.xlabel('Month')
plt.ylabel('Revenue(Million)')
plt.show()
monthly_sum = retail_data1.groupby('Month_name')['Amount'].sum().reset_index()
monthly_sum['Amount'] = monthly_sum['Amount'] / 1000000
monthly_sum
retail_data1['Amount'].sum()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

product_sum = retail_data1.groupby('Category')['Amount'].sum().reset_index()
product_sum['Amount'] = product_sum['Amount']/1000000
product_sum.sort_values(by = 'Amount',ascending = False,inplace = True)
for index, value in enumerate(product_sum['Amount']):
    plt.text(index, value + 0.1 , f'{value:.2f}M', ha='center', va='bottom', fontsize=8)
sns.barplot(data = product_sum,x='Category',y='Amount')
plt.xticks(rotation = 45,fontsize = 8.5)
plt.xlabel('Products')
plt.ylabel('Revenue(Millions)')
plt.title('Product by Sales')
plt.show()
size_sales = retail_data1.groupby('Size')['Amount'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=size_sales.index, y=size_sales.values/1000000)
plt.title('Size-wise Sales Analysis')
plt.xlabel('Size')
plt.ylabel('Total Sales(Millions)')
plt.show()
import pandas as pd
total_revenue = retail_data1['Amount'].sum()
total_orders = retail_data1['Order_ID'].nunique()
average_order_value = total_revenue / total_orders
print(f"The Average Order Value (AOV) is: {average_order_value:.2f} RS")
Day_sales = retail_data1.groupby('Day')['Amount'].sum().reset_index().sort_values(by = 'Amount',ascending=False)
Day_sales['Amount'] = Day_sales['Amount']/1000000
sns.barplot(x=Day_sales.Day,y=Day_sales.Amount)
for index, value in enumerate(Day_sales['Amount']):
    plt.text(index, value + 0.1 , f'{value:.2f}M', ha='center', va='bottom', fontsize=8)
plt.xticks(rotation=90)
plt.xlabel('orderd Day')
plt.ylabel('Revenue(Millions)')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
avg_order = retail_data1.groupby('Business_type')['Amount'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_order, x='Business_type', y='Amount', palette='viridis')
plt.xticks(rotation=90)
plt.title('Average Order Value by Business Type')
plt.xlabel('Business Type')
plt.ylabel('Average Order Value')
plt.show()
import pandas as pd
top_cities = retail_data1.groupby(by = 'Shipping_city')['Amount'].sum().reset_index()
top_cities = top_cities.sort_values(by = 'Amount',ascending=False).head(10)
top_cities['Amount'] = top_cities['Amount']/1000000
for index, value in enumerate(top_cities['Amount']):
    plt.text(index, value + 0.1 , f'{value:.2f}M', ha='center', va='bottom', fontsize=7)
sns.barplot(data = top_cities,x='Shipping_city',y='Amount',errorbar=None,width = 0.5)
plt.xticks(rotation=90)
plt.title('Revenue by City')
plt.xlabel('Cities')
plt.ylabel('Revenue(Millions)')
plt.show()
columns = ['Order_status', 'Fulfilment', 'Shipping_type', 'Category',
           'Quantity','Shipping_city', 'Amount', 'Fulfilled_by']
for column in columns:
    unique_values = retail_data1[column].unique()
    unique_values_count = len(unique_values)
    print(f"Column: {column}")
    print(f"Number of unique values: {unique_values_count}")
    print("Unique values in: {}".format(column))
    for value in unique_values:
        print(value)
    print("-----------------------")
    print("\n")
retail_data1['Order_status'] = retail_data1['Order_status'].replace({
    'Shipped - Delivered to Buyer': 'Delivered',
    'Shipped - Returned to Seller': 'Cancelled',
    'Shipped - Rejected by Buyer': 'Cancelled',
    'Shipped - Lost in Transit': 'Lost in Transit',
    'Shipped - Out for Delivery': 'Out for Delivery',
    'Shipped - Returning to Seller': 'Cancelled',
    'Shipped - Picked Up': 'Picked Up', 'Pending - Waiting for Pick Up': 'Waiting for Pick Up',
    'Shipped - Damaged': 'Damaged'
})
trends = retail_data1[['Category','Shipping_state','Size','Quantity','Month_name']]
cancelled_orders = retail_data1[retail_data1['Order_status'] == 'Cancelled']
for i,predictor in enumerate(trends.drop('Quantity',axis=1)):
    cancelled_reasons = cancelled_orders.groupby(predictor)['Quantity'].count().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    cancelled_reasons.plot.bar(color='salmon')
    plt.title('Reasons for Cancelled Orders')
    plt.xlabel(predictor)
    plt.ylabel('Number of Cancelled Orders')
    plt.show()
fulfillment_method = retail_data1.groupby('Business_type')['Order_ID'].nunique().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=fulfillment_method.index, y=fulfillment_method.values, palette='pastel')
plt.title('B2B vs B2C')
plt.xlabel('Business_type')
plt.ylabel('Total Customers')
plt.show()
fulfillment_method
Shipping = retail_data1[retail_data1['Order_status'] == 'Shipping'].groupby('Order_ID')['Day'].count().sort_values(ascending=False).reset_index()
Avg_time = Shipping['Day'].mean()
Avg_time
Duration = retail_data1.groupby(['Shipping_type','Order_ID'])['Day'].count().reset_index()
Avg_time = Duration.groupby('Shipping_type')['Day'].mean().reset_index()
Avg_time
Orders = retail_data1.groupby('Fulfilment')['Order_ID'].count().sort_values(ascending=False).reset_index()
sns.barplot(x=Orders['Fulfilment'],y=Orders['Order_ID']/retail_data1.shape[0] *100)
plt.xlabel('Fulfilled by')
plt.ylabel('Orders %')
plt.show()
order_status_dist = retail_data1['Order_status'].value_counts()
for index, value in enumerate(order_status_dist.values):
    plt.text(index, value , f'{value:.0f}', ha='center', va='bottom', fontsize=7)
sns.barplot(x=order_status_dist.index, y=order_status_dist.values)
plt.xticks(rotation = 90)
plt.xlabel('Order_status')
plt.ylabel('Count')
plt.title('Order Status Distribution')
plt.show()
Avg_qty = retail_data1.groupby('Category')['Quantity'].sum().sort_values(ascending=False).reset_index()
sns.barplot(x = Avg_qty['Category'],y=Avg_qty['Quantity'].apply(lambda x: x/9))
plt.xticks(rotation=90)
plt.xlabel('Products')
plt.ylabel('Avg Qty')
plt.show()
cancelled_orders = retail_data1[retail_data1['Order_status'] == 'Cancelled']
cancelled_reasons = cancelled_orders.groupby('Category')['Quantity'].count().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
cancelled_reasons.plot.bar(color='salmon')
plt.title('Reasons for Cancelled Orders')
plt.xlabel('Category')
plt.ylabel('Number of Cancelled Orders')
plt.show()
plt.figure(figsize=(8, 6))
sns.boxplot(x='Business_type', y='Amount', data=retail_data1, palette='Set2')
plt.title('B2B vs. B2C')
plt.xlabel('Customer Type')
plt.ylabel('Amount')
plt.show()
avg_amount_order = retail_data1.groupby('Fulfilment')['Amount'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_amount_order.index, y=avg_amount_order.values, palette='Blues_r')
plt.title('Average Amount per Order by Fulfillment Method')
plt.xlabel('Fulfillment Method')
plt.ylabel('Average Amount')
plt.show()
cancelled_orders = retail_data1[retail_data1['Order_status'] == 'Cancelled'].groupby('Month')['Order_status'].count()
return_rate = (cancelled_orders / total_orders * 100).fillna(0)
plt.figure(figsize=(8, 6))
sns.lineplot(x=return_rate.index, y=return_rate.values, marker='o')
plt.grid(c = 'grey', ls = 'dashed', lw = 1)
plt.xticks([3,4,5,6],['March','April','May','June'])
plt.xlabel('Month')
plt.ylabel('Return Rate (%)')
plt.title('Monthly Return Rate')
plt.show()
cancelled_orders = retail_data1[retail_data1['Order_status'] == 'Cancelled'].groupby('Category')['Order_status'].count()
return_rate = (cancelled_orders / total_orders * 100).fillna(0)
plt.figure(figsize=(8, 6))
sns.lineplot(x=return_rate.index, y=return_rate.values, marker='o')
plt.grid(c = 'grey', ls = 'dashed', lw = 1)
plt.xlabel('Product')
plt.ylabel('Return Rate (%)')
plt.title('Product wise Return Rate')
plt.show()