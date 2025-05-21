import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Amazon Sales Dashboard 2025", layout="wide")
sns.set_style("whitegrid")

# Load dataset
df = pd.read_csv("../data/amazon_sales_data_2025.csv")

# Data overview
st.title("Amazon Sales Dashboard")

st.markdown("<div style='margin-top: 0px;'><hr></div>", unsafe_allow_html=True)

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = df['Price'].astype(int)
df['Quantity'] = df['Quantity'].astype(int)
df['Total Sales'] = df['Total Sales'].astype(int)
df['Month'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day_name()

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)


# Sidebar filters
st.sidebar.header("Filter Data")

# Get min and max dates
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

# --- Date Range Filter ---
with st.sidebar.expander("Date Range", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", min_value=min_date, max_value=max_date, value=min_date, key="start")
    with col2:
        end_date = st.date_input("End", min_value=min_date, max_value=max_date, value=max_date, key="end")

    if start_date > end_date:
        st.error("End date must be after start date.")

# --- Product Category ---
with st.sidebar.expander("Product Category", expanded=False):
    categories = df['Category'].dropna().unique().tolist()
    selected_categories = st.multiselect("Category", options=categories, default=categories)

# --- Payment Method ---
with st.sidebar.expander("Payment Method", expanded=False):
    payment_methods = df['Payment Method'].dropna().unique().tolist()
    selected_payments = st.multiselect("Method", options=payment_methods, default=payment_methods)

# --- Order Status ---
with st.sidebar.expander("Order Status", expanded=False):
    statuses = df['Status'].dropna().unique().tolist()
    selected_statuses = st.multiselect("Status", options=statuses, default=statuses)

# --- Apply Filters to Data ---
df = df[
    (df['Date'].dt.date >= start_date) & 
    (df['Date'].dt.date <= end_date) &
    (df['Category'].isin(selected_categories)) &
    (df['Payment Method'].isin(selected_payments)) &
    (df['Status'].isin(selected_statuses))
]



#===========================
# KPI calculations
total_sales = df['Total Sales'].sum()
average_order_value = round(df['Total Sales'].mean(), 2)
total_units_sold = df['Quantity'].sum()
unique_customers = df['Customer Name'].nunique()
total_orders = len(df)
cancelled_orders = len(df[df['Status'] == 'Cancelled'])
fulfilled_orders = len(df[df['Status'] == 'Completed'])
order_fulfillment_rate = round((fulfilled_orders / total_orders) * 100, 2)
average_units_per_order = round(total_units_sold / total_orders, 2)
top_selling_product = df.groupby('Product')['Quantity'].sum().idxmax()
top_category = df.groupby('Category')['Total Sales'].sum().idxmax()
most_used_payment_method = df['Payment Method'].value_counts().idxmax()

# KPI Display
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Revenue", f"${total_sales:,.2f}")
    st.metric("Total Orders", total_orders)
    st.metric("Fulfillment Rate", f"{order_fulfillment_rate}%")

with col2:
    st.metric("Avg. Order Value", f"${average_order_value}")
    st.metric("Units Sold", total_units_sold)
    st.metric("Avg. Units/Order", int(average_units_per_order))

with col3:
    st.metric("Unique Customers", unique_customers)
    st.metric("Popular Product", top_selling_product)
    st.metric("Top Category", top_category)
    st.metric("Popular Payment Method", most_used_payment_method)


st.markdown("<div style='margin-top: 30px;'><hr></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    # Monthly sales
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Monthly Sales Trend</h4>", unsafe_allow_html=True)
    monthly_sales = df.groupby("Month", observed=True)[["Total Sales"]].sum().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=monthly_sales.index, y=monthly_sales["Total Sales"], ax=ax1)
    plt.xticks(rotation=60)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Product-wise sales</h4>", unsafe_allow_html=True)
    # Product-wise sales
    product_sales = df.groupby("Product")["Total Sales"].sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=product_sales.values, y=product_sales.index, ax=ax2)
    st.pyplot(fig2)

st.markdown("<div style='margin-top: 30px;'><hr></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Order volume by day</h4>", unsafe_allow_html=True)
    # Order volume by day
    orders_by_day = df['Day'].value_counts().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig3, ax3 = plt.subplots(figsize=(6, 5))  # Adjusted to match pie chart height
    sns.barplot(x=orders_by_day.index, y=orders_by_day.values, ax=ax3)
    ax3.set_xlabel("Day of the Week")
    ax3.set_ylabel("Order Count")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

with col2:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Customer Segmentation</h4>", unsafe_allow_html=True)
    # Customer Segmentation
    order_counts = df.groupby("Customer Name").size()
    customer_types = order_counts.apply(lambda x: "Returning" if x > 1 else "New")
    type_counts = customer_types.value_counts()
    fig4, ax4 = plt.subplots(figsize=(6, 5))  # Match height with col1
    ax4.pie(type_counts, labels=type_counts.index, autopct="%1.1f%%", startangle=140, colors=["#66b3ff", "#ff9999"])
    st.pyplot(fig4)



st.markdown("<div style='margin-top: 30px;'><hr></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Customer Retention Cohort Heatmap</h4>", unsafe_allow_html=True)
    # Cohort Analysis
    df['OrderPeriod'] = df['Date'].dt.to_period('M')
    df['CohortMonth'] = df.groupby('Customer Name')['Date'].transform('min').dt.to_period('M')
    df['CohortIndex'] = (df['OrderPeriod'].dt.year - df['CohortMonth'].dt.year) * 12 + \
                        (df['OrderPeriod'].dt.month - df['CohortMonth'].dt.month) + 1
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['Customer Name'].nunique().unstack(0)
    cohort_percent = cohort_data / cohort_data.iloc[0]
    fig_cohort, ax_cohort = plt.subplots(figsize=(12, 6))
    sns.heatmap(cohort_percent, annot=True, fmt='.0%', cmap='Blues', ax=ax_cohort)
    #ax_cohort.set_title('Customer Retention Cohort Heatmap')
    st.pyplot(fig_cohort)

with col2:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Sales By Category</h4>", unsafe_allow_html=True)
    # Sales by Category
    category_sales = df.groupby("Category")[["Total Sales"]].sum().sort_values("Total Sales", ascending=False)
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=category_sales["Total Sales"], y=category_sales.index, ax=ax5)
    st.pyplot(fig5)



st.markdown("<div style='margin-top: 30px;'><hr></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    # Top 5 Customers
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Top 5 Customers by Sales</h4>", unsafe_allow_html=True)
    top_customers = df.groupby('Customer Name')['Total Sales'].sum().sort_values(ascending=False).head(5)
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    top_customers.plot(kind='bar', color='skyblue', ax=ax6)
    ax6.set_title("")  # Optional: remove matplotlib title
    st.pyplot(fig6)

with col2:
    # Sales by Payment Method
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Sales by Payment Method</h4>", unsafe_allow_html=True)
    payment_sales = df.groupby('Payment Method')['Total Sales'].sum().sort_values(ascending=False)
    fig7, ax7 = plt.subplots(figsize=(8, 4))
    payment_sales.plot(kind='bar', color='orange', ax=ax7)
    ax7.set_title("")  # Optional: remove matplotlib title
    st.pyplot(fig7)

st.markdown("<div style='margin-top: 30px;'><hr></div>", unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Sales by Day of the Week</h4>", unsafe_allow_html=True)
    # Sales by Day of the Week
    sales_by_day = df.groupby('Day')['Total Sales'].sum().sort_values(ascending=False)
    fig8, ax8 = plt.subplots()
    sales_by_day.plot(kind='bar', color='seagreen', ax=ax8)
    st.pyplot(fig8)

with col2:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Top 10 Locations by Total Sales</h4>", unsafe_allow_html=True)
    # Sales by Customer Location
    location_sales = df.groupby('Customer Location')['Total Sales'].sum().sort_values(ascending=False).head(10)
    fig9, ax9 = plt.subplots()
    location_sales.plot(kind='barh', color='purple', ax=ax9)
    st.pyplot(fig9)

st.markdown("<div style='margin-top: 30px;'><hr></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Order Status Summary</h4>", unsafe_allow_html=True)
    status_counts = df['Status'].value_counts()
    fig10, ax10 = plt.subplots(figsize=(6, 4))  # Fixed height and width
    status_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'tomato', 'gold'], startangle=90, ax=ax10)
    plt.ylabel("")
    ax10.set_aspect('equal')  # Ensures the pie chart stays circular
    st.pyplot(fig10)

with col2:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Sales Forecast (Linear Regression)</h4>", unsafe_allow_html=True)

    df['Date'] = pd.to_datetime(df['Date'])
    daily_sales = df.groupby('Date')['Total Sales'].sum()
    daily_sales.index = pd.to_datetime(daily_sales.index)
    X = daily_sales.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = daily_sales.values

    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range(daily_sales.index.max() + pd.Timedelta(days=1), periods=30)
    future_ordinals = future_dates.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    future_preds = model.predict(future_ordinals)

    fig, ax = plt.subplots(figsize=(6, 4))  # Same size as pie chart
    ax.plot(daily_sales.index, y, label='Actual Sales', color='blue')
    ax.plot(future_dates, future_preds, label='Forecasted Sales', color='red', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.markdown("<div style='margin-top: 30px;'><hr></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # 📊 RFM Customer Segmentation
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Customer Segmentation Based on RFM</h4>", unsafe_allow_html=True)

    # Ensure date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Step 1: Compute RFM metrics
    snapshot_date = df['Date'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer Name').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,   # Recency
        'Order ID': 'nunique',                              # Frequency
        'Total Sales': 'sum'                                # Monetary
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    def qcut_with_dynamic_labels(series, q=5, ascending=True):
        bins = pd.qcut(series, q, duplicates='drop')
        n_bins = bins.cat.categories.size
        labels = list(range(n_bins, 0, -1)) if ascending else list(range(1, n_bins + 1))
        return pd.qcut(series, q, labels=labels, duplicates='drop').astype(int)

    # Assign RFM scores
    rfm['R_score'] = qcut_with_dynamic_labels(rfm['Recency'], q=5, ascending=True)
    rfm['F_score'] = qcut_with_dynamic_labels(rfm['Frequency'], q=5, ascending=False)
    rfm['M_score'] = qcut_with_dynamic_labels(rfm['Monetary'], q=5, ascending=False)
    rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

    def segment(rfm_row):
        if rfm_row['R_score'] >= 4 and rfm_row['F_score'] >= 4:
            return 'Loyal Customer'
        elif rfm_row['R_score'] >= 4:
            return 'Recent Customer'
        elif rfm_row['F_score'] >= 4:
            return 'Frequent Buyer'
        elif rfm_row['M_score'] >= 4:
            return 'High Spender'
        else:
            return 'Other'

    rfm['Segment'] = rfm.apply(segment, axis=1)

    # Plot RFM segments
    fig_rfm, ax_rfm = plt.subplots(figsize=(8, 5))
    sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index,
                  palette='Set2', hue='Segment', legend=False, ax=ax_rfm)
    ax_rfm.set_xlabel('Segment')
    ax_rfm.set_ylabel('Number of Customers')
    plt.xticks(rotation=30)
    ax_rfm.grid(axis='y')
    st.pyplot(fig_rfm)

with col2:
    # Average Sales per Customer
    avg_sales = df.groupby('Customer Name')['Total Sales'].mean().sort_values(ascending=False)

    st.markdown("<h4 style='text-align: center; font-size:16px;'>Average Sales per Customer</h4>", unsafe_allow_html=True)
    fig_avg, ax_avg = plt.subplots(figsize=(8, 4))
    avg_sales.head(10).plot(kind='bar', color='teal', ax=ax_avg)
    ax_avg.set_ylabel("Average Sales")
    ax_avg.set_xlabel("Customer")
    ax_avg.tick_params(axis='x', rotation=45)
    st.pyplot(fig_avg)


# ---------------------------
# Customer Churn Analysis Row
# ---------------------------
st.markdown("<div style='margin-top: 30px;'><hr></div>", unsafe_allow_html=True)

# Step 1: Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Step 2: Define the analysis date
analysis_date = df['Date'].max()

# Step 3: Calculate recency
recency_df = df.groupby('Customer Name')['Date'].max().reset_index()
recency_df['Recency'] = (analysis_date - recency_df['Date']).dt.days

# Step 4: Define churn threshold
churn_threshold = 60
recency_df['Churned'] = recency_df['Recency'] > churn_threshold

# Step 5: Calculate churn rate
churn_rate = recency_df['Churned'].mean()

# Layout for 2 side-by-side charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Customer Churn Status</h4>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.countplot(x='Churned', data=recency_df, ax=ax1)
    ax1.set_xlabel('Churned')
    ax1.set_ylabel('Number of Customers')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Active', 'Churned'])
    st.pyplot(fig1)

with col2:
    st.markdown("<h4 style='text-align: center; font-size:16px;'>Recency Distribution of Customers</h4>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(recency_df['Recency'], bins=30, kde=True, ax=ax2)
    ax2.axvline(churn_threshold, color='red', linestyle='--', label='Churn Threshold')
    ax2.set_xlabel('Days Since Last Purchase')
    ax2.set_ylabel('Number of Customers')
    ax2.legend()
    st.pyplot(fig2)


