import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import scipy.stats as stats
import urllib
import matplotlib.image as mpimg
import os
from func import BrazilMapPlotter

# Load datasets
items = pd.read_csv('data/order_items_dataset.csv')
products = pd.read_csv('data/products_dataset.csv')
orders = pd.read_csv('data/orders_dataset.csv')
payments = pd.read_csv('data/order_payments_dataset.csv')
customers = pd.read_csv('data/customers_dataset.csv')
geolocation = pd.read_csv('data/geolocation_dataset.csv')  # Dataset untuk pertanyaan 3

# Konversi kolom tanggal menjadi format datetime
datetime_cols = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp"]
for col in datetime_cols:
    orders[col] = pd.to_datetime(orders[col])

min_date = orders["order_approved_at"].min()
max_date = orders["order_approved_at"].max()

# Sidebar
with st.sidebar:
    st.title("Fadiyah Nur Aulia Sari")
    image_path = "dashboard/foto_saya.jpg"
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.warning(f"Image file '{image_path}' not found.")
    
    # Date Range
    start_date, end_date = st.date_input(
        label="Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Filter berdasarkan rentang tanggal
orders_filtered = orders[(orders["order_approved_at"] >= str(start_date)) & 
                         (orders["order_approved_at"] <= str(end_date))]

# Menggabungkan dataset orders dengan items dan products
items_products = items.merge(products, on='product_id', how='inner')
orders_items = orders_filtered.merge(items_products, on='order_id', how='inner')

# Menghitung total revenue dan jumlah transaksi berdasarkan 'product_id'
product_revenue = orders_items.groupby('product_id').agg(
    total_revenue=pd.NamedAgg(column='price', aggfunc='sum'),
    order_count=pd.NamedAgg(column='order_id', aggfunc='count')
).reset_index()

# Menghitung probabilitas penjualan dan total penjualan
total_orders = len(orders_filtered)
product_revenue['sell_probability'] = product_revenue['order_count'] / total_orders

# **Pertanyaan 1: Produk apa yang memiliki jumlah penjualan tertinggi dalam enam bulan terakhir?**
st.title("Product Analysis and Customer Spend Analysis")

st.subheader("Pertanyaan 1: Produk apa yang memiliki jumlah penjualan tertinggi dalam enam bulan terakhir?")

# Top-Selling Product
top_selling_product = product_revenue.sort_values(by='total_revenue', ascending=False).head(1)
st.write(f"**Top-Selling Product ID:** {top_selling_product['product_id'].values[0]}")
st.write(f"**Total Revenue:** Rp{top_selling_product['total_revenue'].values[0]:,.2f}")
st.write(f"**Sell Probability:** {top_selling_product['sell_probability'].values[0]:,.4f}")

# **Visualisasi untuk Pertanyaan 1:**
st.write("**Visualisasi: Hubungan antara Harga Produk dan Probabilitas Terjual (Sell Probability)**")

# Visualization: Product Price vs. Sell Probability
x = np.log(product_revenue.sell_probability)
y = np.log(product_revenue.total_revenue)

fig, ax = plt.subplots(figsize=(8, 6))
sns.set(style='darkgrid')
plt.title('Product Revenue vs. Sell Probability', fontsize=16)
plt.xlabel('Log Sell Probability', fontsize=12)
plt.ylabel('Log Product Revenue', fontsize=12)

hb = ax.hexbin(x, y, gridsize=14, C=product_revenue.total_revenue, reduce_C_function=np.sum, cmap='cividis')

cb = fig.colorbar(hb, ax=ax)
cb.set_label('Total Revenue (R$)', rotation=270, labelpad=20, fontsize=12)

st.pyplot(fig)

# **Penjelasan Visualisasi Pertanyaan 1:**
st.write("""
Produk dengan probabilitas penjualan rendah, tetapi dengan pendapatan tinggi, terlihat di grafik. Ini menunjukkan bahwa meskipun beberapa produk jarang terjual, mereka tetap menghasilkan pendapatan signifikan karena harga tinggi. Sementara itu, produk dengan probabilitas penjualan tinggi sering terjual tetapi mungkin dengan harga lebih rendah.
""")

# **Pertanyaan 2: Berapa rata-rata pengeluaran per pelanggan berdasarkan lokasi geografis dalam periode tiga bulan terakhir?**
st.subheader("Pertanyaan 2: Berapa rata-rata pengeluaran per pelanggan berdasarkan lokasi geografis dalam periode tiga bulan terakhir?")

# Filter tiga bulan terakhir
three_months_ago = orders['order_approved_at'].max() - pd.DateOffset(months=3)
orders_filtered_3months = orders[orders['order_approved_at'] >= three_months_ago]

# Merge dataset orders dengan payments dan customers
pay_ord_cust = orders_filtered_3months.merge(payments, on='order_id', how='outer').merge(customers, on='customer_id', how='outer')
customer_spent = pay_ord_cust.groupby('customer_unique_id').agg({'payment_value': 'sum'}).sort_values(by='payment_value', ascending=False)

# Menghitung rata-rata pengeluaran per pelanggan berdasarkan lokasi geografis
customer_regions = pay_ord_cust.groupby('customer_state').agg({'payment_value': [np.mean, np.std], 'customer_unique_id': 'count'})
customer_regions.reset_index(inplace=True)

# Confidence intervals untuk pengeluaran rata-rata
cis = stats.t.interval(0.95, loc=customer_regions['payment_value']['mean'], scale=customer_regions['payment_value']['std'] / np.sqrt(customer_regions['customer_unique_id']['count']), df=customer_regions['customer_unique_id']['count'] - 1)
customer_regions['ci_low'] = cis[0]
customer_regions['ci_hi'] = cis[1]

# **Visualisasi untuk Pertanyaan 2:**
st.write("**Visualisasi: Rata-rata Pengeluaran per Pelanggan Berdasarkan Lokasi Geografis (95% CI)**")

# Sort ascendingly for easier to read plot
plot = customer_regions.sort_values(by=('payment_value', 'mean'))

# Make figure
fig, ax = plt.subplots(figsize=(12, 4))
plt.xticks(rotation=30)
plt.xlabel('State')
plt.ylabel('Mean Transaction (95% CI)')
plt.xlim(-0.5, len(plot) - 0.5)
plt.ylim(0, plot['payment_value']['mean'].max() + 50)

# Scatter plot dengan confidence interval
plt.scatter(plot['customer_state'], plot['payment_value']['mean'], s=100, c=plot['payment_value']['mean'])
plt.vlines(plot['customer_state'], plot['ci_low'], plot['ci_hi'], lw=0.5)

st.pyplot(fig)

# **Penjelasan Visualisasi Pertanyaan 2:**
st.write("""
Grafik di atas menunjukkan rata-rata pengeluaran per pelanggan berdasarkan lokasi geografis (state), dengan interval kepercayaan 95%. Lokasi dengan perbedaan pengeluaran yang paling signifikan terlihat pada titik-titik dengan interval kepercayaan yang lebih lebar.
""")

# **Pertanyaan 3: Lokasi geografis mana yang memiliki jumlah pelanggan terbanyak dalam kuartal terakhir, dan bagaimana tren perubahan jumlah pelanggan di lokasi tersebut dibandingkan dengan lokasi lainnya?**
st.subheader("Pertanyaan 3: Lokasi geografis mana yang memiliki jumlah pelanggan terbanyak dalam kuartal terakhir?")

# Filter untuk kuartal terakhir (tiga bulan terakhir)
orders_filtered_quarter = orders[orders['order_approved_at'] >= three_months_ago]

# Menggabungkan dataset customers dengan geolocation_silver
geolocation_group = geolocation.groupby(['geolocation_zip_code_prefix'])['geolocation_state'].nunique().reset_index(name='count')
max_state = geolocation.groupby(['geolocation_zip_code_prefix', 'geolocation_state']).size().reset_index(name='count').drop_duplicates(subset='geolocation_zip_code_prefix').drop('count', axis=1)
geolocation_merged = geolocation.groupby(['geolocation_zip_code_prefix', 'geolocation_city', 'geolocation_state'])[['geolocation_lat', 'geolocation_lng']].median().reset_index()
geolocation_silver = geolocation_merged.merge(max_state, on=['geolocation_zip_code_prefix', 'geolocation_state'], how='inner')

# Menggabungkan dengan dataset customers
customers_silver = customers.merge(geolocation_silver, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='inner')


# **Visualisasi untuk Pertanyaan 3:**
st.write("**Visualisasi: Jumlah Pelanggan Berdasarkan Lokasi Geografis (Kuartal Terakhir)**")

# Panggil class BrazilMapPlotter
map_plotter = BrazilMapPlotter(data=customers_silver.drop_duplicates(subset='customer_unique_id'), plt=plt, mpimg=mpimg, urllib=urllib, st=st)

# Panggil metode plot untuk memvisualisasikan peta
map_plotter.plot()
# **Penjelasan Visualisasi Pertanyaan 3:**
st.write("""
Peta di atas menunjukkan distribusi geografis pelanggan berdasarkan lokasi pada kuartal terakhir. Jumlah pelanggan terbesar dapat dilihat di beberapa titik pusat, dan tren perubahan dapat dilihat dengan membandingkan jumlah pelanggan di lokasi tersebut dengan lokasi lainnya.
""")

st.caption('Copyright (C) FadiyahSari 2024')
