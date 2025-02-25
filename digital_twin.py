# digital_twin_inventory.py
import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import copy
from datetime import datetime

# Initialize session state
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = pd.DataFrame()
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}

class DigitalTwinInventory:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.data = []
        self.total_demand = 0
        self.total_immediate_fulfilled = 0  # Track demand fulfilled without backorders
        self.total_backorder_fulfilled = 0   # Track backorder resolutions
        self.reset_costs()
        self.setup_supply_chain()

    # In Retailer.sell_items():
    def sell_items(self, demand):
        if demand <= 0:
            yield self.env.timeout(0)
            return
        
        available = self.inventory.level
        fulfilled = min(available, demand)
        shortage = demand - fulfilled
        
        if fulfilled > 0:
            yield self.inventory.get(fulfilled)
            self.parent.total_immediate_fulfilled += fulfilled  # Track immediate fulfillment
        
        if shortage > 0:
            self.backorders += shortage
            self.parent.costs['shortage'] += shortage * self.parent.params['shortage_cost']
        
        yield self.env.timeout(0)

    # In Retailer.receive_shipment():
    def receive_shipment(self, quantity):
        if self.backorders > 0:
            fulfilled = min(quantity, self.backorders)
            self.parent.total_backorder_fulfilled += fulfilled  # Track backorder resolution
            self.backorders -= fulfilled
            quantity -= fulfilled
        
        if quantity > 0:
            available_space = self.inventory.capacity - self.inventory.level
            qty_to_add = min(quantity, available_space)
            if qty_to_add > 0:
                yield self.inventory.put(qty_to_add)
        
        yield self.env.timeout(0)

    # Updated service level calculation
    def calculate_service_level(self):
        return (
            self.total_immediate_fulfilled / self.total_demand 
            if self.total_demand > 0 
            else 1.0
        )

    def reset_costs(self):
        self.costs = {
            'holding': 0,
            'ordering': 0,
            'shortage': 0
        }

    def setup_supply_chain(self):
        self.supplier = self.Supplier(self)
        self.warehouse = self.Warehouse(self)
        self.retailer = self.Retailer(self)
        self.env.process(self.demand_generator())
        self.env.process(self.disruption_manager())

    class Supplier:
        def __init__(self, parent):
            self.parent = parent
            self.env = parent.env
            self.original_reliability = parent.params['supplier_reliability']
            self.reliability = self.original_reliability
            self.env.process(self.production_process())

        def production_process(self):
            while True:
                yield self.env.timeout(max(
                    self.parent.params['production_interval'],
                    np.random.exponential(self.parent.params['production_interval'])
                ))
                if np.random.random() <= self.reliability:
                    batch = min(
                        self.parent.params['production_batch_size'],
                        self.parent.warehouse.inventory.capacity - 
                        self.parent.warehouse.inventory.level
                    )
                    if batch > 0:
                        yield self.env.process(
                            self.parent.warehouse.replenish(batch)
                        )

    class Warehouse:
        def __init__(self, parent):
            self.parent = parent
            self.env = parent.env
            self.inventory = simpy.Container(
                self.env,
                init=parent.params['initial_warehouse'],
                capacity=parent.params['warehouse_capacity']
            )
            self.pipeline = []

        def calculate_reorder_point(self):
            lead_time = self.parent.params['shipment_delay'] + \
                       self.parent.params['production_interval']
            
            if self.parent.data:
                demand_history = [d['demand'] for d in self.parent.data[-7:]]
                avg_demand = np.mean(demand_history)
                std_demand = np.std(demand_history)
                safety_stock = (self.parent.params['safety_stock_factor'] * 
                              (std_demand * np.sqrt(lead_time) + 
                               self.parent.retailer.backorders))
                return int(avg_demand * lead_time + safety_stock)
            return int(self.parent.params['average_daily_demand'] * lead_time)

        def replenish(self, quantity):
            available_space = self.inventory.capacity - self.inventory.level
            qty_to_add = min(quantity, available_space)
            if qty_to_add > 0:
                yield self.inventory.put(qty_to_add)

        def ship(self, quantity):
            available = self.inventory.level
            ship_qty = min(available, quantity)
            if ship_qty > 0:
                yield self.inventory.get(ship_qty)
                order = {
                    'quantity': ship_qty,
                    'created': self.env.now,
                    'arrives': self.env.now + self.parent.params['shipment_delay']
                }
                self.pipeline.append(order)
                self.env.process(self.deliver_order(order))

        def deliver_order(self, order):
            yield self.env.timeout(order['arrives'] - self.env.now)
            if order in self.pipeline:
                self.pipeline.remove(order)
                yield self.env.process(
                    self.parent.retailer.receive_shipment(order['quantity'])
                )

    class Retailer:
        def __init__(self, parent):
            self.parent = parent
            self.env = parent.env
            self.inventory = simpy.Container(
                self.env,
                init=parent.params['initial_retailer'],
                capacity=parent.params['retailer_capacity']
            )
            self.backorders = 0
            self.env.process(self.inventory_policy())

        def get_effective_inventory(self):
            current_time = self.env.now
            incoming = sum([
                o['quantity'] for o in self.parent.warehouse.pipeline
                if o['arrives'] <= current_time + self.parent.params['shipment_delay'] + 1
            ])
            return self.inventory.level + incoming - self.backorders

        def inventory_policy(self):
            while True:
                if self.parent.params['policy_type'] == 'sS':
                    yield self.env.process(self.s_S_policy())
                elif self.parent.params['policy_type'] == 'ROP':
                    yield self.env.process(self.ROP_policy())
                yield self.env.timeout(1)

        def s_S_policy(self):
            s = self.parent.warehouse.calculate_reorder_point()
            S = s + self.parent.params['sS_buffer']
            if self.get_effective_inventory() < s:
                order_qty = max(0, S - self.get_effective_inventory())
                if order_qty > 0:
                    self.parent.costs['ordering'] += self.parent.params['ordering_cost']
                    yield self.env.process(self.parent.warehouse.ship(order_qty))

        def ROP_policy(self):
            rop = self.parent.warehouse.calculate_reorder_point()
            if self.get_effective_inventory() < rop:
                D = np.mean([d['demand'] for d in self.parent.data]) if self.parent.data else self.parent.params['average_daily_demand']
                EOQ = int(np.sqrt(
                    (2 * D * self.parent.params['ordering_cost']) / 
                    self.parent.params['holding_cost']
                ))
                order_qty = max(EOQ, rop - self.get_effective_inventory())
                if order_qty > 0:
                    self.parent.costs['ordering'] += self.parent.params['ordering_cost']
                    yield self.env.process(self.parent.warehouse.ship(order_qty))

        def sell_items(self, demand):
            if demand <= 0:
                yield self.env.timeout(0)
                return
            
            available = self.inventory.level
            fulfilled = min(available, demand)
            shortage = demand - fulfilled
            
            if fulfilled > 0:
                yield self.inventory.get(fulfilled)
                # Update immediate fulfillment tracking
                self.parent.total_immediate_fulfilled += fulfilled
            
            if shortage > 0:
                self.backorders += shortage
                self.parent.costs['shortage'] += shortage * self.parent.params['shortage_cost']
            
            yield self.env.timeout(0)

        def receive_shipment(self, quantity):
            if self.backorders > 0:
                fulfilled = min(quantity, self.backorders)
                # Update backorder fulfillment tracking
                self.parent.total_backorder_fulfilled += fulfilled
                self.backorders -= fulfilled
                quantity -= fulfilled
            
            if quantity > 0:
                available_space = self.inventory.capacity - self.inventory.level
                qty_to_add = min(quantity, available_space)
                if qty_to_add > 0:
                    yield self.inventory.put(qty_to_add)
            
            yield self.env.timeout(0)

    def demand_generator(self):
        while True:
            day = self.env.now
            demand = max(0, int(
                (self.params['average_daily_demand'] +
                 self.params['linear_growth'] * day +
                 self.params['seasonal_amplitude'] * np.sin(2 * np.pi * (day % 7) / 7) +
                 np.random.normal(0, self.params['random_std'])) *
                self.params['demand_multiplier']
            ))
            
            self.total_demand += demand
            yield self.env.process(self.retailer.sell_items(demand))
            self.record_data(demand)
            yield self.env.timeout(1)

    def disruption_manager(self):
        while True:
            yield self.env.timeout(np.random.exponential(self.params['disruption_interval']))
            if np.random.random() < self.params['disruption_probability']:
                duration = np.random.uniform(
                    self.params['min_downtime'],
                    self.params['max_downtime']
                )
                self.supplier.reliability = self.supplier.original_reliability * self.params['reliability_reduction']
                yield self.env.timeout(duration)
                self.supplier.reliability = self.supplier.original_reliability

    def record_data(self, demand):
        holding_cost = self.retailer.inventory.level * self.params['holding_cost']
        self.costs['holding'] += holding_cost
        
        self.data.append({
            'time': self.env.now,
            'inventory': self.retailer.inventory.level,
            'backorders': self.retailer.backorders,
            'warehouse_inventory': self.warehouse.inventory.level,
            'demand': demand,
            'holding_cost': self.costs['holding'],
            'shortage_cost': self.costs['shortage'],
            'ordering_cost': self.costs['ordering'],
            'service_level': self.calculate_service_level(),
            'inventory_turns': self.calculate_turns(),
            'supplier_reliability': self.supplier.reliability,
            'in_transit': sum(o['quantity'] for o in self.warehouse.pipeline),
            'safety_stock': self.warehouse.calculate_reorder_point(),
            'cycle_time': self.env.now - (self.data[-1]['time'] if self.data else 0),
            # Add new metrics to dataframe
            'total_immediate_fulfilled': self.total_immediate_fulfilled,
            'total_backorder_fulfilled': self.total_backorder_fulfilled
        })


    def calculate_turns(self):
        if not self.data or np.mean([d['inventory'] for d in self.data]) == 0:
            return 0
        return sum(d['demand'] for d in self.data) / np.mean([d['inventory'] for d in self.data])

def run_simulation(params):
    env = simpy.Environment()
    sim = DigitalTwinInventory(env, params)
    env.run(until=params['duration'])
    return pd.DataFrame(sim.data)

# Streamlit Interface
st.set_page_config(layout="wide")
st.title("📈 Advanced Inventory Management Digital Twin")

with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    params = {
        'duration': st.slider("Simulation Duration (days)", 7, 90, 30),
        'production_interval': st.slider("Production Interval (days)", 0.5, 5.0, 1.0),
        'production_batch_size': st.number_input("Production Batch Size", 100, 1000, 500),
        'supplier_reliability': st.slider("Supplier Reliability", 0.7, 1.0, 0.9),
        'safety_stock_factor': st.slider("Safety Stock Factor", 1.0, 5.0, 2.0),
        'policy_type': st.selectbox("Inventory Policy", ['sS', 'ROP']),
        'sS_buffer': st.number_input("sS Buffer Stock", 50, 500, 200),
        'ordering_cost': st.number_input("Ordering Cost ($)", 50.0, 500.0, 100.0),
        'holding_cost': st.number_input("Holding Cost/Unit/Day ($)", 0.1, 5.0, 0.5),
        'shortage_cost': st.number_input("Shortage Cost/Unit/Day ($)", 0.1, 10.0, 5.0),
        'initial_warehouse': st.number_input("Initial Warehouse Stock", 0, 5000, 500),
        'warehouse_capacity': st.number_input("Warehouse Capacity", 500, 10000, 2000),
        'initial_retailer': st.number_input("Initial Retailer Stock", 0, 1000, 100),
        'retailer_capacity': st.number_input("Retailer Capacity", 100, 2000, 500),
        'shipment_delay': st.slider("Shipment Delay (days)", 1, 7, 2),
        'disruption_interval': st.slider("Disruption Frequency (days)", 30, 365, 100),
        'disruption_probability': st.slider("Disruption Probability", 0.1, 1.0, 0.3),
        'reliability_reduction': st.slider("Reliability Reduction During Disruptions", 0.1, 0.9, 0.5),
        'min_downtime': st.number_input("Minimum Downtime (days)", 1, 7, 2),
        'max_downtime': st.number_input("Maximum Downtime (days)", 3, 14, 5),
        'average_daily_demand': st.number_input("Base Daily Demand", 20, 500, 100),
        'linear_growth': st.number_input("Daily Demand Growth", 0.0, 10.0, 2.0),
        'seasonal_amplitude': st.number_input("Seasonal Amplitude", 0.0, 50.0, 15.0),
        'random_std': st.number_input("Demand Variability (σ)", 0.0, 20.0, 5.0),
        'demand_multiplier': st.selectbox("Demand Scenario", [
            ("Pessimistic", 0.7), ("Neutral", 1.0), ("Optimistic", 1.3)
        ], format_func=lambda x: x[0])[1]
    }

    if st.button("▶️ Run Simulation"):
        st.session_state.sim_data = run_simulation(params)
    
    st.header("🔍 Scenario Management")
    scenario_name = st.text_input("Scenario Name", f"Scenario_{datetime.now().strftime('%Y%m%d_%H%M')}")
    if st.button("💾 Save Scenario"):
        st.session_state.scenarios[scenario_name] = {
            'data': st.session_state.sim_data.copy(),
            'params': copy.deepcopy(params)
        }

if not st.session_state.sim_data.empty:
    df = st.session_state.sim_data
    
    # Alert System
    last_row = df.iloc[-1]
    if last_row['inventory'] < last_row['safety_stock']:
        st.error(f"🚨 Low Stock Alert! Inventory ({last_row['inventory']}) below safety stock ({last_row['safety_stock']})")
    if last_row['backorders'] > 0:
        st.warning(f"⚠️ Active Backorders: {last_row['backorders']} units")
    
    # KPI Summary
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("📦 Service Level", f"{last_row['service_level']*100:.1f}%")
    with col2:
        st.metric("🔄 Inventory Turns", f"{last_row['inventory_turns']:.1f}")
    with col3:
        total_cost = last_row['holding_cost'] + last_row['shortage_cost'] + last_row['ordering_cost']
        st.metric("💰 Total Costs", f"${total_cost:,.0f}")
    with col4:
        st.metric("⚡ Supplier Reliability", f"{last_row['supplier_reliability']*100:.1f}%")
    with col5:
        st.metric("📦 Immediate Fulfillment", 
             f"{last_row['total_immediate_fulfilled']} units")
    with col6:
        st.metric("🔄 Backorder Recovery", 
             f"{last_row['total_backorder_fulfilled']} units")

    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏭 Operations", "💰 Finance", "📦 Inventory", "📈 Predictions"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Inventory Dynamics")
            fig = px.line(df, x='time', y=['inventory', 'demand', 'safety_stock'],
                         labels={'value': 'Units', 'variable': 'Metric'},
                         color_discrete_map={
                             'inventory': '#1f77b4',
                             'demand': '#ff7f0e',
                             'safety_stock': '#2ca02c'
                         })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🚚 Lead Time Distribution")
            fig = px.histogram(df, x='cycle_time', nbins=15,
                              labels={'cycle_time': 'Order Cycle Time (days)'},
                              color_discrete_sequence=['#17becf'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💵 Cost Analysis")
        # Calculate daily costs instead of cumulative
        cost_df = df[['holding_cost', 'shortage_cost', 'ordering_cost']].diff().fillna(0)
        
        # Stacked area chart of daily costs
        fig = px.area(cost_df, 
                    labels={"value": "Daily Cost ($)", "variable": "Cost Type"},
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#d62728'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📉 Cost Composition")
        # Sum all daily costs
        total_holding = cost_df['holding_cost'].sum()
        total_shortage = cost_df['shortage_cost'].sum()
        total_ordering = cost_df['ordering_cost'].sum()
        
        # Create pie chart with proper values
        if (total_holding + total_shortage + total_ordering) > 0:
            fig = px.pie(
                values=[total_holding, total_shortage, total_ordering],
                names=['Holding Costs', 'Shortage Costs', 'Ordering Costs'],
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#d62728']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cost data available for composition analysis")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📦 Warehouse Inventory")
            fig = px.bar(df, x='time', y='warehouse_inventory',
                        labels={'warehouse_inventory': 'Units in Warehouse'},
                        color_discrete_sequence=['#9467bd'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📤 In-Transit Inventory")
            fig = px.area(df, x='time', y='in_transit',
                         labels={'in_transit': 'Units in Transit'},
                         color_discrete_sequence=['#e377c2'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔮 Demand Forecasting")
        try:
            if len(df) >= 14:
                model = ExponentialSmoothing(df['demand'], seasonal='add', seasonal_periods=7).fit()
                forecast = model.forecast(14)
                fig = px.line(forecast, 
                             labels={'value': 'Demand', 'index': 'Days Ahead'},
                             title="14-Day Demand Forecast")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for forecasting (minimum 14 days required)")
        except Exception as e:
            st.error(f"Forecasting error: {str(e)}")

# Scenario Comparison
if st.session_state.scenarios:
    st.subheader("📊 Scenario Benchmarking")
    selected = st.multiselect("Select scenarios to compare", list(st.session_state.scenarios.keys()))
    
    if selected:
        comparison = []
        for name in selected:
            scenario = st.session_state.scenarios[name]
            data = scenario['data']
            p = scenario['params']
            comparison.append({
                'Scenario': name,
                'Policy': p['policy_type'],
                'Service Level': data['service_level'].iloc[-1],
                'Total Cost': data['holding_cost'].iloc[-1] + data['shortage_cost'].iloc[-1] + data['ordering_cost'].iloc[-1],
                'Avg Inventory': data['inventory'].mean(),
                'Max Backorder': data['backorders'].max(),
                'Lead Time': p['shipment_delay']
            })
        
        df_compare = pd.DataFrame(comparison).set_index('Scenario')
        st.dataframe(df_compare.style.format({
            'Service Level': '{:.1%}',
            'Total Cost': '${:,.0f}',
            'Avg Inventory': '{:.0f}',
            'Max Backorder': '{:.0f}',
            'Lead Time': '{:.1f} days'
        }))

# Mobile Optimization
st.markdown("""
<style>
    @media (max-width: 768px) {
        .st-emotion-cache-1v7f65g { flex-direction: column !important; }
        .stButton>button { padding: 15px !important; }
    }
    .stPlotlyChart { background-color: #ffffff; border-radius: 10px; padding: 15px; }
        /* Enhanced Tab Styling */
    div[data-testid="stTabs"] {
        justify-content: center;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    button[data-testid="stTab"] {
        flex: 1;
        height: 50px;
        margin: 0 5px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
        border: 1px solid #dee2e6 !important;
        background-color: white !important;
    }
    
    button[data-testid="stTab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
    }
    
    button[data-testid="stTab"].st-emotion-cache-1n1p1fh {
        background-color: #4e79a7 !important;
        color: white !important;
        border-color: #4e79a7 !important;
    }
    
    /* Tab Label Alignment */
    .stTabLabel {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        font-weight: 500 !important;
    }
    
    /* Mobile Optimization */
    @media (max-width: 768px) {
        div[data-testid="stTabs"] {
            flex-direction: column;
        }
        button[data-testid="stTab"] {
            margin: 5px 0 !important;
        }
    }
        /* KPI Card Styling */
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .kpi-title {
        font-size: 0.9rem !important;
        color: #6c757d !important;
        margin-bottom: 8px !important;
        font-weight: 500 !important;
    }
    
    .kpi-value {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin-bottom: 4px !important;
    }
    
    .kpi-icon {
        font-size: 1.8rem !important;
        margin-bottom: 10px !important;
    }
    
    /* Responsive Grid */
    @media (max-width: 1200px) {
        .kpi-grid {
            grid-template-columns: repeat(3, 1fr) !important;
        }
    }
    
    @media (max-width: 768px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }
    
    @media (max-width: 480px) {
        .kpi-grid {
            grid-template-columns: 1fr !important;
        }
    }
</style>
""", unsafe_allow_html=True)

