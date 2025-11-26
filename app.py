# ev_bus_dashboard.py
import heapq
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import streamlit as st
import pandas as pd

# -------------------------
# Core simulation classes
# -------------------------
class Graph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u: str, v: str, w: float):
        self.adj.setdefault(u, []).append((v, w))
        self.adj.setdefault(v, []).append((u, w))

    def neighbors(self, u: str):
        return self.adj.get(u, [])

    def dijkstra(self, src: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        dist = {node: float('inf') for node in self.adj}
        parent = {node: None for node in self.adj}
        dist[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            for nei, w in self.adj[node]:
                nd = d + w
                if nd < dist[nei]:
                    dist[nei] = nd
                    parent[nei] = node
                    heapq.heappush(heap, (nd, nei))
        return dist, parent

    def shortest_path(self, src: str, dst: str) -> Tuple[List[str], float]:
        dist, parent = self.dijkstra(src)
        if dist.get(dst, float('inf')) == float('inf'):
            return [], float('inf')
        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path, dist[dst]


class Battery:
    def __init__(self, capacity: float = 100.0, base_rate: float = 0.5):
        self.capacity = capacity
        self.level = capacity
        self.base_rate = base_rate

    def drain(self, distance: float, terrain_factor: float = 0.0) -> float:
        used = self.base_rate * distance + terrain_factor * distance
        used = min(used, self.level)
        self.level -= used
        return used

    def needs_charging(self, threshold: float = 20.0) -> bool:
        return self.level <= threshold

    def charge(self):
        self.level = self.capacity


class Inventory:
    def __init__(self, items: Dict[str, int]):
        self.items = dict(items)
        self.thresholds = {k: max(1, int(0.2 * v)) for k, v in items.items()}

    def sell(self, item: str, qty: int = 1) -> int:
        available = self.items.get(item, 0)
        sold = min(qty, available)
        self.items[item] = available - sold
        return sold

    def needs_refill(self) -> Dict[str, int]:
        return {k: v for k, v in self.items.items() if v <= self.thresholds.get(k, 0)}


class SalesPredictor:
    def __init__(self, base_sales: Dict[str, int], high_demand_multiplier: float = 1.8, high_prob: float = 0.2):
        self.base_sales = base_sales
        self.high_multiplier = high_demand_multiplier
        self.high_prob = high_prob

    def predict(self) -> Dict[str, int]:
        is_high = random.random() < self.high_prob
        multiplier = self.high_multiplier if is_high else 1.0
        return {k: int(max(0, round(v * multiplier))) for k, v in self.base_sales.items()}


class Bus:
    def __init__(self, name: str, battery: Battery, inventory: Inventory, location: str):
        self.name = name
        self.battery = battery
        self.inventory = inventory
        self.location = location
        self.log = []

    def travel_edge(self, neighbor: str, distance: float, terrain: float = 0.0):
        used = self.battery.drain(distance, terrain_factor=terrain)
        self.location = neighbor
        self.log.append(f"Traveled to {neighbor} | distance={distance}km | battery_used={used:.2f} | remaining={self.battery.level:.2f}")
        return used

    def do_sales(self, predictor: SalesPredictor, prices: Dict[str, float]):
        pred = predictor.predict()
        sold_summary = {}
        money = 0
        for item, qty in pred.items():
            sold = self.inventory.sell(item, qty)
            sold_summary[item] = sold
            money += sold * prices.get(item, 0)
        self.log.append(f"Sales @ {self.location}: {sold_summary} | revenue={money}")
        return sold_summary, money


class Simulation:
    def __init__(self, graph: Graph, bus: Bus, predictor: SalesPredictor, prices: Dict[str, float]):
        self.graph = graph
        self.bus = bus
        self.predictor = predictor
        self.prices = prices
        self.metrics = {
            "total_distance": 0.0,
            "total_revenue": 0.0,
            "battery_used": 0.0,
            "stops": []
        }

    def run_route(self, path: List[Tuple[str, float]]):
        for node, dist in path[1:]:
            used = self.bus.travel_edge(node, dist)
            self.metrics["total_distance"] += dist
            self.metrics["battery_used"] += used
            sold, revenue = self.bus.do_sales(self.predictor, self.prices)
            self.metrics["total_revenue"] += revenue
            self.metrics["stops"].append({
                "node": node,
                "distance": dist,
                "battery_after": self.bus.battery.level,
                "sold": sold,
                "revenue": revenue
            })
            if self.bus.battery.needs_charging():
                self.bus.log.append("Battery low! Suggest charging.")
                break

    def run_shortest_path(self, dst: str):
        path_nodes, total_dist = self.graph.shortest_path(self.bus.location, dst)
        if not path_nodes:
            raise ValueError("No path found")
        node_pairs = []
        for i in range(len(path_nodes)):
            if i == 0:
                node_pairs.append((path_nodes[0], 0.0))
            else:
                prev = path_nodes[i - 1]
                cur = path_nodes[i]
                dist = next((w for n, w in self.graph.neighbors(prev) if n == cur), 0.0)
                node_pairs.append((cur, dist))
        self.run_route(node_pairs)

    def summary(self):
        return {
            "total_distance": self.metrics["total_distance"],
            "total_revenue": self.metrics["total_revenue"],
            "battery_used": self.metrics["battery_used"],
            "stops": self.metrics["stops"],
            "logs": self.bus.log
        }

# -------------------------
# Streamlit UI + glue
# -------------------------
st.set_page_config(page_title="EV Bus Sales Dashboard", layout="wide")
st.title("🚌 Connected EV Bus - Simulator + Sales Dashboard")

# default product list & prices (you can change)
PRODUCT_PRICES = {
    "Water Bottle": 50,
    "Chips Pack": 40,
    "Chocolate": 30,
    "Sandwich": 60,
    "Juice": 45
}

# default inventory counts
DEFAULT_QTY = 50
DEFAULT_INVENTORY = {p: DEFAULT_QTY for p in PRODUCT_PRICES.keys()}

# Sidebar controls
st.sidebar.header("Simulation Controls")
seed = st.sidebar.number_input("Random seed (use for reproducible runs)", value=42, format="%d")
battery_capacity = st.sidebar.number_input("Battery capacity", value=100.0, step=10.0)
battery_base_rate = st.sidebar.number_input("Battery base rate (units/km)", value=0.8)
high_demand_prob = st.sidebar.slider("High demand probability per stop", 0.0, 1.0, 0.25)
high_multiplier = st.sidebar.slider("High demand multiplier", 1.0, 3.0, 1.8)

st.sidebar.markdown("---")

# buttons
init_col, run_col, reset_col = st.sidebar.columns(3)
with init_col:
    init_clicked = st.button("Init Simulation")
with run_col:
    run_clicked = st.button("Run Route → Destination")
with reset_col:
    reset_clicked = st.button("Reset All")

# Initialize or reset session-state data structures
if reset_clicked:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

if "sales_data" not in st.session_state:
    st.session_state.sales_data = []  # list of dicts: Date, Location, Product, Quantity, Price, Total

if "inventory" not in st.session_state:
    st.session_state.inventory = dict(DEFAULT_INVENTORY)

if "total_revenue" not in st.session_state:
    st.session_state.total_revenue = 0.0

if "graph" not in st.session_state or init_clicked:
    # build demo graph
    g = Graph()
    g.add_edge("Depot", "Stop A", 5.0)
    g.add_edge("Stop A", "Stop B", 3.0)
    g.add_edge("Stop B", "Stop C", 4.0)
    g.add_edge("Stop A", "Stop C", 8.0)
    g.add_edge("Stop C", "Stop D", 6.0)
    g.add_edge("Depot", "Stop D", 15.0)
    st.session_state.graph = g

    # inventory object (keeps counts in Inventory.items)
    st.session_state.inv_obj = Inventory(st.session_state.inventory)

    # predictor using product base sales (a tiny base pattern)
    base_sales = {
        "Water Bottle": 12,
        "Chips Pack": 6,
        "Chocolate": 4,
        "Sandwich": 3,
        "Juice": 5
    }
    # e.g., 5 for 50
    st.session_state.predictor = SalesPredictor(base_sales, high_demand_multiplier=high_multiplier, high_prob=high_demand_prob)

    # battery + bus
    st.session_state.battery = Battery(capacity=battery_capacity, base_rate=battery_base_rate)
    st.session_state.bus = Bus("EV-1", st.session_state.battery, st.session_state.inv_obj, location="Depot")

    # simulation object
    st.session_state.sim = Simulation(st.session_state.graph, st.session_state.bus, st.session_state.predictor, PRODUCT_PRICES)

    # bookkeeping
    st.session_state.total_revenue = 0.0
    st.session_state.last_run_time = None
    st.session_state.logs = []

# let user choose destination
nodes = list(st.session_state.graph.adj.keys())
dst = st.sidebar.selectbox("Destination (shortest path will be computed)", options=nodes, index=max(0, nodes.index("Stop D") if "Stop D" in nodes else 0))

# run the route
if run_clicked:
    random.seed(seed)
    try:
        st.session_state.sim.run_shortest_path(dst)
        # after run, take sim.metrics and append sales to session_state.sales_data and update global inventory
        for stop in st.session_state.sim.metrics["stops"]:
            node = stop["node"]
            sold_map = stop["sold"]
            for prod, qty_sold in sold_map.items():
                if qty_sold <= 0:
                    continue
                entry = {
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Location": node,
                    "Product": prod,
                    "Quantity": qty_sold,
                    "Price": PRODUCT_PRICES.get(prod, 0),
                    "Total": qty_sold * PRODUCT_PRICES.get(prod, 0)
                }
                st.session_state.sales_data.append(entry)
                st.session_state.total_revenue += entry["Total"]

        # sync inventory dict from inv_obj (so UI shows correct numbers)
        st.session_state.inventory = dict(st.session_state.inv_obj.items)
        st.session_state.last_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.logs = st.session_state.bus.log.copy()

        st.success(f"Route run complete. Reached {st.session_state.bus.location}.")
    except Exception as e:
        st.error(f"Error running route: {e}")

# -------------------------
# Dashboard (main area)
# -------------------------
st.subheader("🚦 Simulation Status")
col1, col2, col3 = st.columns(3)
col1.metric("Battery level", f"{st.session_state.battery.level:.2f} / {st.session_state.battery.capacity}")
col2.metric("Bus location", st.session_state.bus.location)
col3.metric("Last run", st.session_state.get("last_run_time", "Never"))

st.subheader("💰 Totals & Metrics")
df_all = pd.DataFrame(st.session_state.sales_data)
if not df_all.empty:
    total_rev = df_all["Total"].sum()
else:
    total_rev = 0.0
st.write(f"Total revenue (session): ₹{total_rev:.2f}")
st.write(f"Total recorded transactions: {len(st.session_state.sales_data)}")

st.subheader("📦 Current Inventory")
inventory_df = pd.DataFrame.from_dict(st.session_state.inventory, orient="index", columns=["Available Quantity"])
st.table(inventory_df)

st.subheader("📋 Sales Records")
if not df_all.empty:
    st.dataframe(df_all)
else:
    st.write("No sales recorded yet. Run a route to simulate sales.")

st.subheader("📊 Product Sales Chart")
if not df_all.empty:
    prod_chart = df_all.groupby("Product")["Quantity"].sum().sort_values(ascending=False)
    st.bar_chart(prod_chart)

st.subheader("🗺 Location-wise Revenue")
if not df_all.empty:
    loc_chart = df_all.groupby("Location")["Total"].sum().sort_values(ascending=False)
    st.line_chart(loc_chart)

st.subheader("🏷 Best Sellers")
if not df_all.empty:
    try:
        best_product = df_all.groupby("Product")["Quantity"].sum().idxmax()
        st.success(f"Best seller: {best_product}")
    except Exception:
        st.write("Not enough data yet.")
else:
    st.write("No sales yet.")

st.subheader("🧾 Simulation Logs")
if st.session_state.get("logs"):
    for l in st.session_state.logs[-20:]:
        st.write(" -", l)
else:
    st.write("No logs yet. Initialize and run the simulation.")

st.write("---")
st.write("Pro tips:")
st.write("- Use `Init Simulation` after changing sidebar parameters to rebuild objects.")
st.write("- `Reset All` clears session state (useful to start fresh).")