import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="CFST Bearing Capacity Prediction",
    layout="wide"
)

st.markdown("""
<style>
    label, p, .st-emotion-cache-16idsys p {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['input_scaler'], data['output_scaler']

def calculate_cost(section_type, B, H, D, t, L, fc, fy):
    """
    Calculates the cost in USD for the column based on material volumes and grades.
    Prices are selected based on the provided table.
    """
    if fc < 75:
        concrete_price_per_m3 = 142
    elif fc < 125:
        concrete_price_per_m3 = 199
    else:
        concrete_price_per_m3 = 555

    if fy < 367.5:
        steel_price_per_m3 = 7410
    elif fy < 575:
        steel_price_per_m3 = 10880
    else:
        steel_price_per_m3 = 14250

    if "Circular" in section_type:
        area_steel = np.pi * t * (D - t) / 1e6
        area_concrete = np.pi * (D - 2*t)**2 / 4 / 1e6
    else:
        area_steel = (B*H - (B - 2*t)*(H - 2*t)) / 1e6
        area_concrete = ((B - 2*t)*(H - 2*t)) / 1e6

    cost = (L / 1000) * (area_steel * steel_price_per_m3 + area_concrete * concrete_price_per_m3)
    return cost

@st.cache_data
def load_pareto_data(model_key):
    """Loads Pareto front data for NSGA and RVEA."""
    data = {}
    section_key, slender_type = model_key.split('_')
    section_abbr = "Cir" if section_key == "Circular" else "Rec"

    for algo in ["NSGA", "RVEA"]:
        filename = f"{section_abbr}_{slender_type}_Pareto_{algo}.xlsx"
        file_path = os.path.join("Resources", filename)
        try:            
            df = pd.read_excel(file_path)
            if "Cost (USD)" in df.columns and "Nu (kN)" in df.columns:
                data[algo] = df
            else:
                st.warning(f"File '{file_path}' was found, but is missing required columns 'Cost (USD)' or 'Nu (kN)'.")
                data[algo] = None
        except FileNotFoundError:
            st.warning(f"Pareto front data file not found at '{file_path}'. The chart will be displayed without this data.")
            data[algo] = None
    return data

def get_axis_limit_and_ticks(max_val, num_ticks=6):
    """Calculate a 'nice' axis limit and tick interval."""
    if max_val == 0:
        return 1, 0.2
    
    exponent = np.floor(np.log10(max_val))
    rel_val = max_val / (10**exponent)
    
    if rel_val < 1.5: step_factor = 0.2
    elif rel_val < 3: step_factor = 0.5
    elif rel_val < 7: step_factor = 1
    else: step_factor = 2
    
    tick_interval = step_factor * (10**exponent)
    axis_limit = np.ceil(max_val / tick_interval) * tick_interval
    return axis_limit, tick_interval

def plot_pareto_fronts(pareto_data, current_point):
    """Plots the Pareto fronts and the current design point."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig, ax = plt.subplots(figsize=(8, 6))

    max_nu = current_point["nu"]
    max_cost = current_point["cost"]

    if pareto_data.get("RVEA") is not None:
        df_rvea = pareto_data["RVEA"]
        max_nu = max(max_nu, df_rvea["Nu (kN)"].max())
        max_cost = max(max_cost, df_rvea["Cost (USD)"].max())
        ax.scatter(df_rvea["Nu (kN)"], df_rvea["Cost (USD)"], s=50, marker='^', color='blue', label="RVEA Optimal Front", alpha=0.7, zorder=2)

    if pareto_data.get("NSGA") is not None:
        df_nsga = pareto_data["NSGA"]
        max_nu = max(max_nu, df_nsga["Nu (kN)"].max())
        max_cost = max(max_cost, df_nsga["Cost (USD)"].max())
        ax.scatter(df_nsga["Nu (kN)"], df_nsga["Cost (USD)"], s=50, marker='^', edgecolors='red', facecolors='none', label="NSGA-II Optimal Front", alpha=0.8, zorder=2)
    
    ax.scatter(current_point["nu"], current_point["cost"], s=250, c='yellow', marker='*', label="Current Design", zorder=3, edgecolors='black')
    
    ax.set_xlabel("Bearing Capacity $N_u$ (kN)", fontsize=18)
    ax.set_ylabel("Cost (USD)", fontsize=18)
    ax.set_title("Comparison with Optimal Pareto Fronts", fontsize=20)

    nu_limit, nu_step = get_axis_limit_and_ticks(max_nu)
    cost_limit, cost_step = get_axis_limit_and_ticks(max_cost)

    ax.set_xlim(left=0, right=nu_limit)
    ax.set_ylim(bottom=0, top=cost_limit)
    ax.xaxis.set_major_locator(plt.MultipleLocator(nu_step))
    ax.yaxis.set_major_locator(plt.MultipleLocator(cost_step))

    ax.tick_params(axis='both', which='major', direction='in', labelsize=16, length=8, width=1.2)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=16)
    fig.tight_layout()
    return fig

CONSTRAINTS = {
    "Circular_S": {
        "D": {"min": 45.0, "max": 1100.0, "default": 150.0}, "t": {"min": 0.5, "max": 23.0, "default": 4.5}, "L": {"min": 152.3, "max": 5000.0, "default": 750.0}
    },
    "Circular_L": {
        "D": {"min": 45.0, "max": 1100.0, "default": 150.0}, "t": {"min": 0.5, "max": 23.0, "default": 4.5}, "L": {"min": 152.3, "max": 5000.0, "default": 3500.0}
    },
    "Rectangular_S": {
        "B": {"min": 50.0, "max": 1001.0, "default": 150.0}, "H": {"min": 60.0, "max": 1001.0, "default": 150.0}, "t": {"min": 0.7, "max": 22.4, "default": 4.5}, "L": {"min": 100.0, "max": 4500.0, "default": 750.0}
    },
    "Rectangular_L": {
        "B": {"min": 50.0, "max": 1001.0, "default": 150.0}, "H": {"min": 60.0, "max": 1001.0, "default": 150.0}, "t": {"min": 0.7, "max": 22.4, "default": 4.5}, "L": {"min": 100.0, "max": 4500.0, "default": 3500.0}
    },
    "Material": {
        "fc": {"min": 10.0, "max": 178.8, "default": 50.0},
        "fy": {"min": 200.2, "max": 1153.0, "default": 460.0},
    }
}

st.title("CFST Column Bearing Capacity Prediction")
st.caption("Application for predicting the axial compression capacity ($N_u$) of Concrete-Filled Steel Tube (CFST) columns.")

st.markdown("---")

st.subheader("1. Select Cross-Section Type")
section_type = st.radio(
    "Select the type of the column's cross-section.",
    ("Circular", "Rectangular"),
    key="section_type",
    horizontal=True,
    label_visibility="collapsed"
)

col1, col2 = st.columns([0.55, 0.45])

with col1:
    st.subheader("2. Input Parameters")
    
    if "Circular" in section_type:
        c = CONSTRAINTS["Circular_L"]
        D = st.number_input("Outer Diameter $D$ (mm)", min_value=c["D"]["min"], max_value=c["D"]["max"], value=c["D"]["default"], step=1.0, key="D")
        B, H = D, D
    else:
        c = CONSTRAINTS["Rectangular_L"]
        B = st.number_input("Section Width $B$ (mm)", min_value=c["B"]["min"], max_value=c["B"]["max"], value=c["B"]["default"], step=1.0, key="B")
        H = st.number_input("Section Height $H$ (mm)", min_value=c["H"]["min"], max_value=c["H"]["max"], value=c["H"]["default"], step=1.0, key="H")
        D = 0

    t_min = min(CONSTRAINTS["Circular_L"]["t"]["min"], CONSTRAINTS["Rectangular_L"]["t"]["min"])
    t_max = max(CONSTRAINTS["Circular_L"]["t"]["max"], CONSTRAINTS["Rectangular_L"]["t"]["max"])
    t = st.number_input("Steel Tube Thickness $t$ (mm)", min_value=t_min, max_value=t_max, value=4.5, step=0.1, key="t")
    
    l_min = min(c['L']['min'] for c in CONSTRAINTS.values() if isinstance(c, dict) and 'L' in c)
    l_max = max(c['L']['max'] for c in CONSTRAINTS.values() if isinstance(c, dict) and 'L' in c)
    L = st.number_input("Column Length $L$ (mm)", min_value=l_min, max_value=l_max, value=3500.0, step=10.0, key="L")

    mat_c = CONSTRAINTS["Material"]
    fc = st.number_input(
        "Concrete Strength $f_c$ (MPa)",
        min_value=mat_c["fc"]["min"], max_value=mat_c["fc"]["max"],
        value=mat_c["fc"]["default"], step=5.0
    )
    fy = st.number_input(
        "Steel Yield Strength $f_y$ (MPa)",
        min_value=mat_c["fy"]["min"], max_value=mat_c["fy"]["max"],
        value=mat_c["fy"]["default"], step=10.0
    )

with col2:
    image_path = os.path.join("Resources", "Picture1.png")
    st.image(
        image_path, 
        caption="Illustration of CFST column parameters.",
        use_container_width=True
    )

D_eq = min(B, H) if "Rectangular" in section_type else D
slenderness_ratio = L / D_eq if D_eq > 0 else 0

slender_type = "S" if slenderness_ratio <= 5 else "L"
section_map = {"Circular": "Circular", "Rectangular": "Rectangular"}
section_key = section_map[section_type]
model_key = f"{section_key}_{slender_type}"
model_name = f"NSGA_{model_key}.pkl"

st.info(f"Slenderness ratio ($L/D_{{eq}}$): **{slenderness_ratio:.2f}** → Automatically selected model for **{'short' if slender_type == 'S' else 'slender'} column**.")

try:
    model, input_scaler, output_scaler = load_model(model_name)
except FileNotFoundError:
    st.error(f"Model file not found: '{model_name}'. Please ensure the file exists in the directory.")
    st.stop()

c = CONSTRAINTS[model_key]
if not (c['L']['min'] <= L <= c['L']['max']):
    st.warning(f"The value $L$ = {L}mm is outside the valid range ({c['L']['min']} - {c['L']['max']}mm) for the selected {'short' if slender_type == 'S' else 'slender'} column model. Please adjust.")
st.markdown("---")

if st.button("Predict Bearing Capacity ($N_u$)", type="primary", use_container_width=True):
    valid_input = True
    if t >= D_eq / 2:
        st.error(f"Error: Steel tube thickness $t$ ({t}mm) must be less than half of the smallest dimension $D_{{eq}}/2$ ({D_eq/2 :.1f}mm).")
        valid_input = False
    
    if "Rectangular" in section_type and B > H:
        st.error("Error: Section Width $B$ cannot be greater than Section Height $H$. Please ensure $B \le H$.")
        valid_input = False

    if valid_input:
        if "Circular" in section_type:
            input_array = np.array([[D, t, L, fc, fy]])
        else:
            input_array = np.array([[B, H, t, L, fc, fy]])
    else:
        st.stop()
    
    input_scaled = input_scaler.transform(input_array)
    output_scaled = model.predict(input_scaled)
    output_unscaled = output_scaler.inverse_transform(output_scaled.reshape(-1, 1))
    
    nu_pred = float(output_unscaled[0, 0])
    
    st.markdown("---")
    st.markdown("### Prediction Results")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Predicted Bearing Capacity $N_u$:")
        st.markdown(f"<h2 style='color:green; text-align: left;'>{nu_pred:.2f} kN</h2>", unsafe_allow_html=True)
    
    if slender_type == "L" and slenderness_ratio > 45:
        st.warning(f"Note: The slenderness ratio $L/D_{{eq}} = {slenderness_ratio:.1f} > 45$. The result may be less accurate as it exceeds the optimal data range for slender columns.")

    with col2:
        with st.spinner("Loading Pareto fronts and plotting..."):
            current_cost = calculate_cost(section_type, B, H, D, t, L, fc, fy)
            current_point = {"cost": current_cost, "nu": nu_pred}
            
            pareto_data = load_pareto_data(model_key)
            fig = plot_pareto_fronts(pareto_data, current_point)
            st.pyplot(fig)