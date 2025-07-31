# Before running enter: "cd C:\Users\hanis\New folder\Websites"
# Then enter: "python -m streamlit run AstroSim.py"


import streamlit as st
import numpy as np
import plotly.graph_objs as go
import math

# Constants
mu_earth = 3.986e5  # km^3/s^2
mu_sun = 1.327e11  # km^3/s^2
R_earth = 6371  # km
g = 9.81  # m/s^2

# Sidebar Navigation
modules = [
    "Hohmann Transfer",
    "Lunar Transfer (TLI)",
    "Rocket Equation",
    "Delta-v Budget",
    "Reentry Profile",
    "Launch Window Calculator"
]
st.sidebar.title("AstroTools Modules")
selected_module = st.sidebar.radio("Choose a tool:", modules)

st.title(f"AstroTools: {selected_module}")

# --- Module 1: Hohmann Transfer ---
def hohmann_transfer():
    st.write("Select departure and target planets (from inner Solar System):")
    planets = {
        "Mercury": 0.387,
        "Venus": 0.723,
        "Earth": 1.0,
        "Mars": 1.524,
        "Jupiter": 5.203,
        "Saturn": 9.537,
        "Uranus": 19.191,
        "Neptune": 30.068
    }

    planet_names = list(planets.keys())
    departure = st.selectbox("Departure Planet", planet_names, index=planet_names.index("Earth"))
    target = st.selectbox("Target Planet", planet_names, index=planet_names.index("Mars"))

    if departure == target:
        st.error("Departure and target planets must be different.")
        return

    r1 = planets[departure]
    r2 = planets[target]

    r1_km = r1 * 1.496e8
    r2_km = r2 * 1.496e8
    a = (r1_km + r2_km) / 2

    # Orbital velocities
    v1 = math.sqrt(mu_sun / r1_km)
    v2 = math.sqrt(mu_sun / r2_km)
    v_transfer1 = math.sqrt(mu_sun * (2 / r1_km - 1 / a))
    v_transfer2 = math.sqrt(mu_sun * (2 / r2_km - 1 / a))

    delta_v1 = v_transfer1 - v1
    delta_v2 = v2 - v_transfer2
    total_delta_v = abs(delta_v1) + abs(delta_v2)
    transfer_time = math.pi * math.sqrt(a**3 / mu_sun) / (3600 * 24)  # days

    # Display values
    st.markdown(f"**Δv1 (Departure Burn):** {abs(delta_v1):.3f} km/s")
    st.markdown(f"**Δv2 (Arrival Burn):** {abs(delta_v2):.3f} km/s")
    st.markdown(f"**Total Δv:** {total_delta_v:.3f} km/s")
    st.markdown(f"**Transfer Time:** {transfer_time:.1f} days")

    # Orbit Plots
    theta = np.linspace(0, 2 * np.pi, 500)
    departure_x = r1 * np.cos(theta)
    departure_y = r1 * np.sin(theta)
    target_x = r2 * np.cos(theta)
    target_y = r2 * np.sin(theta)

    # Transfer orbit (ellipse in polar form)
    e = abs(r2 - r1) / (r1 + r2)
    transfer_theta = np.linspace(0, np.pi, 500) if r2 > r1 else np.linspace(np.pi, 2 * np.pi, 500)
    r_transfer = (a / 1.496e8) * (1 - e**2) / (1 + e * np.cos(transfer_theta))
    transfer_x = r_transfer * np.cos(transfer_theta)
    transfer_y = r_transfer * np.sin(transfer_theta)

    fig = go.Figure()

    # Plot orbits
    fig.add_trace(go.Scatter(x=departure_x, y=departure_y, name=f"{departure} Orbit"))
    fig.add_trace(go.Scatter(x=target_x, y=target_y, name=f"{target} Orbit"))
    fig.add_trace(go.Scatter(x=transfer_x, y=transfer_y, name="Transfer Orbit", line=dict(dash='dash')))

    # Plot planet markers
    fig.add_trace(go.Scatter(x=[r1], y=[0], mode='markers+text', name=departure,
                             text=[departure], textposition='top center', marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=[r2 * np.cos(transfer_theta[-1])],
                             y=[r2 * np.sin(transfer_theta[-1])],
                             mode='markers+text', name=target,
                             text=["Arrival"], textposition='top center', marker=dict(size=10, color='orange')))

    fig.update_layout(
        title=f"Hohmann Transfer: {departure} ➡ {target}",
        xaxis_title="X (AU)",
        yaxis_title="Y (AU)",
        showlegend=True,
        autosize=True,
        xaxis=dict(scaleanchor='y', scaleratio=1),
        yaxis=dict(scaleanchor='x', scaleratio=1),
        width=700, height=700
    )

    st.plotly_chart(fig)

# --- Module 2: Lunar Transfer (TLI) ---
def lunar_transfer():
    st.markdown("### Lunar Transfer (TLI) Visualization")

    # User Inputs
    r_leo = st.number_input("LEO Radius (km)", value=6678)  # 300 km above Earth
    r_moon = st.number_input("Moon Orbit Radius (km)", value=384400)

    # Semi-major axis and velocities
    a_transfer = (r_leo + r_moon) / 2
    v_leo = math.sqrt(mu_earth / r_leo)
    v_transfer = math.sqrt(mu_earth * (2 / r_leo - 1 / a_transfer))
    delta_v = v_transfer - v_leo

    st.write(f"Δv for TLI: {delta_v:.2f} km/s")

    # Theta for full circles and transfer
    theta_full = np.linspace(0, 2 * np.pi, 500)
    theta_transfer = np.linspace(0, np.pi, 250)

    # Circular orbits
    leo_x = r_leo * np.cos(theta_full)
    leo_y = r_leo * np.sin(theta_full)

    moon_x = r_moon * np.cos(theta_full)
    moon_y = r_moon * np.sin(theta_full)

    # Transfer ellipse (from perigee to apogee)
    e_transfer = (r_moon - r_leo) / (r_moon + r_leo)
    r_transfer = (a_transfer * (1 - e_transfer ** 2)) / (1 + e_transfer * np.cos(theta_transfer))
    trans_x = r_transfer * np.cos(theta_transfer)
    trans_y = r_transfer * np.sin(theta_transfer)

    # Plot
    fig = go.Figure()

    # Earth and Moon circular orbits
    fig.add_trace(go.Scatter(x=leo_x, y=leo_y, name="LEO Orbit", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=moon_x, y=moon_y, name="Moon Orbit", line=dict(color="gray")))

    # Transfer orbit (half ellipse)
    fig.add_trace(go.Scatter(x=trans_x, y=trans_y, name="TLI Transfer Orbit", line=dict(color="orange", dash="dash")))

    # Labels: TLI and Moon Arrival
    fig.add_trace(go.Scatter(x=[trans_x[0]], y=[trans_y[0]],
                             mode="markers+text", name="TLI Burn",
                             text=["TLI Burn"], textposition="bottom right", marker=dict(size=10, color="red")))

    fig.add_trace(go.Scatter(x=[trans_x[-1]], y=[trans_y[-1]],
                             mode="markers+text", name="Arrival",
                             text=["Lunar Arrival"], textposition="top right", marker=dict(size=10, color="purple")))

    # Earth
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", name="Earth",
                             text=["Earth"], textposition="top center", marker=dict(size=12, color="green")))

    fig.update_layout(
        title="Lunar Transfer via Trans-Lunar Injection (TLI)",
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=700,
        height=700,
        showlegend=True
    )

    st.plotly_chart(fig)

# --- Module 3: Rocket Equation ---
def rocket_equation():
    m0 = st.number_input("Initial Mass (kg)", value=500000)
    mf = st.number_input("Final Mass (kg)", value=200000)
    isp = st.number_input("ISP (s)", value=350)

    ve = isp * g
    delta_v = ve * math.log(m0 / mf)
    st.write(f"Δv: {delta_v:.2f} m/s")

# --- Module 4: Delta-v Budget ---
def delta_v_budget():
    stages = st.number_input("Number of Stages", min_value=1, max_value=10, value=3)
    total_delta_v = 0
    for i in range(stages):
        dv = st.number_input(f"Δv for Stage {i+1} (m/s)", key=f"stage_{i}")
        total_delta_v += dv
    st.write(f"Total Mission Δv: {total_delta_v:.2f} m/s")

# --- Module 5: Reentry Profile ---
def reentry_profile():
    v_entry = st.number_input("Entry Velocity (m/s)", value=7800)
    angle = st.number_input("Entry Angle (degrees)", value=6.5)
    time = np.linspace(0, 300, 300)
    
    # Deceleration (simplified model)
    decel = -0.05 * v_entry * np.exp(-0.03 * time)
    
    # Approximate velocity over time by integrating deceleration
    velocity = v_entry + np.cumsum(decel) * (time[1] - time[0])
    velocity = np.maximum(velocity, 0)  # Ensure no negative velocity
    
    # Approximate heat flux q(t) ~ v^3 * k
    k = 1e-5
    heat_flux = k * velocity**3

    # --- Deceleration Graph ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time, y=decel, name="Deceleration (m/s²)"))
    
    # Add labels to deceleration curve
    fig1.add_trace(go.Scatter(
        x=[time[0]], y=[decel[0]],
        mode="markers+text", text=["Reentry Start"],
        textposition="top right", marker=dict(color="blue", size=10), name="Reentry Start"
    ))
    max_idx = np.argmin(decel)
    fig1.add_trace(go.Scatter(
        x=[time[max_idx]], y=[decel[max_idx]],
        mode="markers+text", text=["Max G-Load"],
        textposition="bottom center", marker=dict(color="red", size=10), name="Max G-Load"
    ))
    fig1.add_trace(go.Scatter(
        x=[time[-1]], y=[decel[-1]],
        mode="markers+text", text=["End"],
        textposition="top left", marker=dict(color="green", size=10), name="End"
    ))

    fig1.update_layout(title="G-Load Profile During Reentry", xaxis_title="Time (s)", yaxis_title="Deceleration (m/s²)")
    st.plotly_chart(fig1)

    # --- Heat Flux Graph ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time, y=heat_flux, name="Heat Flux (W/m²)", line=dict(color="firebrick")))
    fig2.update_layout(title="Estimated Heat Flux During Reentry", xaxis_title="Time (s)", yaxis_title="Heat Flux (W/m²)")
    st.plotly_chart(fig2)

# --- Module 6: Launch Window Calculator ---
def launch_window():
    synodic = 780  # days between Mars/earth optimal launches
    last = st.number_input("Last Optimal Launch (year)", value=2022)
    year = st.number_input("Target Year", value=2026)
    interval = (year - last) * 365
    next_launch = synodic - (interval % synodic)
    st.write(f"Days until next optimal launch: {next_launch:.0f} days")

# --- Module Dispatcher ---
if selected_module == modules[0]:
    hohmann_transfer()
elif selected_module == modules[1]:
    lunar_transfer()
elif selected_module == modules[2]:
    rocket_equation()
elif selected_module == modules[3]:
    delta_v_budget()
elif selected_module == modules[4]:
    reentry_profile()
elif selected_module == modules[5]:
    launch_window()