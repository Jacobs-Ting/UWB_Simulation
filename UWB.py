import streamlit as st
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import fftconvolve, correlate
import plotly.graph_objects as go

# ==========================================
# 1. Page Config & Navigation
# ==========================================
st.set_page_config(page_title="FiRa UWB PHY System Analyzer", layout="wide")

st.sidebar.title("⚡ UWB System Analyzer")
page = st.sidebar.radio("Select Analysis Mode", ["📡 Tx Analysis (Spectrum & Power)", "🎯 Rx Analysis (Ranging & Link Budget)"])
st.sidebar.markdown("---")

# Shared Physical Constants
c = 3e8 
chip_rate = 499.2e6 
Tc = 1.0 / chip_rate 
oversample = 64
fs = chip_rate * oversample 

# Shared Channel Settings
fira_channel = st.sidebar.selectbox("FiRa Channel", ["Channel 5 (6.4896 GHz)", "Channel 9 (7.9872 GHz)"])
fc_ghz = 6.4896 if "Channel 5" in fira_channel else 7.9872
fc = fc_ghz * 1e9

st.sidebar.markdown("---")

def generate_rrc_pulse(t, Tc, alpha=0.6):
    t = t + 1e-14 
    num = np.sin(np.pi * t * (1 - alpha) / Tc) + 4 * alpha * (t / Tc) * np.cos(np.pi * t * (1 + alpha) / Tc)
    den = np.pi * (t / Tc) * (1 - (4 * alpha * t / Tc)**2)
    return num / den

# ==========================================
# Page 1: Tx Analysis (Spectrum & Power)
# ==========================================
if page == "📡 Tx Analysis (Spectrum & Power)":
    st.title("📡 Tx Analysis (Time Domain & Spectrum)")
    st.markdown("Evaluate Tx hardware direct-drive voltage, time-domain waveform characteristics, and FCC PSD compliance.")

    with st.sidebar:
        st.header("⚙️ Tx Hardware Settings")
        dac_vp_mv = st.slider("Peak Output Voltage Vp (mV)", 10, 100, 40, 1)
        num_chips_tx = st.slider("Number of Simulated Chips", 500, 5000, 2000, step=500)
        smoothing_mhz = st.slider("VBW Smoothing Window (MHz)", 1.0, 50.0, 15.0, 1.0)

    # Tx Computation Logic
    t_template = np.arange(-10 * Tc, 10 * Tc, 1/fs)
    pulse_ideal = generate_rrc_pulse(t_template, Tc)
    pulse_ideal = (pulse_ideal / np.max(np.abs(pulse_ideal))) * (dac_vp_mv / 1000.0)

    np.random.seed(42)
    chips = np.random.randint(0, 2, num_chips_tx)
    bpsk_symbols = np.where(chips == 1, 1.0, -1.0)
    impulse_train = np.zeros(num_chips_tx * oversample)
    impulse_train[::oversample] = bpsk_symbols

    tx_baseband = fftconvolve(impulse_train, pulse_ideal, mode='same')
    tx_baseband -= np.mean(tx_baseband)
    t_tx = np.arange(len(tx_baseband)) / fs
    tx_rf = tx_baseband * np.sqrt(2) * np.cos(2 * np.pi * fc * t_tx)

    # Spectrum & Power Calculation
    N = len(tx_rf)
    freqs = fftfreq(N, 1/fs)          
    pulse_fft = fft(tx_rf)            
    pos_mask = freqs > 0
    f_pos, fft_pos = freqs[pos_mask], pulse_fft[pos_mask]

    Power_W = ((2.0 / N) * np.abs(fft_pos) / np.sqrt(2))**2 / 50.0
    total_power_dBm = 10 * np.log10(np.sum(Power_W) * 1000 + 1e-20)

    df = f_pos[1] - f_pos[0] 
    kernel_size = max(3, int((smoothing_mhz * 1e6) / df))
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_Power_W = np.convolve(Power_W, kernel, mode='same')
    spectrum_dbm = 10 * np.log10(smoothed_Power_W * 1000 + 1e-20)

    peak_idx = np.argmax(spectrum_dbm)
    max_val = spectrum_dbm[peak_idx]        
    above_idx = np.where(spectrum_dbm >= max_val - 10.0)[0]
    
    if len(above_idx) > 0:
        bw_10db_ghz = (f_pos[above_idx[-1]] - f_pos[above_idx[0]]) / 1e9
        psd_dbm_mhz = total_power_dBm - 10 * np.log10(bw_10db_ghz * 1e9) + 60
    else:
        bw_10db_ghz, psd_dbm_mhz = 0.0, -150.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚡ Set Voltage (Vp)", f"{dac_vp_mv} mV")
    col2.metric("📏 -10dB Effective BW", f"{bw_10db_ghz:.3f} GHz")
    col3.metric("📈 Total Tx Power", f"{total_power_dBm:.2f} dBm")
    col4.metric("📊 Effective PSD (FCC)", f"{psd_dbm_mhz:.2f} dBm/MHz", delta="Over Limit" if psd_dbm_mhz > -41.3 else "Pass (-41.3)", delta_color="inverse" if psd_dbm_mhz > -41.3 else "normal")

    st.markdown("---")
    
    fig_time = go.Figure()
    mask_t = t_tx <= 0.05e-6 
    fig_time.add_trace(go.Scatter(x=t_tx[mask_t]*1e6, y=tx_rf[mask_t], line=dict(color='#00FF00', width=1)))
    fig_time.update_layout(title="Time Domain RF Waveform (Passband, First 0.05 μs)", xaxis_title="Time (μs)", yaxis_title="Amplitude (V)", plot_bgcolor='#000000', paper_bgcolor='#0E1117', font=dict(color='white'), height=300)
    st.plotly_chart(fig_time, use_container_width=True)

    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=f_pos*1e-9, y=spectrum_dbm, line=dict(color='#FFFF00', width=1.5)))
    fig_freq.update_layout(title="Absolute Spectrum Analysis", xaxis_title="Frequency (GHz)", yaxis_title="Power (dBm)", xaxis=dict(range=[fc_ghz-1, fc_ghz+1]), plot_bgcolor='#000000', paper_bgcolor='#0E1117', font=dict(color='white'), height=350)
    st.plotly_chart(fig_freq, use_container_width=True)


# ==========================================
# Page 2: Rx Analysis (Ranging, AoA & Sensitivity)
# ==========================================
elif page == "🎯 Rx Analysis (Ranging & Link Budget)":
    st.title("🎯 Rx Dual-Antenna Ranging & Link Budget")
    st.markdown("Verify baseband algorithm's CFO tolerance, AoA performance, and rigorously calculate FiRa Receiver Sensitivity.")

    with st.sidebar:
        st.header("🛸 Space & Channel Settings")
        distance_m = st.slider("Target Distance (ToF, m)", 0.5, 100.0, 10.0, 0.5)
        actual_aoa_deg = st.slider("Actual AoA (Degrees)", -90.0, 90.0, 30.0, 1.0)
        antenna_spacing = st.slider("Antenna Spacing (x λ)", 0.2, 1.0, 0.5, 0.1)
        
        st.markdown("---")
        st.header("📡 RF Transceiver Params")
        tx_vp_rx = st.slider("Tx Output Voltage Vp (mV)", 10, 100, 40, 1)
        rx_nf = st.slider("Rx Noise Figure (NF, dB)", 2.0, 12.0, 6.0, 0.1, help="LNA and Mixer noise contribution.")
        cfo_mhz = st.slider("Carrier Freq Offset (CFO, MHz)", -50.0, 50.0, 20.0, 0.5)

        st.markdown("---")
        st.header("🧠 Rx Baseband Algorithm")
        num_chips_rx = st.slider("STS Length (Chips)", 100, 2000, 500, 100)
        rx_algo = st.radio("Sync & Compensation Strategy", ["1. Traditional Full-Coherent", "2. Partial Correlation", "3. Preamble De-rotation"])
        
        num_segments = 1
        estimated_cfo_mhz = 0.0
        
        if "Partial" in rx_algo:
            num_segments = st.slider("Segments", 2, 20, 5)
        elif "Preamble" in rx_algo:
            st.info("🟢 Preamble Estimation Active")
            # FIX: Offset seed to support negative CFO values without crashing
            np.random.seed(int((cfo_mhz + 100.0) * 100))
            estimated_cfo_mhz = cfo_mhz + np.random.normal(0, 1.5) 
            st.write(f"Estimated CFO: {estimated_cfo_mhz:+.2f} MHz")

    # --- Theoretical Link Budget & Sensitivity Math ---
    # 1. Calculate Actual Tx Power
    v_rms = (tx_vp_rx / 1000.0) / np.sqrt(2)
    p_tx_w = (v_rms**2) / 50.0 # Approximation for BPSK peak. For RRC, average is roughly half. Let's use 0.5 factor.
    p_tx_dbm = 10 * np.log10(p_tx_w * 0.5 * 1000 + 1e-20)
    
    # 2. Free Space Path Loss (FSPL)
    fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(fc) - 147.55
    p_rx_dbm = p_tx_dbm - fspl_db
    
    # 3. FiRa Sensitivity Calculation
    n_thermal = -174 + 10 * np.log10(499.2e6) # ~ -87 dBm
    snr_min = 12.0 # Minimum SNR required for reliable peak detection
    g_proc = 10 * np.log10(num_chips_rx)
    l_algo = 10 * np.log10(num_segments) if "Partial" in rx_algo else 0.0
    g_div = 3.0 # 2Rx EGC Gain
    
    sensitivity_dbm = n_thermal + rx_nf + snr_min - g_proc + l_algo - g_div
    link_margin = p_rx_dbm - sensitivity_dbm

    # --- Tx Generation (for Simulation) ---
    t_template = np.arange(-10 * Tc, 10 * Tc, 1/fs)
    pulse_ideal = generate_rrc_pulse(t_template, Tc)
    pulse_ideal = (pulse_ideal / np.max(np.abs(pulse_ideal))) * (tx_vp_rx / 1000.0)

    np.random.seed(123) 
    chips = np.random.randint(0, 2, num_chips_rx)
    bpsk_symbols = np.where(chips == 1, 1.0, -1.0)
    impulse_train = np.zeros(num_chips_rx * oversample)
    impulse_train[::oversample] = bpsk_symbols

    tx_baseband = fftconvolve(impulse_train, pulse_ideal, mode='same')
    t_tx = np.arange(len(tx_baseband)) / fs

    actual_fc = fc + (cfo_mhz * 1e6)
    phase_shift_rad = 2 * np.pi * antenna_spacing * np.sin(np.radians(actual_aoa_deg))

    tx_rf1 = tx_baseband * np.sqrt(2) * np.cos(2 * np.pi * actual_fc * t_tx)
    tx_rf2 = tx_baseband * np.sqrt(2) * np.cos(2 * np.pi * actual_fc * t_tx - phase_shift_rad)

    # Simulated Channel (Visual scaling based on Link Margin to keep plots visible but mathematically representative)
    prop_delay = distance_m / c
    delay_samples = int(prop_delay * fs)
    rx_rf1 = np.zeros(len(tx_rf1) + delay_samples + 500)
    rx_rf2 = np.zeros(len(tx_rf2) + delay_samples + 500)
    
    # Apply normalized attenuation for visual representation
    vis_att = 1.0 / (distance_m/5.0 + 1.0)
    rx_rf1[delay_samples : delay_samples + len(tx_rf1)] = tx_rf1 * vis_att
    rx_rf2[delay_samples : delay_samples + len(tx_rf2)] = tx_rf2 * vis_att

    # Dynamic noise injection based on calculated Link Margin
    np.random.seed(None)
    base_peak_amp = (tx_vp_rx/1000.0) * vis_att * (num_chips_rx/2) # Approximate matched filter peak
    # If Margin = 0, simulated SNR should hit the 12dB threshold.
    simulated_snr_target = 12.0 + link_margin
    noise_std = base_peak_amp / (10**(simulated_snr_target / 20.0))
    
    rx_rf1 += np.random.normal(0, noise_std, len(rx_rf1))
    rx_rf2 += np.random.normal(0, noise_std, len(rx_rf2))

    # Rx Algorithm Core
    rx_lo_freq = fc + (estimated_cfo_mhz * 1e6) if "Preamble" in rx_algo else fc
    t_ref = np.arange(len(tx_baseband)) / fs
    rx_template_cplx = tx_baseband * np.exp(1j * 2 * np.pi * rx_lo_freq * t_ref)

    L = len(rx_template_cplx)
    N_out = len(rx_rf1) - L + 1
    total_corr1 = np.zeros(N_out)
    total_corr2 = np.zeros(N_out)
    cross_corr = np.zeros(N_out, dtype=complex)

    seg_len = L // num_segments
    for k in range(num_segments):
        start_idx = k * seg_len
        end_idx = start_idx + seg_len if k < num_segments - 1 else L
        temp_seg = rx_template_cplx[start_idx:end_idx]
        
        c1 = correlate(rx_rf1, temp_seg, mode='valid', method='fft')
        c2 = correlate(rx_rf2, temp_seg, mode='valid', method='fft')
        
        total_corr1 += np.abs(c1[start_idx : start_idx + N_out])
        total_corr2 += np.abs(c2[start_idx : start_idx + N_out])
        cross_corr += c2[start_idx : start_idx + N_out] * np.conj(c1[start_idx : start_idx + N_out])

    t_corr = np.arange(N_out) / fs
    combined_corr = total_corr1 + total_corr2
    peak_idx = np.argmax(combined_corr)

    measured_distance = t_corr[peak_idx] * c

    # Pure Energy Calculation
    peak_1 = total_corr1[peak_idx]
    peak_comb = combined_corr[peak_idx]
    signal_power_gain = 20 * np.log10(peak_comb / (peak_1 + 1e-20))
    diversity_gain = signal_power_gain - 10 * np.log10(2)

    measured_pdoa_rad_corrected = -np.angle(cross_corr[peak_idx])
    sin_theta = np.clip(measured_pdoa_rad_corrected / (2 * np.pi * antenna_spacing), -1.0, 1.0)
    measured_aoa_deg_corrected = np.degrees(np.arcsin(sin_theta))
    is_wrapped = abs(measured_pdoa_rad_corrected) > (np.pi * 0.95)

    # --- Rx Dashboard 1: Link Budget & Sensitivity ---
    st.subheader("📡 RF Link Budget & Sensitivity (FiRa Model)")
    lb1, lb2, lb3, lb4 = st.columns(4)
    lb1.metric("Transmitted Power (Tx)", f"{p_tx_dbm:.1f} dBm")
    lb2.metric("Received Power (Prx)", f"{p_rx_dbm:.1f} dBm", delta=f"FSPL: -{fspl_db:.1f} dB", delta_color="off")
    lb3.metric("Required Sensitivity (S)", f"{sensitivity_dbm:.1f} dBm", help="Calculated from Thermal Noise (-87) + NF + SNR_min(12) - G_proc + L_algo - G_div(3).")
    
    # Margin evaluation
    margin_color = "normal" if link_margin > 0 else "inverse"
    margin_text = "Link Stable" if link_margin > 0 else "Link Broken (Out of Range)"
    lb4.metric("Link Margin", f"{link_margin:+.1f} dB", delta=margin_text, delta_color=margin_color)

    st.markdown("---")

    # --- Rx Dashboard 2: Ranging & AoA Performance ---
    st.subheader("🎯 Decoding Performance & Spatial Decoding")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Ranging Error (ToF)", f"{(measured_distance - distance_m)*100:+.2f} cm")
    c2.metric("🧭 Measured AoA", f"{measured_aoa_deg_corrected:+.1f}°", delta=f"Actual: {actual_aoa_deg:+.1f}°", delta_color="inverse" if is_wrapped else "off")
    c3.metric("🌪️ CFO Status", f"{cfo_mhz} MHz", delta="De-rotation Active" if "Preamble" in rx_algo else "Uncompensated", delta_color="normal" if "Preamble" in rx_algo else "off")
    c4.metric("💪 Theo. Diversity Gain", f"{diversity_gain:+.2f} dB")

    if is_wrapped:
        st.error("⚠️ Phase Wrapping Detected! The extreme angle caused PDoA to exceed ±180°.")
    st.markdown("---")

    # Chart Area
    col_p1, col_p2 = st.columns([2, 1])
    
    fig_corr = go.Figure()
    corr_limit = prop_delay + 100e-9 
    mask = t_corr <= corr_limit
    fig_corr.add_trace(go.Scatter(x=t_corr[mask]*1e9, y=total_corr1[mask], name='Single Antenna (Rx1)', line=dict(color='cyan', width=1, dash='dot')))
    fig_corr.add_trace(go.Scatter(x=t_corr[mask]*1e9, y=combined_corr[mask], name='Dual Antenna Combined (Rx1+Rx2)', line=dict(color='#00FF00', width=2)))
    fig_corr.add_vline(x=prop_delay*1e9, line_dash="dash", line_color="white", annotation_text="Actual ToF")
    fig_corr.update_layout(title="Matched Filter Output", xaxis_title="Delay Time (ns)", yaxis_title="Correlation Strength", plot_bgcolor='#000000', paper_bgcolor='#0E1117', font=dict(color='white'), height=350, margin=dict(l=40, r=40, t=40, b=40))
    col_p1.plotly_chart(fig_corr, use_container_width=True)

    fig_aoa = go.Figure()
    fig_aoa.add_trace(go.Scatterpolar(r=[0, 1, 1, 0], theta=[0, actual_aoa_deg, actual_aoa_deg, 0], fill='toself', fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name='Actual Direction'))
    fig_aoa.add_trace(go.Scatterpolar(r=[0, 1], theta=[0, measured_aoa_deg_corrected], mode='lines+markers', line=dict(color='#FF0000' if is_wrapped else '#00FF00', width=5), marker=dict(size=10, color='white'), name='Measured AoA'))
    fig_aoa.update_layout(title="Angle of Arrival (AoA) Radar", polar=dict(radialaxis=dict(visible=False, range=[0, 1]), angularaxis=dict(direction="clockwise", rotation=90, tickmode="array", tickvals=[-90, -45, 0, 45, 90], ticktext=["-90° (Left)", "-45°", "0° (Front)", "45°", "90° (Right)"], gridcolor="#333", linecolor="#555"), bgcolor='#0E1117'), plot_bgcolor='#000000', paper_bgcolor='#0E1117', font=dict(color='white'), height=350, showlegend=False, margin=dict(l=40, r=40, t=40, b=40))
    col_p2.plotly_chart(fig_aoa, use_container_width=True)