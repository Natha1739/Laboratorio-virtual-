"""Laboratorio Virtual Avanzado de Nanomateriales - Teoría DLVO y Cinética.

Simulador de estabilidad coloidal con integración numérica del factor
de estabilidad de Fuchs (W), análisis de sensibilidad 2D y bitácora.

Autor: Colega de Programación e IA
Licencia: MIT
"""

import io
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="Laboratorio Virtual DLVO & Cinética Coloidal",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 14px; border-radius: 8px; border: 1px solid #e2e8f0; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. BASE DE DATOS FÍSICA
# ==============================================================================
NANOMATERIALES = {
    "Dióxido de Silicio (SiO2)": {
        "A132": 0.85e-20,
        "descripcion": "Baja constante de Hamaker. Altamente estable en agua.",
    },
    "Dióxido de Titanio (TiO2)": {
        "A132": 6.0e-20,
        "descripcion": "Atracción vdW moderada-alta. Sensible a electrolitos.",
    },
    "Nanopartículas de Oro (Au)": {
        "A132": 25.0e-20,
        "descripcion": "Atracción vdW muy elevada. Requiere estabilización estérica.",
    },
    "Óxido de Hierro (Fe3O4)": {
        "A132": 3.3e-20,
        "descripcion": "Comportamiento magnético y atracción vdW intermedia.",
    },
}


# ==============================================================================
# 3. MOTOR FÍSICO-MATEMÁTICO: DLVO & INTEGRACIÓN DE FUCHS
# ==============================================================================
def calcular_curva_dlvo(
    zeta_mv: float,
    conc_elec_mm: float,
    ph: float,
    material_nombre: str,
    mecanismo: str,
) -> dict:
    """Calcula el perfil de energía DLVO y el Factor de Estabilidad de Fuchs W."""
    A132 = NANOMATERIALES[material_nombre]["A132"]
    radio_nm = 20.0  # Radio de partícula r_p = 20 nm
    radio_m = radio_nm * 1e-9
    kb_t = 4.11e-21  # k_B * T a 298 K (Joules)

    # Distancia centro a centro r = 2*r_p + h (donde h es la separación de superficie)
    h_nm = np.linspace(0.2, 20.0, 500)
    h_m = h_nm * 1e-9
    r_m = 2.0 * radio_m + h_m

    # 1. Energía Atractiva (Van der Waals)
    v_vdw_j = -(A132 * radio_m) / (12.0 * h_m)
    v_vdw_kbt = v_vdw_j / kb_t

    # 2. Longitud de Debye y Repulsión Electrostática
    kappa = np.sqrt(conc_elec_mm) * 0.33  # nm^-1
    debye_nm = 1.0 / (kappa + 1e-6)

    factor_ph = np.clip((ph - 6.5) / 3.5, -1.0, 1.0)
    zeta_efectivo = zeta_mv * (1.0 + 0.05 * factor_ph)
    zeta_v = zeta_efectivo / 1000.0

    permitividad = 78.5 * 8.854e-12
    v_elec_j = (
        2.0
        * np.pi
        * permitividad
        * radio_m
        * (zeta_v**2)
        * np.exp(-kappa * h_nm)
    )
    v_elec_kbt = v_elec_j / kb_t

    # 3. Impedimento Estérico (No-DLVO)
    v_esterica_kbt = np.zeros_like(h_nm)
    if mecanismo in ["Estérica", "Electroestérica"]:
        grosor_polimero = 3.0  # nm
        v_esterica_kbt = np.where(
            h_nm < grosor_polimero,
            80.0 * ((grosor_polimero - h_nm) / grosor_polimero) ** 2,
            0.0,
        )

    # Energía Total
    v_total_kbt = v_vdw_kbt + v_elec_kbt + v_esterica_kbt

    # Barrera y Mínimos
    barrera = (
        float(np.max(v_total_kbt[h_nm > 0.5]))
        if np.max(v_total_kbt[h_nm > 0.5]) > 0
        else 0.0
    )

    sub_sec = v_total_kbt[(h_nm >= 2.0) & (h_nm <= 8.0)]
    min_secundario = float(np.min(sub_sec)) if len(sub_sec) > 0 else 0.0

    # INTEGRACIÓN NUMÉRICA DE FUCHS (Factor de Estabilidad W)
    # W = 2 * r_p * integral_2rp^inf ( exp(V/kbT) / r^2 ) dr
    integrando = np.exp(np.clip(v_total_kbt, -20, 100)) / (r_m**2)
    fuchs_integral = np.trapz(integrando, r_m)
    fuchs_w = 2.0 * radio_m * fuchs_integral

    # Tiempo estimado de agregación rápida k_fast ~ 1e-17 m^3/s
    # Tiempo de vida ~ W / (k_fast * N_0)
    tiempo_horas = (fuchs_w * 0.05) / 3600.0

    return {
        "h_nm": h_nm,
        "v_vdw": v_vdw_kbt,
        "v_elec": v_elec_kbt,
        "v_esterica": v_esterica_kbt,
        "v_total": v_total_kbt,
        "barrera": barrera,
        "min_secundario": min_secundario,
        "debye_nm": debye_nm,
        "zeta_efectivo": zeta_efectivo,
        "fuchs_w": fuchs_w,
        "tiempo_horas": tiempo_horas,
    }


def generar_matriz_sensibilidad(
    material: str, ph: float, mecanismo: str
) -> tuple:
    """Genera una matriz 2D de la barrera DLVO variando Zeta y Electrolito."""
    zetas = np.linspace(-60, 60, 30)
    elecs = np.linspace(0.1, 80, 30)
    Z_barrera = np.zeros((len(elecs), len(zetas)))

    for i, e in enumerate(elecs):
        for j, z in enumerate(zetas):
            res = calcular_curva_dlvo(z, e, ph, material, mecanismo)
            Z_barrera[i, j] = res["barrera"]

    return zetas, elecs, Z_barrera


# ==============================================================================
# 4. GESTIÓN DEL ESTADO
# ==============================================================================
if "ronda" not in st.session_state:
    st.session_state.ronda = 1
if "bitacora" not in st.session_state:
    st.session_state.bitacora = []
if "vars_previas" not in st.session_state:
    st.session_state.vars_previas = {}
if "ultimo_exp" not in st.session_state:
    st.session_state.ultimo_exp = None
if "puntaje_acumulado" not in st.session_state:
    st.session_state.puntaje_acumulado = 0
if "ultrasonido" not in st.session_state:
    st.session_state.ultrasonido = False

# ==============================================================================
# 5. PANEL DE CONTROL
# ==============================================================================
st.sidebar.header("🎛️ PARÁMETROS DEL EXPERIMENTO")

material_sel = st.sidebar.selectbox(
    "Nanomaterial", list(NANOMATERIALES.keys())
)
st.sidebar.caption(f"ℹ️ {NANOMATERIALES[material_sel]['descripcion']}")

zeta_sel = st.sidebar.slider(
    "Potencial Zeta (mV)", -60.0, 60.0, -30.0, step=1.0
)
elec_sel = st.sidebar.slider(
    "Electrolito NaCl (mM)", 0.1, 100.0, 10.0, step=0.5
)
ph_sel = st.sidebar.slider("pH de la Solución", 3.0, 12.0, 7.0, step=0.1)
mecanismo_sel = st.sidebar.radio(
    "Mecanismo de Estabilización",
    ["Electrostática", "Estérica", "Electroestérica"],
)

vars_act = {
    "material": material_sel,
    "zeta": zeta_sel,
    "elec": elec_sel,
    "ph": ph_sel,
    "mecanismo": mecanismo_sel,
}

modificaciones = 0
if st.session_state.vars_previas:
    for k in vars_act:
        if vars_act[k] != st.session_state.vars_previas.get(k):
            modificaciones += 1

st.sidebar.markdown("---")
st.sidebar.write(f"**Ronda:** {st.session_state.ronda} / 6")
st.sidebar.write(f"**Cambios en ronda:** {modificaciones} / 2")

bloqueado = modificaciones > 2
if bloqueado:
    st.sidebar.error("⚠️ Máximo 2 cambios de variables por ronda.")

btn_ejecutar = st.sidebar.button(
    "🚀 EJECUTAR SIMULACIÓN",
    disabled=bloqueado or (st.session_state.ronda > 6),
)

if btn_ejecutar:
    st.session_state.ultrasonido = False
    ruido = random.uniform(0.97, 1.03)
    res_dlvo = calcular_curva_dlvo(
        zeta_sel, elec_sel, ph_sel, material_sel, mecanismo_sel
    )

    if res_dlvo["barrera"] > 15 or mecanismo_sel != "Electrostática":
        estado = "Suspensión Altamente Estable"
    elif res_dlvo["min_secundario"] < -1.5:
        estado = "Floculación Reversible (Mín. Secundario)"
    else:
        estado = "Coagulación Rápida (Mín. Primario)"

    st.session_state.ultimo_exp = {
        "dlvo": res_dlvo,
        "estado": estado,
        "material": material_sel,
        "ruido": ruido,
    }
    st.session_state.vars_previas = vars_act.copy()

# ==============================================================================
# 6. INTERFAZ Y PESTAÑAS
# ==============================================================================
st.title("🧪 Laboratorio Virtual de Estabilidad Coloidal")
st.caption(
    "Simulador físico-químico interactivo con cinéticas de Fuchs y mapas de estabilidad 2D."
)

tab_lab, tab_sens, tab_bitacora, tab_guia = st.tabs(
    [
        "📊 Experimento & Cinética",
        "🗺️ Mapa de Estabilidad 2D",
        "📝 Bitácora & Reporte",
        "📚 Manual Físico",
    ]
)

with tab_lab:
    if st.session_state.ultimo_exp is None:
        st.info("👈 Configura los parámetros y presiona **EJECUTAR SIMULACIÓN**.")
    else:
        exp = st.session_state.ultimo_exp
        d = exp["dlvo"]

        st.subheader("Resultados Cuantitativos")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Barrera DLVO", f"{d['barrera']:.1f} kBT")
        c2.metric("Factor de Fuchs (Log10 W)", f"{np.log10(max(d['fuchs_w'], 1.0)):.2f}")
        c3.metric(
            "Tiempo de Vida Estimado",
            f"{d['tiempo_horas']:.1f} hrs"
            if d["tiempo_horas"] < 72
            else "> 3 meses",
        )
        c4.metric("Longitud Debye", f"{d['debye_nm']:.2f} nm")

        col_g1, col_g2 = st.columns([1.2, 1.0])

        with col_g1:
            st.markdown("### Perfil de Energía de Interacción DLVO")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                d["h_nm"],
                d["v_vdw"],
                "--",
                color="#e74c3c",
                label="Atracción vdW",
                alpha=0.7,
            )
            ax.plot(
                d["h_nm"],
                d["v_elec"],
                "--",
                color="#3498db",
                label="Repulsión Electrostática",
                alpha=0.7,
            )
            if mecanismo_sel in ["Estérica", "Electroestérica"]:
                ax.plot(
                    d["h_nm"],
                    d["v_esterica"],
                    "--",
                    color="#8e44ad",
                    label="Repulsión Estérica",
                    alpha=0.7,
                )

            ax.plot(
                d["h_nm"],
                d["v_total"],
                "-",
                color="#2c3e50",
                linewidth=2.5,
                label="Energía Total DLVO",
            )
            ax.axhline(0, color="black", linestyle=":", linewidth=0.8)

            ax.set_xlim(0.2, 15)
            ax.set_ylim(-15, max(40, d["barrera"] + 10))
            ax.set_xlabel("Separación de superficie h (nm)")
            ax.set_ylabel("Energía de Interacción (kBT)")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)

        with col_g2:
            st.markdown("### Dispersión y Reversibilidad")
            fig_sim, ax_sim = plt.subplots(figsize=(5, 3.6))
            ax_sim.set_facecolor("#f8fafc")

            np.random.seed(42)
            n_p = 20

            if st.session_state.ultrasonido and (
                d["barrera"] > 5 or d["min_secundario"] < 0
            ):
                x_p = np.random.uniform(1, 9, n_p)
                y_p = np.random.uniform(1, 9, n_p)
                color_p = "#3498db"
                tit = "Redispersión por Ultrasonido"
            elif exp["estado"] == "Suspensión Altamente Estable":
                x_p = np.random.uniform(1, 9, n_p)
                y_p = np.random.uniform(1, 9, n_p)
                color_p = "#2ecc71"
                tit = exp["estado"]
            elif exp["estado"] == "Floculación Reversible (Mín. Secundario)":
                x_p = np.concatenate(
                    [
                        np.random.normal(3, 0.6, 10),
                        np.random.normal(7, 0.6, 10),
                    ]
                )
                y_p = np.concatenate(
                    [
                        np.random.normal(3, 0.6, 10),
                        np.random.normal(7, 0.6, 10),
                    ]
                )
                color_p = "#f39c12"
                tit = exp["estado"]
            else:
                x_p = np.random.normal(5, 0.3, n_p)
                y_p = np.random.normal(5, 0.3, n_p)
                color_p = "#e74c3c"
                tit = exp["estado"]

            ax_sim.scatter(
                x_p, y_p, color=color_p, s=120, edgecolors="black", zorder=3
            )
            ax_sim.set_xlim(0, 10)
            ax_sim.set_ylim(0, 10)
            ax_sim.set_xticks([])
            ax_sim.set_yticks([])
            ax_sim.set_title(tit, fontsize=10)
            st.pyplot(fig_sim)

            if st.button("🔊 Prueba de Ultrasonido"):
                st.session_state.ultrasonido = True
                if d["barrera"] < 2 and d["min_secundario"] >= 0:
                    st.error(
                        "❌ Agregación IRREVERSIBLE en pozo primario. La agitación mecánica no redispersa."
                    )
                else:
                    st.success(
                        "✅ Agregación REVERSIBLE. La energía acústica redispersa el coloide."
                    )

with tab_sens:
    st.subheader("🗺️ Mapa de Sensibilidad 2D: Zona de Estabilidad Coloidal")
    st.write(
        "Este mapa evalúa la barrera energética ($k_BT$) al variar simultáneamente el **Potencial Zeta** y la **Concentración de Electrolito**."
    )

    if st.button("📊 Generar Mapa 2D de Estabilidad"):
        with st.spinner("Calculando superficie de energía DLVO..."):
            zetas_m, elecs_m, Z_map = generar_matriz_sensibilidad(
                material_sel, ph_sel, mecanismo_sel
            )

            fig_map, ax_map = plt.subplots(figsize=(7, 4.5))
            c = ax_map.contourf(
                zetas_m, elecs_m, Z_map, levels=15, cmap="YlGnBu"
            )
            fig_map.colorbar(c, label="Barrera DLVO (kBT)")
            ax_map.axvline(
                zeta_sel,
                color="red",
                linestyle="--",
                label=f"Zeta Actual ({zeta_sel} mV)",
            )
            ax_map.axhline(
                elec_sel,
                color="orange",
                linestyle="--",
                label=f"Electrolito Actual ({elec_sel} mM)",
            )
            ax_map.set_xlabel("Potencial Zeta (mV)")
            ax_map.set_ylabel("Concentración NaCl (mM)")
            ax_map.legend(fontsize=8, loc="upper right")
            st.pyplot(fig_map)

with tab_bitacora:
    st.subheader("Registro Experimental")

    if st.session_state.ultimo_exp is None:
        st.warning("Ejecuta una simulación primero.")
    else:
        with st.form(f"form_ronda_{st.session_state.ronda}"):
            p_hip = st.text_area(
                "Conclusión e Hipótesis para el siguiente paso:"
            )
            btn_guardar = st.form_submit_button("💾 Registrar en Bitácora")

            if btn_guardar:
                st.session_state.bitacora.append(
                    {
                        "Ronda": st.session_state.ronda,
                        "Material": material_sel,
                        "Zeta (mV)": round(
                            st.session_state.ultimo_exp["dlvo"][
                                "zeta_efectivo"
                            ],
                            1,
                        ),
                        "Electrolito (mM)": elec_sel,
                        "Barrera (kBT)": round(
                            st.session_state.ultimo_exp["dlvo"]["barrera"], 1
                        ),
                        "Log10 W (Fuchs)": round(
                            np.log10(
                                max(
                                    st.session_state.ultimo_exp["dlvo"][
                                        "fuchs_w"
                                    ],
                                    1.0,
                                )
                            ),
                            2,
                        ),
                        "Estado": st.session_state.ultimo_exp["estado"],
                        "Hipótesis": p_hip if p_hip else "Sin registro",
                    }
                )
                st.session_state.puntaje_acumulado += 15
                if st.session_state.ronda < 6:
                    st.session_state.ronda += 1
                    st.success("¡Registrado con éxito!")

    if len(st.session_state.bitacora) > 0:
        df_bit = pd.DataFrame(st.session_state.bitacora)
        st.dataframe(df_bit, use_container_width=True)

        csv_data = df_bit.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar Bitácora (.CSV)",
            csv_data,
            "bitacora_dlvo_fuchs.csv",
            "text/csv",
        )

with tab_guia:
    st.subheader("📚 Fundamentos Teóricos: Factor de Fuchs")
    st.markdown(
        """
    La relación entre la cinética de coagulación y el potencial de interacción se describe mediante el **Factor de Estabilidad de Fuchs ($W$)**:
    
    $$W = 2a \\int_{2a}^{\\infty} \\frac{\\exp\\left(\\frac{V_{\\text{total}}(r)}{k_B T}\\right)}{r^2} dr$$
    
    * **Si $W \\approx 1$ ($|\\log W| \\to 0$):** Coagulación rápida controlada por difusión (sin barrera repulsiva).
    * **Si $W \\gg 10^5$ ($\\log W > 5$):** Suspensión altamente estable a largo plazo.
    """
    )
