"""Laboratorio Virtual de Nanomateriales - Teoría DLVO.

Aplicación interactiva construida en Python puro con Streamlit, NumPy,
Matplotlib y Pandas para la simulación física de estabilidad coloidal.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="Laboratorio Virtual de Teoría DLVO",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. BASE DE DATOS FÍSICA DE NANOMATERIALES
# ==============================================================================
NANOMATERIALES = {
    "Dióxido de Silicio (SiO2)": {
        "A132": 0.85e-20,
        "descripcion": "Atracción vdW muy baja. Alta estabilidad natural en agua.",
    },
    "Dióxido de Titanio (TiO2)": {
        "A132": 6.0e-20,
        "descripcion": "Atracción vdW elevada. Propenso a la coagulación rápida.",
    },
    "Nanopartículas de Oro (Au)": {
        "A132": 25.0e-20,
        "descripcion": "Fuerzas de van der Waals extremadamente altas.",
    },
    "Óxido de Hierro (Fe3O4)": {
        "A132": 3.3e-20,
        "descripcion": "Atracción vdW moderada-alta en medio acuoso.",
    },
}


# ==============================================================================
# 3. MOTOR DE CÁLCULO FÍSICO-MATEMÁTICO (DLVO)
# ==============================================================================
def calcular_interacciones_dlvo(
    zeta_mv: float,
    conc_elec_mm: float,
    ph: float,
    material_nombre: str,
    mecanismo: str,
) -> dict:
    """Calcula las curvas de energía de atracción de van der Waals, repulsión

    electrostática e impedimento estérico según el modelo DLVO.
    """
    A132 = NANOMATERIALES[material_nombre]["A132"]
    radio_p = 20.0  # Radio de la partícula en nm
    distancia = np.linspace(0.3, 15.0, 350)  # Distancia interpartícula en nm
    kb_t = 4.11e-21  # Energía térmica k_B * T en Joules

    # 1. Atracción de Van der Waals (V_vdW)
    v_vdw_j = -(A132 * (radio_p * 1e-9)) / (12.0 * (distancia * 1e-9))
    v_vdw_kbt = v_vdw_j / kb_t

    # 2. Longitud de Debye (compresión de doble capa por fuerza iónica)
    kappa = np.sqrt(conc_elec_mm) * 0.33
    debye_nm = 1.0 / (kappa + 1e-6)

    factor_ph = np.clip((ph - 6.5) / 3.5, -1.0, 1.0)
    zeta_efectivo = zeta_mv * (1.0 + 0.05 * factor_ph)

    # 3. Repulsión Electrostática (V_elec)
    permitividad = 78.5 * 8.854e-12
    zeta_v = zeta_efectivo / 1000.0
    v_elec_j = (
        2.0
        * np.pi
        * permitividad
        * (radio_p * 1e-9)
        * (zeta_v**2)
        * np.exp(-kappa * distancia)
    )
    v_elec_kbt = v_elec_j / kb_t

    # 4. Repulsión Estérica (Mecanismo adicional No-DLVO)
    v_esterica_kbt = np.zeros_like(distancia)
    if mecanismo in ["Estérica", "Electroestérica"]:
        grosor_polimero = 3.0  # nm
        v_esterica_kbt = np.where(
            distancia < grosor_polimero,
            75.0 * ((grosor_polimero - distancia) / grosor_polimero) ** 2,
            0.0,
        )

    # Energía Total DLVO
    v_total_kbt = v_vdw_kbt + v_elec_kbt + v_esterica_kbt

    # Análisis de barrera y mínimos de energía
    filtro_dist = distancia > 0.6
    d_eval = distancia[filtro_dist]
    v_eval = v_total_kbt[filtro_dist]

    barrera = float(np.max(v_eval)) if np.max(v_eval) > 0 else 0.0
    min_primario = float(np.min(v_eval[d_eval < 1.8]))

    sub_secundario = v_eval[(d_eval >= 2.0) & (d_eval <= 7.0)]
    min_secundario = (
        float(np.min(sub_secundario)) if len(sub_secundario) > 0 else 0.0
    )

    return {
        "distancia": distancia,
        "v_vdw": v_vdw_kbt,
        "v_elec": v_elec_kbt,
        "v_esterica": v_esterica_kbt,
        "v_total": v_total_kbt,
        "barrera": barrera,
        "min_primario": min_primario,
        "min_secundario": min_secundario,
        "debye_nm": debye_nm,
        "zeta_efectivo": zeta_efectivo,
    }


def calcular_indice_estabilidad(
    barrera: float,
    zeta: float,
    conc_elec: float,
    mecanismo: str,
    ruido: float,
) -> float:
    """Calcula un índice numérico de estabilidad coloidal (0 a 100)."""
    score_barrera = np.clip(barrera * 2.5, 0, 50)
    score_zeta = np.clip(abs(zeta) * 0.4, 0, 25)
    score_elec = np.clip(25.0 - (conc_elec * 0.22), 0, 25)

    base = score_barrera + score_zeta + score_elec
    if mecanismo in ["Estérica", "Electroestérica"]:
        base = max(base, 82.0)

    return float(np.clip(base * ruido, 0, 100))


# ==============================================================================
# 4. ESTADO DE LA SESIÓN (SESSION STATE)
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
if "ultrasonido_aplicado" not in st.session_state:
    st.session_state.ultrasonido_aplicado = False

# ==============================================================================
# 5. PANEL DE CONTROL LATERAL
# ==============================================================================
st.sidebar.header("🎛️ CONTROL DE LA SUSPENSIÓN")

material_sel = st.sidebar.selectbox(
    "Nanomaterial (Constante de Hamaker)", list(NANOMATERIALES.keys())
)
st.sidebar.caption(
    f"ℹ️️ {NANOMATERIALES[material_sel]['descripcion']} (A132 = {NANOMATERIALES[material_sel]['A132']:.1e} J)"
)

zeta_sel = st.sidebar.slider(
    "Potencial Zeta (mV)", -60.0, 60.0, -25.0, step=1.0
)
elec_sel = st.sidebar.slider(
    "Concentración Electrolito (mM)", 0.1, 100.0, 5.0, step=0.5
)
ph_sel = st.sidebar.slider("pH del Medio", 3.0, 12.0, 7.0, step=0.1)
conc_part_sel = st.sidebar.slider(
    "Concentración Partículas (mg/mL)", 0.01, 10.0, 1.0, step=0.1
)
mecanismo_sel = st.sidebar.radio(
    "Mecanismo de Estabilización",
    ["Electrostática", "Estérica", "Electroestérica"],
)

# Validación de máximo 2 modificaciones por ronda
vars_actuales = {
    "material": material_sel,
    "zeta": zeta_sel,
    "elec": elec_sel,
    "ph": ph_sel,
    "conc_part": conc_part_sel,
    "mecanismo": mecanismo_sel,
}

modificaciones = 0
if st.session_state.vars_previas:
    for k in vars_actuales:
        if vars_actuales[k] != st.session_state.vars_previas.get(k):
            modificaciones += 1

st.sidebar.markdown("---")
st.sidebar.write(f"**Ronda Experimental:** {st.session_state.ronda} / 6")
st.sidebar.write(
    f"**Variables modificadas en la ronda:** {modificaciones} / 2"
)

deshabilitar_ejecucion = False
if modificaciones > 2:
    st.sidebar.error(
        "⚠️ **Límite experimental excedido.** Solo puedes modificar hasta 2 variables por ronda."
    )
    deshabilitar_ejecucion = True

# ==============================================================================
# 6. EJECUCIÓN EXPERIMENTAL
# ==============================================================================
btn_ejecutar = st.sidebar.button(
    "🚀 EJECUTAR EXPERIMENTO",
    disabled=deshabilitar_ejecucion or (st.session_state.ronda > 6),
)

if btn_ejecutar:
    st.session_state.ultrasonido_aplicado = False
    ruido_exp = random.uniform(0.95, 1.05)

    dlvo_res = calcular_interacciones_dlvo(
        zeta_sel, elec_sel, ph_sel, material_sel, mecanismo_sel
    )
    indice_est = calcular_indice_estabilidad(
        dlvo_res["barrera"],
        dlvo_res["zeta_efectivo"],
        elec_sel,
        mecanismo_sel,
        ruido_exp,
    )

    if indice_est < 25:
        estado_coloidal = "Coagulación Fuerte (Mínimo Primario)"
        tam_agregados = "Grande (10 - 50 µm)"
    elif indice_est < 60:
        estado_coloidal = "Floculación Reversible (Mínimo Secundario)"
        tam_agregados = "Moderado (2 - 10 µm)"
    else:
        estado_coloidal = "Suspensión Estable / Redispersa"
        tam_agregados = "Nanométrico (< 200 nm)"

    pct_disp = float(
        np.clip(
            (
                indice_est * 0.88
                + (12.0 if mecanismo_sel != "Electrostática" else 0.0)
            )
            * ruido_exp,
            4,
            98,
        )
    )

    st.session_state.ultimo_exp = {
        "dlvo": dlvo_res,
        "indice": indice_est,
        "pct_dispersas": pct_disp,
        "estado": estado_coloidal,
        "tam_agregados": tam_agregados,
        "material": material_sel,
    }
    st.session_state.vars_previas = vars_actuales.copy()

# ==============================================================================
# 7. INTERFAZ PRINCIPAL
# ==============================================================================
st.title("🧪 Laboratorio Virtual de Estabilización Coloidal")
st.caption(
    "Simulador científico universitario basado en la Teoría DLVO y Mecanismos de Estabilización."
)

tab_lab, tab_bitacora, tab_eval, tab_guia = st.tabs(
    [
        "📊 Laboratorio Virtual",
        "📝 Bitácora & Reportes",
        "🎯 Desafío Final",
        "📚 Manual Teórico",
    ]
)

with tab_lab:
    if st.session_state.ultimo_exp is None:
        st.info(
            "👉 Ajusta los parámetros en el panel izquierdo y presiona **EJECUTAR EXPERIMENTO**."
        )
    else:
        exp = st.session_state.ultimo_exp
        data = exp["dlvo"]

        st.subheader("Resultados Fisicoquímicos")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Índice de Estabilidad", f"{exp['indice']:.1f} / 100")
        col_m2.metric(
            "Potencial Zeta Efectivo", f"{data['zeta_efectivo']:.1f} mV"
        )
        col_m3.metric("Barrera de Energía", f"{data['barrera']:.1f} kBT")
        col_m4.metric("Partículas Dispersas", f"{exp['pct_dispersas']:.1f} %")

        col_c1, col_c2 = st.columns([1.2, 1.0])

        with col_c1:
            st.markdown("### Curva DLVO de Energía de Interacción")
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.plot(
                data["distancia"],
                data["v_vdw"],
                "--",
                color="#e74c3c",
                label="Atracción vdW",
                alpha=0.8,
            )
            ax.plot(
                data["distancia"],
                data["v_elec"],
                "--",
                color="#3498db",
                label="Repulsión Electrostática",
                alpha=0.8,
            )
            if mecanismo_sel in ["Estérica", "Electroestérica"]:
                ax.plot(
                    data["distancia"],
                    data["v_esterica"],
                    "--",
                    color="#8e44ad",
                    label="Repulsión Estérica",
                    alpha=0.8,
                )

            ax.plot(
                data["distancia"],
                data["v_total"],
                "-",
                color="#2c3e50",
                linewidth=2.5,
                label="Energía Total",
            )
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

            ax.set_xlim(0.3, 12)
            ax.set_ylim(-20, max(45, data["barrera"] + 10))
            ax.set_xlabel("Distancia entre partículas d (nm)")
            ax.set_ylabel("Energía de Interacción Total (kBT)")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig)

        with col_c2:
            st.markdown("### Estado Visual y Reversibilidad")
            fig_sim, ax_sim = plt.subplots(figsize=(5, 3.8))
            ax_sim.set_facecolor("#f1f5f9")

            np.random.seed(10)
            n_p = 25

            if st.session_state.ultrasonido_aplicado and (
                exp["indice"] >= 25 or data["min_secundario"] < 0
            ):
                x_p = np.random.uniform(1, 9, n_p)
                y_p = np.random.uniform(1, 9, n_p)
                c_p = "#3498db"
                estado_vis = "Redispersión Temporal por Ultrasonido"
            elif exp["indice"] > 60:
                x_p = np.random.uniform(1, 9, n_p)
                y_p = np.random.uniform(1, 9, n_p)
                c_p = "#2ecc71"
                estado_vis = exp["estado"]
            elif exp["indice"] > 25:
                x_p = np.concatenate(
                    [
                        np.random.normal(3, 0.5, 12),
                        np.random.normal(7, 0.5, 13),
                    ]
                )
                y_p = np.concatenate(
                    [
                        np.random.normal(3, 0.5, 12),
                        np.random.normal(7, 0.5, 13),
                    ]
                )
                c_p = "#f39c12"
                estado_vis = exp["estado"]
            else:
                x_p = np.random.normal(5, 0.35, n_p)
                y_p = np.random.normal(5, 0.35, n_p)
                c_p = "#e74c3c"
                estado_vis = exp["estado"]

            debye_r = 0.25 + (data["debye_nm"] * 0.08)
            for x, y in zip(x_p, y_p):
                ax_sim.add_patch(
                    plt.Circle(
                        (x, y), debye_r, color="#3498db", alpha=0.20, zorder=2
                    )
                )

            ax_sim.scatter(
                x_p, y_p, color=c_p, s=110, edgecolors="black", zorder=3
            )
            ax_sim.set_xlim(0, 10)
            ax_sim.set_ylim(0, 10)
            ax_sim.set_xticks([])
            ax_sim.set_yticks([])
            ax_sim.set_title(estado_vis, fontsize=10)
            st.pyplot(fig_sim)

            if st.button("🔊 Aplicar Ultrasonido (Prueba de Reversibilidad)"):
                st.session_state.ultrasonido_aplicado = True
                if exp["indice"] < 25:
                    st.error(
                        "❌ La agregación en el Mínimo Primario es IRREVERSIBLE. El ultrasonido no logra dispersar las partículas."
                    )
                else:
                    st.success(
                        "✅ La floculación en el Mínimo Secundario es REVERSIBLE. Las partículas se han redispersado temporalmente."
                    )

with tab_bitacora:
    st.subheader("Interpretación Experimental y Registro en Bitácora")

    if st.session_state.ultimo_exp is None:
        st.warning(
            "Debes ejecutar un experimento primero para habilitar esta sección."
        )
    else:
        with st.form(f"form_ronda_{st.session_state.ronda}"):
            p1 = st.radio(
                "1. Diagnóstico de la Suspensión:",
                ["Estable", "Baja Estabilidad / Floculada", "Inestable / Coagulada"],
            )
            p2 = st.radio(
                "2. Estado del balance de fuerzas:",
                [
                    "La repulsión domina debido a una barrera DLVO alta",
                    "La atracción de van der Waals domina (barrera nula)",
                    "El impedimento estérico compensa la falta de carga",
                ],
            )
            p3 = st.text_area(
                "3. Hipótesis del Experimento (Justificación de cambios para la siguiente ronda):"
            )

            btn_guardar = st.form_submit_button("💾 Guardar en Bitácora")

            if btn_guardar:
                reg = {
                    "Ronda": st.session_state.ronda,
                    "Material": exp["material"],
                    "Zeta (mV)": round(exp["dlvo"]["zeta_efectivo"], 1),
                    "Electrolito (mM)": elec_sel,
                    "Barrera (kBT)": round(exp["dlvo"]["barrera"], 1),
                    "Estado Coloidal": exp["estado"],
                    "Índice Estabilidad": round(exp["indice"], 1),
                    "Hipótesis del Estudiante": p3
                    if p3
                    else "Sin justificación",
                }
                st.session_state.bitacora.append(reg)
                st.session_state.puntaje_acumulado += 15

                if st.session_state.ronda < 6:
                    st.session_state.ronda += 1
                    st.success(
                        "¡Experimento e hipótesis registrados en la bitácora!"
                    )
                else:
                    st.balloons()
                    st.success(
                        "¡Has completado las 6 rondas del laboratorio virtual!"
                    )

    st.markdown("---")
    st.subheader("📖 BITÁCORA DEL LABORATORIO")

    if len(st.session_state.bitacora) > 0:
        df_bitacora = pd.DataFrame(st.session_state.bitacora)
        st.dataframe(df_bitacora, use_container_width=True)

        col_exp1, col_exp2 = st.columns(2)

        csv_buffer = df_bitacora.to_csv(index=False).encode("utf-8")
        col_exp1.download_button(
            label="📥 Descargar Bitácora (.CSV)",
            data=csv_buffer,
            file_name="bitacora_laboratorio_dlvo.csv",
            mime="text/csv",
        )

        reporte_txt = f"# REPORTE DE LABORATORIO VIRTUAL DLVO\n\n"
        reporte_txt += f"Puntuación de Razonamiento Científico: {min(100, st.session_state.puntaje_acumulado)} / 100 pts\n"
        reporte_txt += f"Total de Rondas Ejecutadas: {len(st.session_state.bitacora)}\n\n"
        reporte_txt += "## RESUMEN DE RONDAS EXPERIMENTALES\n"
        for idx, row in df_bitacora.iterrows():
            reporte_txt += f"### Ronda {row['Ronda']} - {row['Material']}\n"
            reporte_txt += f"- Potencial Zeta: {row['Zeta (mV)']} mV | Electrolito: {row['Electrolito (mM)']} mM\n"
            reporte_txt += f"- Barrera DLVO: {row['Barrera (kBT)']} kBT | Índice: {row['Índice Estabilidad']}/100\n"
            reporte_txt += f"- Hipótesis: {row['Hipótesis del Estudiante']}\n\n"

        col_exp2.download_button(
            label="📄 Descargar Informe Académico (.TXT)",
            data=reporte_txt,
            file_name="informe_laboratorio_dlvo.txt",
            mime="text/plain",
        )
    else:
        st.info("La bitácora está vacía. Ejecuta e interpreta experimentos.")

with tab_eval:
    st.subheader("🎯 Desafío Final: Ordenamiento y Justificación DLVO")
    st.markdown(
        """
    Evalúa las tres suspensiones coloidales planteadas:
    * **Suspensión A:** Potencial zeta bajo ($-8\\text{ mV}$) y alta fuerza iónica ($80\\text{ mM}$).
    * **Suspensión B:** Elevado potencial zeta ($-50\\text{ mV}$) y baja fuerza iónica ($1\\text{ mM}$).
    * **Suspensión C:** Potencial zeta moderado ($-15\\text{ mV}$) con estabilización estérica (polímero).
    """
    )

    with st.form("form_desafio_final"):
        orden_alumno = st.multiselect(
            "Ordena las suspensiones de MENOR a MAYOR tendencia a la agregación (la más estable al final):",
            ["Suspensión A", "Suspensión B", "Suspensión C"],
            default=["Suspensión A", "Suspensión C", "Suspensión B"],
        )
        justificacion_txt = st.text_area(
            "Justificación Científica (Integra compresión de la doble capa, barrera energética e impedimento estérico):"
        )

        btn_evaluar = st.form_submit_button("🏆 Enviar Evaluación Final")

        if btn_evaluar:
            puntaje_final = min(100, st.session_state.puntaje_acumulado + 25)
            st.markdown(f"## Puntuación Final: **{puntaje_final} / 100 pts**")
            st.info(
                """
            **Análisis Físico:**
            * **Suspensión A (Menor Estabilidad):** La alta fuerza iónica comprime la doble capa eléctrica, anulando la repulsión electrostática.
            * **Suspensión B (Alta Estabilidad Electrostática):** La doble capa se expande y el alto potencial zeta genera una gran barrera repulsiva DLVO.
            * **Suspensión C (Alta Estabilidad Estérica):** Las cadenas poliméricas crean un impedimento físico que previene la coagulación independientemente de la carga superficial.
            """
            )

with tab_guia:
    st.subheader("📚 Fundamentos Teóricos de la Teoría DLVO")
    st.markdown(
        """
    La teoría **DLVO** explica la estabilidad coloidal mediante el balance entre fuerzas de atracción de van der Waals ($V_{\\text{vdW}}$) y repulsión electrostática de doble capa ($V_{\\text{elec}}$):
    
    $$V_{\\text{total}}(d) = V_{\\text{vdW}}(d) + V_{\\text{elec}}(d) + V_{\\text{estérica}}(d)$$
    
    ### Mecanismos Clave:
    1. **Mínimo Primario:** Atrapamiento irreversible a distancias muy cortas.
    2. **Mínimo Secundario:** Agregación débil y reversible que permite la floculación/redispersión.
    3. **Compresión de la Doble Capa:** Un aumento en los iones del medio (fuerza iónica) reduce el alcance de la repulsión electrostática.
    """
    )
