"""Laboratorio Virtual de Teoría DLVO y Estabilización de Suspensiones

Coloidales.

Aplicación interactiva construida en Streamlit para simulación científica
de interacciones coloidales, curvas de energía y bitácora experimental.
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL DE LA APLICACIÓN
# ==============================================================================
st.set_page_config(
    page_title="Laboratorio Virtual DLVO",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 2. FUNCIONES DE MODELADO FÍSICO Y MATEMÁTICO (TEORÍA DLVO)
# ==============================================================================
def calcular_curvas_dlvo(
    zeta_mv: float,
    conc_elec_mm: float,
    ph: float,
    hamaker_nivel: str,
    mecanismo: str,
) -> dict:
    """Calcula las componentes atractivas y repulsivas del modelo DLVO.

    Parámetros:
        zeta_mv: Potencial zeta en milivoltios (-60 a +60 mV).
        conc_elec_mm: Concentración de electrolito en mM (0.1 a 100 mM).
        ph: Nivel de pH del medio (3 a 12).
        hamaker_nivel: Magnitud de la constante de Hamaker ('Baja', 'Media',
          'Alta').
        mecanismo: Tipo de estabilización ('Electrostática', 'Estérica',
          'Electroestérica').

    Retorna:
        Un diccionario con los arreglos de distancia, energías de interacción y
        métricas clave.
    """
    # Mapeo de la Constante de Hamaker (A) en Joules
    hamaker_map = {"Baja": 0.8e-20, "Media": 1.5e-20, "Alta": 3.2e-20}
    constante_a = hamaker_map.get(hamaker_nivel, 1.5e-20)

    # Radio equivalente de partícula (nm) y vector de distancias (nm)
    radio_nm = 20.0
    distancia_nm = np.linspace(0.4, 15.0, 300)

    # 1. Componente Atractiva de Van der Waals (V_vdW)
    # V_vdW = - (A * R) / (12 * d)
    v_vdw_joules = -(constante_a * (radio_nm * 1e-9)) / (
        12.0 * (distancia_nm * 1e-9)
    )
    v_vdw_kbt = v_vdw_joules / 4.11e-21  # Normalizado a unidades k_B * T

    # 2. Longitud de Debye y Fuerza Iónica (Compresión de la doble capa)
    # kappa proporcional a la raíz cuadrada de la concentración de electrolito
    kappa_inv_nm = 1.0 / (np.sqrt(conc_elec_mm) * 0.35 + 1e-5)

    # Ajuste cualitativo del potencial zeta efectivo según el pH
    factor_ph = np.clip((ph - 6.0) / 4.0, -1.2, 1.2)
    zeta_efectivo = zeta_mv * (1.0 + 0.08 * factor_ph)

    # 3. Componente Repulsiva Electrostática (V_elec)
    permitividad = 78.5 * 8.854e-12  # Medio acuoso
    zeta_volts = zeta_efectivo / 1000.0
    v_elec_joules = (
        2.0
        * np.pi
        * permitividad
        * (radio_nm * 1e-9)
        * (zeta_volts**2)
        * np.exp(-distancia_nm / kappa_inv_nm)
    )
    v_elec_kbt = v_elec_joules / 4.11e-21

    # 4. Componente Repulsiva Estérica / Electroestérica (No DLVO clásica)
    v_esterica_kbt = np.zeros_like(distancia_nm)
    if mecanismo in ["Estérica", "Electroestérica"]:
        # Resistencia elástica/entrópica a distancias cortas (< 3.5 nm)
        v_esterica_kbt = np.where(
            distancia_nm < 3.5, 65.0 * (3.5 - distancia_nm) ** 2, 0.0
        )

    # Energía Total DLVO
    v_total_kbt = v_vdw_kbt + v_elec_kbt + v_esterica_kbt

    # Identificación de Barrera Energética y Mínimos
    mascara_valida = distancia_nm > 0.8
    dist_sub = distancia_nm[mascara_valida]
    v_sub = v_total_kbt[mascara_valida]

    barrera_energia = float(np.max(v_sub)) if np.max(v_sub) > 0 else 0.0
    min_primario = float(np.min(v_sub[dist_sub < 2.0]))

    region_secundaria = v_sub[(dist_sub >= 2.0) & (dist_sub <= 8.0)]
    min_secundario = (
        float(np.min(region_secundaria)) if len(region_secundaria) > 0 else 0.0
    )

    return {
        "distancia": distancia_nm,
        "v_vdw": v_vdw_kbt,
        "v_elec": v_elec_kbt,
        "v_esterica": v_esterica_kbt,
        "v_total": v_total_kbt,
        "barrera": barrera_energia,
        "min_primario": min_primario,
        "min_secundario": min_secundario,
        "debye_nm": kappa_inv_nm,
        "zeta_efectivo": zeta_efectivo,
    }


def calcular_indice_estabilidad(
    barrera: float, zeta: float, elec_mm: float, mecanismo: str
) -> float:
    """Calcula el índice cuantitativo de estabilidad coloidal (0 a 100)."""
    score_barrera = np.clip(barrera * 2.8, 0, 50)
    score_zeta = np.clip(abs(zeta) * 0.45, 0, 25)
    score_elec = np.clip(25.0 - (elec_mm * 0.25), 0, 25)

    indice = score_barrera + score_zeta + score_elec

    if mecanismo in ["Estérica", "Electroestérica"]:
        indice = max(
            indice, 80.0
        )  # La estabilización estérica proporciona alta estabilidad

    return float(np.clip(indice, 0, 100))


# ==============================================================================
# 3. GESTIÓN DEL ESTADO DE SESIÓN (SESSION STATE)
# ==============================================================================
if "ronda" not in st.session_state:
    st.session_state.ronda = 1
if "bitacora" not in st.session_state:
    st.session_state.bitacora = []
if "variables_previas" not in st.session_state:
    st.session_state.variables_previas = {}
if "ultimo_resultado" not in st.session_state:
    st.session_state.ultimo_resultado = None
if "puntaje" not in st.session_state:
    st.session_state.puntaje = 0

# ==============================================================================
# 4. INTERFAZ Y PANEL DE CONTROL
# ==============================================================================
st.title("🧪 Laboratorio Virtual de Nanomateriales: Teoría DLVO")
st.markdown(
    "Simulador interactivo para analizar el balance de fuerzas interpartícula y la estabilidad de suspensiones coloidales."
)

st.sidebar.header("🎛️ CONTROL DE LA SUSPENSIÓN")

zeta_input = st.sidebar.slider(
    "Potencial Zeta (mV)", -60.0, 60.0, -25.0, step=1.0
)
elec_input = st.sidebar.slider(
    "Concentración de Electrolito (mM)", 0.1, 100.0, 5.0, step=0.5
)
ph_input = st.sidebar.slider("pH del Medio", 3.0, 12.0, 7.0, step=0.1)
conc_part_input = st.sidebar.slider(
    "Concentración Partículas (mg/mL)", 0.01, 10.0, 1.0, step=0.1
)
hamaker_input = st.sidebar.select_slider(
    "Constante de Hamaker", options=["Baja", "Media", "Alta"], value="Media"
)
mecanismo_input = st.sidebar.radio(
    "Mecanismo de Estabilización",
    ["Electrostática", "Estérica", "Electroestérica"],
)

# Control de límite de variables modificadas por ronda (máximo 2)
vars_actuales = {
    "zeta": zeta_input,
    "elec": elec_input,
    "ph": ph_input,
    "conc_part": conc_part_input,
    "hamaker": hamaker_input,
    "mecanismo": mecanismo_input,
}

cambios = 0
if st.session_state.variables_previas:
    for k in vars_actuales:
        if vars_actuales[k] != st.session_state.variables_previas.get(k):
            cambios += 1

st.sidebar.markdown("---")
st.sidebar.write(f"**Ronda experimental:** {st.session_state.ronda} / 6")
st.sidebar.write(f"**Variables modificadas en esta ronda:** {cambios} / 2")

bloquear_boton = False
if cambios > 2:
    st.sidebar.error(
        "⚠️ **Límite experimental excedido.** Solo puedes modificar un máximo de 2 variables por ronda."
    )
    bloquear_boton = True

# ==============================================================================
# 5. EJECUCIÓN EXPERIMENTAL
# ==============================================================================
ejecutar = st.sidebar.button(
    "🚀 EJECUTAR EXPERIMENTO",
    disabled=bloquear_boton or st.session_state.ronda > 6,
)

if ejecutar:
    # Ruido experimental de ±5%
    factor_variabilidad = random.uniform(0.95, 1.05)

    dlvo = calcular_curvas_dlvo(
        zeta_input, elec_input, ph_input, hamaker_input, mecanismo_input
    )
    indice = (
        calcular_indice_estabilidad(
            dlvo["barrera"], dlvo["zeta_efectivo"], elec_input, mecanismo_input
        )
        * factor_variabilidad
    )
    indice = float(np.clip(indice, 0, 100))

    if indice < 25:
        tendencia = "Coagulación Fuerte (Mínimo Primario)"
        tamano_agregados = "Grande (10 - 50 µm)"
    elif indice < 60:
        tendencia = "Floculación Reversible (Mínimo Secundario)"
        tamano_agregados = "Moderado (2 - 10 µm)"
    else:
        tendencia = "Suspensión Estable / Dispersa"
        tamano_agregados = "Nanométrico (< 200 nm)"

    pct_dispersas = float(
        np.clip(
            (
                indice * 0.85
                + (15.0 if mecanismo_input != "Electrostática" else 0.0)
            )
            * factor_variabilidad,
            5,
            98,
        )
    )

    st.session_state.ultimo_resultado = {
        "dlvo": dlvo,
        "indice": indice,
        "pct_dispersas": pct_dispersas,
        "tendencia": tendencia,
        "tamano_agregados": tamano_agregados,
    }

    st.session_state.variables_previas = vars_actuales.copy()

# ==============================================================================
# 6. PESTAÑAS DE TRABAJO
# ==============================================================================
tab_lab, tab_bitacora, tab_eval, tab_teoria = st.tabs(
    [
        "📊 Visualización & Simulación",
        "📝 Registro & Bitácora",
        "🎯 Desafío Final",
        "📚 Conceptos DLVO",
    ]
)

with tab_lab:
    if st.session_state.ultimo_resultado is None:
        st.info(
            "👉 Configura las variables en el panel lateral y presiona **EJECUTAR EXPERIMENTO**."
        )
    else:
        res = st.session_state.ultimo_resultado
        data_dlvo = res["dlvo"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Índice de Estabilidad", f"{res['indice']:.1f} / 100")
        col2.metric(
            "Potencial Zeta Efectivo", f"{data_dlvo['zeta_efectivo']:.1f} mV"
        )
        col3.metric("Barrera DLVO", f"{data_dlvo['barrera']:.1f} kBT")
        col4.metric("Partículas Dispersas", f"{res['pct_dispersas']:.1f} %")

        col_graf, col_sim = st.columns([1.2, 1.0])

        with col_graf:
            st.markdown("### Curva de Energía de Interacción DLVO")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                data_dlvo["distancia"],
                data_dlvo["v_vdw"],
                "--",
                color="#e74c3c",
                label="Atracción vdW",
            )
            ax.plot(
                data_dlvo["distancia"],
                data_dlvo["v_elec"],
                "--",
                color="#3498db",
                label="Repulsión Elec.",
            )
            if mecanismo_input in ["Estérica", "Electroestérica"]:
                ax.plot(
                    data_dlvo["distancia"],
                    data_dlvo["v_esterica"],
                    "--",
                    color="#9b59b6",
                    label="Repulsión Estérica",
                )

            ax.plot(
                data_dlvo["distancia"],
                data_dlvo["v_total"],
                "-",
                color="#2c3e50",
                linewidth=2.5,
                label="Interacción Total",
            )
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

            ax.set_xlim(0.4, 12)
            ax.set_ylim(-20, max(40, data_dlvo["barrera"] + 10))
            ax.set_xlabel("Distancia entre partículas d (nm)")
            ax.set_ylabel("Energía de Interacción (kBT)")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

        with col_sim:
            st.markdown("### Estado Visual de la Suspensión")
            fig_sim, ax_sim = plt.subplots(figsize=(5, 4))
            ax_sim.set_facecolor("#f8f9fa")

            np.random.seed(42)
            n_part = 25

            if res["indice"] > 60:
                x_pos = np.random.uniform(1, 9, n_part)
                y_pos = np.random.uniform(1, 9, n_part)
                color_p = "#2ecc71"
            elif res["indice"] > 25:
                x_pos = np.concatenate(
                    [
                        np.random.normal(3, 0.5, 12),
                        np.random.normal(7, 0.5, 13),
                    ]
                )
                y_pos = np.concatenate(
                    [
                        np.random.normal(3, 0.5, 12),
                        np.random.normal(7, 0.5, 13),
                    ]
                )
                color_p = "#f39c12"
            else:
                x_pos = np.random.normal(5, 0.4, n_part)
                y_pos = np.random.normal(5, 0.4, n_part)
                color_p = "#e74c3c"

            # Renderizado de la Doble Capa Eléctrica
            radio_debye = 0.3 + (data_dlvo["debye_nm"] * 0.08)
            for x, y in zip(x_pos, y_pos):
                c_debye = plt.Circle(
                    (x, y), radio_debye, color="#3498db", alpha=0.25
                )
                ax_sim.add_patch(c_debye)

            ax_sim.scatter(
                x_pos, y_pos, color=color_p, s=110, edgecolors="black", zorder=3
            )
            ax_sim.set_xlim(0, 10)
            ax_sim.set_ylim(0, 10)
            ax_sim.set_xticks([])
            ax_sim.set_yticks([])
            ax_sim.set_title(f"Comportamiento: {res['tendencia']}")
            st.pyplot(fig_sim)

with tab_bitacora:
    st.subheader("Interpreta los Resultados y Registra tu Bitácora")

    if st.session_state.ultimo_resultado is None:
        st.warning("Ejecuta un experimento primero para habilitar el registro.")
    else:
        with st.form(f"form_ronda_{st.session_state.ronda}"):
            p1 = st.radio(
                "1. Estado de la suspensión:",
                ["Estable", "Moderadamente Estable", "Inestable"],
            )
            p2 = st.radio(
                "2. Comportamiento de la agregación observada:",
                [
                    "Mínimo Primario (Coagulación fuerte e irreversible)",
                    "Mínimo Secundario (Floculación débil y reversible)",
                    "Sin agregación apreciable (Partículas dispersas)",
                ],
            )
            p3 = st.text_area(
                "3. Hipótesis Científica: Explica qué fuerza domina (vdW vs Repulsión) y qué cambio propones para la siguiente ronda:"
            )

            guardar = st.form_submit_button("💾 Guardar en Bitácora")

            if guardar:
                registro = {
                    "Ronda": st.session_state.ronda,
                    "Zeta (mV)": round(
                        st.session_state.ultimo_resultado["dlvo"][
                            "zeta_efectivo"
                        ],
                        1,
                    ),
                    "Electrolito (mM)": elec_input,
                    "Barrera (kBT)": round(
                        st.session_state.ultimo_resultado["dlvo"]["barrera"], 1
                    ),
                    "Estado": st.session_state.ultimo_resultado["tendencia"],
                    "Índice Estabilidad": round(
                        st.session_state.ultimo_resultado["indice"], 1
                    ),
                    "Hipótesis": p3 if p3 else "Sin justificación",
                }
                st.session_state.bitacora.append(registro)
                st.session_state.puntaje += 15

                if st.session_state.ronda < 6:
                    st.session_state.ronda += 1
                    st.success(
                        "¡Registro guardado exitosamente! Avanzas a la siguiente ronda."
                    )
                else:
                    st.balloons()
                    st.success(
                        "¡Has completado las 6 rondas experimentales! Continúa con el Desafío Final."
                    )

    st.markdown("---")
    st.subheader("📖 BITÁCORA DE EXPERIMENTOS")
    if len(st.session_state.bitacora) > 0:
        st.dataframe(
            pd.DataFrame(st.session_state.bitacora), use_container_width=True
        )

with tab_eval:
    st.subheader("🎯 Desafío Final: Evaluación de Conceptos")
    st.markdown(
        """
    Analiza tres suspensiones hipotéticas:
    * **Suspensión A:** Potencial zeta bajo ($-8\\text{ mV}$) y alta fuerza iónica ($80\\text{ mM}$).
    * **Suspensión B:** Elevado potencial zeta ($-50\\text{ mV}$) y baja fuerza iónica ($1\\text{ mM}$).
    * **Suspensión C:** Potencial zeta moderado ($-15\\text{ mV}$) con adición de polímero hidrofílico (Estabilización Estérica).
    """
    )

    with st.form("form_desafio"):
        orden = st.multiselect(
            "Selecciona las suspensiones ordenadas de MENOR a MAYOR tendencia a la agregación:",
            ["Suspensión A", "Suspensión B", "Suspensión C"],
            default=["Suspensión B", "Suspensión C", "Suspensión A"],
        )
        justificacion = st.text_area(
            "Justifica tu orden basándote en la teoría DLVO y compresión de la doble capa:"
        )

        enviar_eval = st.form_submit_button("🏆 Enviar Evaluación Final")

        if enviar_eval:
            puntaje_total = min(100, st.session_state.puntaje + 25)
            st.markdown(f"## Puntuación Final: **{puntaje_total} / 100 pts**")
            st.info(
                """
            **Retroalimentación Científica:**
            1. **Suspensión A (Mayor Agregación):** La alta concentración de sal comprime la doble capa eléctrica, cancelando la repulsión electrostática. Domina la atracción de van der Waals (Mínimo Primario).
            2. **Suspensión B (Alta Estabilidad DLVO):** La baja fuerza iónica y la alta carga generan una barrera de energía amplia que impide la colisión.
            3. **Suspensión C (Estabilidad Estérica No-DLVO):** El recubrimiento polimérico crea una barrera física por repulsión entrópica independiente de la carga.
            """
            )

with tab_teoria:
    st.subheader("📚 Fundamentos Teóricos de la Teoría DLVO")
    st.markdown(
        """
    La energía total de interacción entre dos partículas coloidales se expresa como:
    
    $$V_{\\text{total}}(d) = V_{\\text{vdW}}(d) + V_{\\text{elec}}(d) + V_{\\text{estérica}}(d)$$
    
    * **Fuerzas de Van der Waals ($V_{\\text{vdW}}$):** Fuerzas de atracción de largo alcance que dependen de la constante de Hamaker del material.
    * **Repulsión Electrostática ($V_{\\text{elec}}$):** Originada por la superposición de las dobles capas eléctricas cargadas.
    * **Compresión de la Doble Capa:** Al aumentar los iones en solución (electrolito), el grosor de la doble capa (longitud de Debye $\\kappa^{-1}$) se reduce, favoreciendo la coagulación.
    """
    )
