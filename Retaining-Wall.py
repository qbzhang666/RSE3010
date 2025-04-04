import streamlit as st
import pandas as pd

# --- Updated Wall Data with Technique included ---
wall_data = pd.DataFrame({
    "Wall Type": [
        "Sheet piles", "King post",
        "Contiguous pile (CFA)", "Contiguous pile (Rotary bored)",
        "Hard/soft secant (CFA)", "Hard/soft secant (Rotary bored)",
        "Hard/firm secant (CFA)", "Hard/firm & hard/hard secant (Cased/CFA)",
        "Diaphragm wall (Rope grab)", "Diaphragm wall (Hydraulic grab)", "Diaphragm wall (Mill)"
    ],
    "Technique": [
        "Driven", "Conventional rotary bored or driven",
        "CFA", "Conventional rotary bored",
        "CFA", "Conventional rotary bored",
        "CFA", "Cased rotary or cased-CFA using thick-wall casing",
        "Rope grab", "Hydraulic grab", "Mill"
    ],
    "Max Cantilever Height (m)": [5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    "Max Propped Height (m)": [10, 10, 16, 25, 16, 25, 16, 25, 30, 40, 50],
    "Groundwater Control - Temp": ["Yes", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
    "Groundwater Control - Perm": ["Yes", "No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes"],
    "Can Provide Interlock": ["No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
    "Economic Rank (1=Best)": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
})

# --- Wall type categories ---
sheet_piles = ["Sheet piles"]
secant_piles = [
    "Hard/soft secant (CFA)", "Hard/soft secant (Rotary bored)",
    "Hard/firm secant (CFA)", "Hard/firm & hard/hard secant (Cased/CFA)"
]
king_post = ["King post"]
diaphragm_walls = [
    "Diaphragm wall (Rope grab)", "Diaphragm wall (Hydraulic grab)", "Diaphragm wall (Mill)"
]

# --- Streamlit UI ---
st.title("🏗️ CIRIA Retaining Wall Selector")
st.markdown("*Based on C760 Figure 3.1, Tables 3.1 and 3.2*")

height = st.slider("Height (m):", 1.0, 50.0, 6.0, 0.5)
retain_water = st.radio("Water Retention Required?", ["Yes", "No"])
permanent = st.radio("Is this a Permanent Wall?", ["Yes", "No"])
support = st.radio("Support Type", ["Cantilever", "Propped"])

# --- Selection Logic ---
def select_wall_type(height, retain_water, permanent, support):
    preferred_types = sheet_piles + secant_piles
    if retain_water == "No" and permanent == "No":
        preferred_types += king_post

    filtered = wall_data[wall_data["Wall Type"].isin(preferred_types)].copy()

    if retain_water == "Yes":
        filtered = filtered[filtered["Can Provide Interlock"] == "Yes"]
        gw_column = "Groundwater Control - Perm" if permanent == "Yes" else "Groundwater Control - Temp"
        filtered = filtered[filtered[gw_column] == "Yes"]
    else:
        gw_column = "Groundwater Control - Temp" if permanent == "No" else "Groundwater Control - Perm"

    height_column = "Max Cantilever Height (m)" if support == "Cantilever" else "Max Propped Height (m)"
    filtered = filtered[filtered[height_column] >= height]

    if not filtered.empty:
        return filtered.sort_values(by="Economic Rank (1=Best)").iloc[0], height_column, gw_column

    fallback = wall_data[wall_data["Wall Type"].isin(diaphragm_walls)].copy()
    fallback = fallback[fallback[height_column] >= height]
    if not fallback.empty:
        return fallback.sort_values(by="Economic Rank (1=Best)").iloc[0], height_column, gw_column

    return None, height_column, gw_column

# --- Format Output ---
def format_result(wall, height_column, gw_column):
    if wall is None:
        return "⚠️ **No suitable wall found.**\n\nRecommendation: Use Diaphragm Wall (Mill) or consider alternative support."

    return f"""
### ✅ Recommended Wall Type: **{wall['Wall Type']}**

**Key Properties:**
- **Technique:** {wall['Technique']}
- Max {height_column.replace('(m)', '')}: **{wall[height_column]} m**
- Interlock Capability: **{wall['Can Provide Interlock']}**
- Groundwater Control ({'Permanent' if permanent == 'Yes' else 'Temporary'}): **{wall[gw_column]}**
- Economic Rank: **{wall['Economic Rank (1=Best)']} / 11**

📌 *Refer to CIRIA Table 3.2 for full data and application limits.*
"""

# --- Display Result ---
wall, height_column, gw_column = select_wall_type(height, retain_water, permanent, support)
st.markdown(format_result(wall, height_column, gw_column))

# --- Reference Table ---
if st.checkbox("Show full CIRIA Table 3.2 reference"):
    st.dataframe(wall_data.style.set_caption("CIRIA C760 – Table 3.2 Summary"))
